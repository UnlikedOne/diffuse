use crate::compute::request_slice;
use crate::registry::{Peer, PeerRegistry};
use crate::worker::pb::Tensor;
use crate::worker::WorkerHandle;
use diffuse_trust::transport::KeyExchange;
use crate::compute::pb::compute_client::ComputeClient;
use tonic::transport::Channel;

pub enum Replica {
    Local {
        worker: WorkerHandle,
        alive: bool,
    },
    Remote {
        compute_endpoint: String,
        host_kx_public: [u8; 32],
        label: String,
        alive: bool,
        client: Option<ComputeClient<Channel>>,
        protocol_version: u32,
    },
    Relayed {
        relay_endpoint: String,
        target_node_id: Vec<u8>,
        host_kx_public: [u8; 32],
        label: String,
        alive: bool,
        client: Option<crate::relay::pb::relay_client::RelayClient<Channel>>,
        protocol_version: u32,
    },
}
impl Replica {
    pub fn is_alive(&self) -> bool {
        match self {
            Replica::Local { alive, .. } => *alive,
            Replica::Remote { alive, .. } => *alive,
            Replica::Relayed { alive, .. } => *alive,
        }
    }
    pub fn set_dead(&mut self) {
        match self {
            Replica::Local { alive, .. } => *alive = false,
            Replica::Remote { alive, .. } => *alive = false,
            Replica::Relayed { alive, .. } => *alive = false,
        }
    }
    pub fn learn_protocol_version(&mut self, version: u32) {
        match self {
            Replica::Local { .. } => {}
            Replica::Remote {
                protocol_version, ..
            } => *protocol_version = version,
            Replica::Relayed {
                protocol_version, ..
            } => *protocol_version = version,
        }
    }
    pub fn protocol_version(&self) -> u32 {
        match self {
            Replica::Local { .. } => crate::registry::WIRE_VERSION,
            Replica::Remote { protocol_version, .. } => *protocol_version,
            Replica::Relayed { protocol_version, .. } => *protocol_version,
        }
    }
    pub fn speaks_current_wire(&self) -> bool {
        self.protocol_version() >= crate::registry::WIRE_VERSION
    }
    pub fn label(&self) -> String {
        match self {
            Replica::Local { worker, .. } => worker.endpoint.clone(),
            Replica::Remote { label, .. } => label.clone(),
            Replica::Relayed { label, .. } => label.clone(),
        }
    }
}

pub struct Stage {
    pub start_layer: u32,
    pub end_layer: u32,
    pub replicas: Vec<Replica>,
    pub last_compute_ms: u64,
    pub last_network_ms: u64,
}

pub struct Orchestrator {
    pub model_id: String,
    pub stages: Vec<Stage>,
    pub spare_endpoints: Vec<String>,
    pub target_replication: usize,
    pub session_kx: std::sync::Arc<KeyExchange>,
    pub last_forward_compute_ms: u64,
    pub last_forward_network_ms: u64,
    pub chain_enabled: bool,
    /// Embedded media for the session in flight. Kept because a broken route is
    /// recovered by replaying the prefix, and for a multimodal prompt the
    /// prefix is not expressible as token ids.
    pub session_prefill: Option<Tensor>,
}

pub const DECODE_TOP_K: u32 = 8;

const TOPK_DTYPE: &str = "topk_i64_f32";

#[derive(Debug, thiserror::Error)]
#[error("the route broke at {endpoint}; the session must be replayed because the surviving replicas hold no KV cache for it")]
pub struct RouteBroken {
    pub endpoint: String,
}

pub fn route_broken(err: &anyhow::Error) -> Option<&RouteBroken> {
    err.downcast_ref::<RouteBroken>()
}

fn stalled_endpoint(err: &anyhow::Error) -> Option<String> {
    let text = format!("{:#}", err);
    let start = text.find("chain stalled at [")? + "chain stalled at [".len();
    let rest = &text[start..];
    let end = rest.find(']')?;
    Some(rest[..end].to_string())
}

fn ids_to_tensor(ids: &[i64]) -> Tensor {
    let mut data = Vec::with_capacity(ids.len() * 8);
    for id in ids {
        data.extend_from_slice(&id.to_le_bytes());
    }
    Tensor {
        shape: vec![1, ids.len() as i64],
        dtype: "int64".to_string(),
        data,
    }
}

fn next_token(out: &Tensor) -> anyhow::Result<i64> {
    if out.dtype == TOPK_DTYPE {
        let k = out.shape.first().copied().unwrap_or(0) as usize;
        if k == 0 || out.data.len() < k * 12 {
            anyhow::bail!("malformed top-k payload: shape {:?}, {} bytes", out.shape, out.data.len());
        }
        let id = i64::from_le_bytes(out.data[0..8].try_into().unwrap());
        return Ok(id);
    }
    argmax_last_token(out)
}

fn argmax_last_token(logits: &Tensor) -> anyhow::Result<i64> {
    crate::compute::argmax_last_row(logits)
}

impl Stage {
    async fn run_with_failover(
        &mut self,
        model_id: &str,
        session_id: &str,
        input: Tensor,
        use_cache: bool,
        session_kx: &KeyExchange,
        top_k: u32,
        accepts_bf16: bool,
    ) -> anyhow::Result<Tensor> {
        let start = self.start_layer;
        let end = self.end_layer;
        let count = self.replicas.len();

        for idx in 0..count {
            if !self.replicas[idx].is_alive() {
                continue;
            }

            let hop_start = std::time::Instant::now();
            let attempt = match &mut self.replicas[idx] {
                Replica::Local { worker, .. } => {
                    let local_start = std::time::Instant::now();
                    worker
                        .run_slice(
                            model_id, start, end, session_id, 0, input.clone(), use_cache, top_k,
                            accepts_bf16,
                        )
                        .await
                        .map(|t| {
                            (
                                t,
                                local_start.elapsed().as_millis() as u64,
                                crate::registry::WIRE_VERSION,
                            )
                        })
                }
                Replica::Remote {
                    compute_endpoint,
                    host_kx_public,
                    client,
                    ..
                } => {
                    match client {
                        Some(c) => {
                            request_slice(
                                c,
                                host_kx_public,
                                session_kx,
                                model_id,
                                start,
                                end,
                                session_id,
                                &input,
                                top_k,
                                accepts_bf16,
                            )
                            .await
                        }
                        None => {
                            match crate::compute::connect_compute(compute_endpoint).await {
                                Ok(mut c) => {
                                    let r = request_slice(
                                        &mut c, host_kx_public, session_kx, model_id,
                                        start, end, session_id, &input, top_k, accepts_bf16,
                                    )
                                    .await;
                                    *client = Some(c);
                                    r
                                }
                                Err(e) => Err(anyhow::anyhow!("connect failed: {}", e)),
                            }
                        }
                    }
                }
                Replica::Relayed {
                    relay_endpoint,
                    target_node_id,
                    host_kx_public,
                    client,
                    ..
                } => {
                    crate::relay::relay_compute(
                        relay_endpoint,
                        client,
                        target_node_id,
                        host_kx_public,
                        session_kx,
                        model_id,
                        start,
                        end,
                        session_id,
                        &input,
                        top_k,
                        accepts_bf16,
                    )
                    .await
                }
            };

            match attempt {
                Ok((out, compute_ms, peer_version)) => {
                    self.replicas[idx].learn_protocol_version(peer_version);
                    let hop_ms = hop_start.elapsed().as_millis() as u64;
                    let network_ms = hop_ms.saturating_sub(compute_ms);
                    tracing::debug!(
                        "hop {}:{} total {}ms = compute {}ms + network/crypto {}ms",
                        start, end, hop_ms, compute_ms, network_ms
                    );
                    self.last_compute_ms = compute_ms;
                    self.last_network_ms = network_ms;
                    return Ok(out);
                }
                Err(e) => {
                    let label = self.replicas[idx].label();
                    self.replicas[idx].set_dead();
                    if use_cache {
                        tracing::warn!(
                            "replica {} for slice {}:{} failed ({}); a standby replica holds no KV cache for this session, so the prefix must be replayed",
                            label, start, end, e
                        );
                        return Err(anyhow::Error::new(RouteBroken { endpoint: label })
                            .context(format!("slice {}:{}", start, end)));
                    }
                    tracing::warn!(
                        "replica {} for slice {}:{} failed ({}), marking dead, trying next",
                        label, start, end, e
                    );
                }
            }
        }

        anyhow::bail!("all replicas dead for slice {}:{}", start, end)
    }

    /// Whether the replica that will consume this stage's input can parse the
    /// current wire format. A peer that predates it must be fed float32.
    pub fn accepts_bf16(&self) -> bool {
        self.replicas
            .iter()
            .find(|r| r.is_alive())
            .map(|r| r.speaks_current_wire())
            .unwrap_or(false)
    }

    pub fn live_replicas(&self) -> usize {
        self.replicas.iter().filter(|r| r.is_alive()).count()
    }

    fn active_endpoints(&self) -> Vec<String> {
        self.replicas.iter().map(|r| r.label()).collect()
    }
}

impl Orchestrator {
    pub async fn health_sweep(&mut self) {
        for stage in self.stages.iter_mut() {
            for replica in stage.replicas.iter_mut() {
                match replica {
                    Replica::Local { worker, alive } => {
                        let ok = worker.health().await.unwrap_or(false);
                        if *alive && !ok {
                            tracing::warn!(
                                "health sweep: local replica {} for slice {}:{} is down",
                                worker.endpoint,
                                stage.start_layer,
                                stage.end_layer
                            );
                        }
                        *alive = ok;
                    }
                    Replica::Remote { .. } => {}
                    Replica::Relayed { .. } => {}
                }
            }
        }
    }

    pub fn coverage(&self) -> Vec<usize> {
        self.stages.iter().map(|s| s.live_replicas()).collect()
    }

    pub fn is_servable(&self) -> bool {
        self.stages.iter().all(|s| s.live_replicas() > 0)
    }

    fn endpoint_in_use(&self, ep: &str) -> bool {
        self.stages
            .iter()
            .any(|s| s.active_endpoints().iter().any(|e| e == ep))
    }

    pub async fn repair(&mut self) -> usize {
        let mut repaired = 0;
        let model_id = self.model_id.clone();
        let target = self.target_replication;
        let spares = self.spare_endpoints.clone();

        for stage_idx in 0..self.stages.len() {
            let start = self.stages[stage_idx].start_layer;
            let end = self.stages[stage_idx].end_layer;

            while self.stages[stage_idx].live_replicas() < target {
                let spare = spares
                    .iter()
                    .find(|ep| !self.endpoint_in_use(ep))
                    .cloned();

                let Some(ep) = spare else {
                    tracing::warn!(
                        "repair: slice {}:{} under-replicated ({} < {}), no spare available",
                        start,
                        end,
                        self.stages[stage_idx].live_replicas(),
                        target
                    );
                    break;
                };

                tracing::info!("repair: recruiting spare {} for slice {}:{}", ep, start, end);

                match WorkerHandle::connect(ep.clone()).await {
                    Ok(mut worker) => match worker.load_slice(&model_id, start, end, "").await {
                        Ok(_) => {
                            self.stages[stage_idx]
                                .replicas
                                .push(Replica::Local { worker, alive: true });
                            repaired += 1;
                            tracing::info!(
                                "repair: slice {}:{} restored to {} replicas",
                                start,
                                end,
                                self.stages[stage_idx].live_replicas()
                            );
                        }
                        Err(e) => {
                            tracing::warn!("repair: spare {} failed to load slice: {}", ep, e);
                            break;
                        }
                    },
                    Err(e) => {
                        tracing::warn!("repair: could not connect to spare {}: {}", ep, e);
                        break;
                    }
                }
            }
        }
        repaired
    }

    fn build_route(&self, top_k: u32) -> Option<Vec<(usize, crate::compute::pb::Hop)>> {
        let last = self.stages.len().saturating_sub(1);
        let mut hops = Vec::with_capacity(self.stages.len());
        for (i, stage) in self.stages.iter().enumerate() {
            let idx = stage.replicas.iter().position(|r| {
                r.is_alive() && matches!(r, Replica::Remote { .. }) && r.speaks_current_wire()
            })?;
            let Replica::Remote {
                compute_endpoint,
                host_kx_public,
                ..
            } = &stage.replicas[idx]
            else {
                return None;
            };
            hops.push((
                idx,
                crate::compute::pb::Hop {
                    compute_endpoint: compute_endpoint.clone(),
                    kx_public: host_kx_public.to_vec(),
                    start_layer: stage.start_layer,
                    end_layer: stage.end_layer,
                    top_k: if i == last { top_k } else { 0 },
                    accepts_bf16: true,
                },
            ));
        }
        Some(hops)
    }

    async fn forward_chained(
        &mut self,
        input: &Tensor,
        session_id: &str,
        top_k: u32,
    ) -> anyhow::Result<Option<Tensor>> {
        if self.stages.len() < 2 {
            return Ok(None);
        }
        let Some(route) = self.build_route(top_k) else {
            return Ok(None);
        };

        let head_idx = route[0].0;
        let head_hop = route[0].1.clone();
        let head_accepts_bf16 = route.get(1).map(|(_, h)| h.accepts_bf16).unwrap_or(false);
        let rest: Vec<_> = route[1..].iter().map(|(_, hop)| hop.clone()).collect();
        let head_kx: [u8; 32] = match head_hop.kx_public.as_slice().try_into() {
            Ok(k) => k,
            Err(_) => return Ok(None),
        };

        let model_id = self.model_id.clone();
        let session_kx = self.session_kx.clone();
        let start = std::time::Instant::now();

        let Replica::Remote {
            compute_endpoint,
            client,
            ..
        } = &mut self.stages[0].replicas[head_idx]
        else {
            return Ok(None);
        };
        if client.is_none() {
            *client = crate::compute::connect_compute(compute_endpoint).await.ok();
        }
        let Some(head_client) = client else {
            return Ok(None);
        };

        let outcome = crate::compute::request_slice_chained(
            head_client,
            &head_kx,
            &session_kx,
            &model_id,
            head_hop.start_layer,
            head_hop.end_layer,
            session_id,
            input,
            head_hop.top_k,
            rest,
            head_accepts_bf16,
        )
        .await;

        match outcome {
            Ok((out, compute_ms, peer_version)) => {
                if let Some(r) = self.stages[0].replicas.get_mut(head_idx) {
                    r.learn_protocol_version(peer_version);
                }
                let elapsed = start.elapsed().as_millis() as u64;
                self.last_forward_compute_ms = compute_ms;
                self.last_forward_network_ms = elapsed.saturating_sub(compute_ms);
                tracing::debug!(
                    "chained forward over {} stages: {}ms total, {}ms compute",
                    self.stages.len(),
                    elapsed,
                    compute_ms
                );
                Ok(Some(out))
            }
            Err(e) => {
                let endpoint = stalled_endpoint(&e).unwrap_or_else(|| head_hop.compute_endpoint.clone());
                self.mark_dead_by_endpoint(&endpoint);
                Err(anyhow::Error::new(RouteBroken { endpoint }).context(e.to_string()))
            }
        }
    }

    fn mark_dead_by_endpoint(&mut self, endpoint: &str) {
        for stage in self.stages.iter_mut() {
            for replica in stage.replicas.iter_mut() {
                if let Replica::Remote {
                    compute_endpoint, ..
                } = replica
                {
                    if compute_endpoint == endpoint {
                        replica.set_dead();
                    }
                }
            }
        }
    }

    pub async fn forward(
        &mut self,
        ids: &[i64],
        session_id: &str,
        use_cache: bool,
        top_k: u32,
    ) -> anyhow::Result<Tensor> {
        self.forward_tensor(ids_to_tensor(ids), session_id, use_cache, top_k)
            .await
    }

    /// Push an already-embedded sequence through the pipeline.
    ///
    /// Media never becomes token ids: the client turns a picture or a recording
    /// into hidden states itself, and those enter the route in place of a
    /// prompt, which the first stage passes straight to its layers.
    pub async fn forward_tensor(
        &mut self,
        input: Tensor,
        session_id: &str,
        use_cache: bool,
        top_k: u32,
    ) -> anyhow::Result<Tensor> {
        if self.chain_enabled && use_cache {
            match self.forward_chained(&input, session_id, top_k).await {
                Ok(Some(out)) => return Ok(out),
                Ok(None) => {}
                Err(e) => return Err(e),
            }
        }
        let model_id = self.model_id.clone();
        let session_kx = self.session_kx.clone();
        let last_stage = self.stages.len().saturating_sub(1);
        // Each stage must emit a format the *next* stage can parse. The last
        // stage answers the client, which either asked for top-k or accepts the
        // float32 logits every version understands.
        let consumer_accepts: Vec<bool> = (0..self.stages.len())
            .map(|i| {
                self.stages
                    .get(i + 1)
                    .map(|next| next.accepts_bf16())
                    .unwrap_or(false)
            })
            .collect();
        let mut t = input;
        let mut compute_sum = 0u64;
        let mut network_sum = 0u64;
        for (idx, stage) in self.stages.iter_mut().enumerate() {
            let stage_top_k = if idx == last_stage { top_k } else { 0 };
            t = stage
                .run_with_failover(
                    &model_id,
                    session_id,
                    t,
                    use_cache,
                    &session_kx,
                    stage_top_k,
                    consumer_accepts[idx],
                )
                .await?;
            compute_sum += stage.last_compute_ms;
            network_sum += stage.last_network_ms;
        }
        self.last_forward_compute_ms = compute_sum;
        self.last_forward_network_ms = network_sum;
        Ok(t)
    }

    async fn forward_step(
        &mut self,
        step_ids: &[i64],
        full_ids: &[i64],
        session_id: &str,
        top_k: u32,
    ) -> anyhow::Result<Tensor> {
        const MAX_REPLAYS: usize = 3;
        let mut replays = 0;
        let mut ids = step_ids;
        loop {
            match self.forward(ids, session_id, true, top_k).await {
                Ok(out) => return Ok(out),
                Err(e) if route_broken(&e).is_some() && replays < MAX_REPLAYS => {
                    replays += 1;
                    tracing::warn!(
                        "{:#}; replaying {} tokens on the surviving replicas (attempt {}/{})",
                        e,
                        full_ids.len(),
                        replays,
                        MAX_REPLAYS
                    );
                    self.clear_session(session_id).await;
                    // Media has to go back through first: the surviving nodes
                    // hold no cache, and the picture is not in the token ids.
                    if let Some(prefill) = self.session_prefill.clone() {
                        self.forward_tensor(prefill, session_id, true, 0).await?;
                    }
                    ids = full_ids;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Generate from an already-embedded prompt, streaming each token out.
    ///
    /// The prefill carries whatever the client embedded, text and media alike;
    /// decoding then continues on token ids like any other session.
    pub async fn generate_from_embeddings<F>(
        &mut self,
        prefill: Tensor,
        max_new_tokens: usize,
        session_id: &str,
        eos_id: Option<i64>,
        mut on_token: F,
    ) -> anyhow::Result<Vec<i64>>
    where
        F: FnMut(i64),
    {
        self.session_prefill = Some(prefill.clone());
        let logits = self
            .forward_tensor(prefill, session_id, true, DECODE_TOP_K)
            .await?;
        let mut next = next_token(&logits)?;
        let mut produced = vec![next];
        on_token(next);
        if Some(next) == eos_id {
            self.session_prefill = None;
            return Ok(produced);
        }

        for _ in 1..max_new_tokens {
            let logits = self
                .forward_step(&[next], &produced, session_id, DECODE_TOP_K)
                .await?;
            next = next_token(&logits)?;
            if Some(next) == eos_id {
                break;
            }
            produced.push(next);
            on_token(next);
        }
        self.session_prefill = None;
        Ok(produced)
    }

    pub async fn generate(
        &mut self,
        prompt_ids: &[i64],
        max_new_tokens: usize,
        session_id: &str,
        eos_id: Option<i64>,
    ) -> anyhow::Result<Vec<i64>> {
        let gen_start = std::time::Instant::now();
        let mut ids = prompt_ids.to_vec();

        let prefill_start = std::time::Instant::now();
        let logits = self
            .forward_step(&ids, &ids, session_id, DECODE_TOP_K)
            .await?;
        let prefill_ms = prefill_start.elapsed().as_millis();

        let mut next = next_token(&logits)?;
        ids.push(next);
        if Some(next) == eos_id {
            return Ok(ids);
        }

        let mut decode_total = std::time::Duration::ZERO;
        let mut compute_total = 0u64;
        let mut network_total = 0u64;
        let mut token_count = 0usize;
        for step in 1..max_new_tokens {
            let tok_start = std::time::Instant::now();
            let logits = self
                .forward_step(&[next], &ids, session_id, DECODE_TOP_K)
                .await?;
            decode_total += tok_start.elapsed();
            compute_total += self.last_forward_compute_ms;
            network_total += self.last_forward_network_ms;
            token_count += 1;

            next = next_token(&logits)?;
            ids.push(next);
            if Some(next) == eos_id {
                break;
            }
            if step % 10 == 0 {
                tracing::info!("step {}, live replicas per stage: {:?}", step, self.coverage());
            }
        }

        let total_ms = gen_start.elapsed().as_millis();
        let n = token_count.max(1) as u128;
        tracing::info!(
            "generation: {} tokens in {}ms | prefill {}ms | decode avg {}ms/token = compute {}ms + network/crypto {}ms | {} stages",
            token_count + 1,
            total_ms,
            prefill_ms,
            decode_total.as_millis() / n,
            compute_total as u128 / n,
            network_total as u128 / n,
            self.stages.len()
        );

        Ok(ids)
    }

    pub async fn clear_session(&mut self, session_id: &str) {
        for stage in &mut self.stages {
            for replica in &mut stage.replicas {
                match replica {
                    Replica::Local { worker, .. } => {
                        let _ = worker.clear_session(session_id).await;
                    }
                    Replica::Remote { client, compute_endpoint, .. } => {
                        if client.is_none() {
                            *client = ComputeClient::connect(compute_endpoint.clone())
                                .await
                                .ok()
                                .map(|c| {
                                    c.max_decoding_message_size(128 * 1024 * 1024)
                                        .max_encoding_message_size(128 * 1024 * 1024)
                                });
                        }
                        if let Some(c) = client {
                            let _ = c
                                .clear_session(crate::compute::pb::ClearSessionRequest {
                                    session_id: session_id.to_string(),
                                })
                                .await;
                        }
                    }
                    Replica::Relayed { .. } => {}
                }
            }
        }
    }

    pub async fn generate_streaming<F>(
        &mut self,
        prompt_ids: &[i64],
        max_new_tokens: usize,
        session_id: &str,
        eos_id: Option<i64>,
        mut on_token: F,
    ) -> anyhow::Result<Vec<i64>>
    where
        F: FnMut(i64),
    {
        let mut ids = prompt_ids.to_vec();

        let logits = self
            .forward_step(&ids, &ids, session_id, DECODE_TOP_K)
            .await?;
        let mut next = next_token(&logits)?;
        ids.push(next);
        on_token(next);
        if Some(next) == eos_id {
            return Ok(ids);
        }

        for _ in 1..max_new_tokens {
            let logits = self
                .forward_step(&[next], &ids, session_id, DECODE_TOP_K)
                .await?;
            next = next_token(&logits)?;
            ids.push(next);
            if Some(next) == eos_id {
                break;
            }
            on_token(next);
        }
        Ok(ids)
    }

    pub async fn encode_via_any(
        &mut self,
        text: &str,
        chat_template: bool,
    ) -> anyhow::Result<(Vec<i64>, i64)> {
        for stage in self.stages.iter_mut() {
            for replica in stage.replicas.iter_mut() {
                if let Replica::Local { worker, alive } = replica {
                    if *alive {
                        if let Ok(result) = worker.encode(text, chat_template).await {
                            return Ok(result);
                        }
                    }
                }
            }
        }
        anyhow::bail!("no local replica could encode (consumer needs a local tokenizer worker)")
    }

    pub async fn decode_via_any(
        &mut self,
        token_ids: &[i64],
        skip_special: bool,
    ) -> anyhow::Result<String> {
        for stage in self.stages.iter_mut() {
            for replica in stage.replicas.iter_mut() {
                if let Replica::Local { worker, alive } = replica {
                    if *alive {
                        if let Ok(text) = worker.decode(token_ids, skip_special).await {
                            return Ok(text);
                        }
                    }
                }
            }
        }
        anyhow::bail!("no local replica could decode (consumer needs a local tokenizer worker)")
    }
}

pub async fn build_from_registry(
    model_id: &str,
    registry: &PeerRegistry,
    target_replication: usize,
    spare_endpoints: Vec<String>,
    relay_sentinel: Option<String>,
) -> anyhow::Result<Orchestrator> {
    let mut slices: Vec<(u32, u32)> = registry
        .all()
        .iter()
        .filter(|p| p.model_id == model_id)
        .map(|p| (p.start_layer, p.end_layer))
        .collect();
    slices.sort();
    slices.dedup();
    if slices.is_empty() {
        anyhow::bail!("no peers in registry serve model {}", model_id);
    }

    // Refuse to build a route through a model the network only partially holds.
    // Without this, a set of slices that stops short of `total_layers` (or leaves
    // an interior hole) would yield an orchestrator that forwards activations
    // into a dead end and returns garbage. Fail here, naming the missing ranges.
    if let Some(cap) = crate::capacity::analyze(registry)
        .into_iter()
        .find(|c| c.model_id == model_id)
    {
        let gaps = cap.coverage_gaps();
        if !gaps.is_empty() {
            let missing = gaps
                .iter()
                .map(|(start, end)| format!("{}:{}", start, end))
                .collect::<Vec<_>>()
                .join(", ");
            anyhow::bail!(
                "model {} is incomplete: no peer serves layers {} (of {} total)",
                model_id,
                missing,
                cap.total_layers
            );
        }
    }
    let mut stages = Vec::new();
    for (start, end) in slices {
        let peers: Vec<Peer> = registry.replicas_for_slice(model_id, start, end);
        let mut replicas = Vec::new();
        for peer in peers {
            let kx: [u8; 32] = match peer.kx_public.as_slice().try_into() {
                Ok(k) => k,
                Err(_) => {
                    tracing::warn!("route: peer {} has invalid kx_public, skipping", peer.daemon_endpoint);
                    continue;
                }
            };

            if peer.reachable {
                let compute_endpoint = daemon_to_compute_endpoint(&peer.daemon_endpoint);
                tracing::info!(
                    "route: direct encrypted replica {} for slice {}:{}",
                    compute_endpoint,
                    start,
                    end
                );
                let client = match crate::compute::connect_compute(&compute_endpoint).await {
                    Ok(c) => Some(c),
                    Err(e) => {
                        tracing::warn!("could not pre-connect to {}: {}", compute_endpoint, e);
                        None
                    }
                };
                replicas.push(Replica::Remote {
                    compute_endpoint,
                    host_kx_public: kx,
                    label: peer.daemon_endpoint.clone(),
                    alive: true,
                    client,
                    protocol_version: 0,
                });
            } else {
                match &relay_sentinel {
                    Some(relay) => {
                        tracing::info!(
                            "route: relayed replica {} via {} for slice {}:{}",
                            peer.daemon_endpoint,
                            relay,
                            start,
                            end
                        );
                        replicas.push(Replica::Relayed {
                            relay_endpoint: relay.clone(),
                            target_node_id: peer.node_id.clone(),
                            host_kx_public: kx,
                            label: format!("{} (relayed)", peer.daemon_endpoint),
                            alive: true,
                            client: None,
                            protocol_version: 0,
                        });
                    }
                    None => {
                        tracing::warn!(
                            "route: peer {} is behind NAT but no relay sentinel is known, skipping",
                            peer.daemon_endpoint
                        );
                        continue;
                    }
                }
            }
        }
        if replicas.is_empty() {
            anyhow::bail!("no reachable replica for slice {}:{}", start, end);
        }
        stages.push(Stage {
            start_layer: start,
            end_layer: end,
            replicas,
            last_compute_ms: 0,
            last_network_ms: 0,
        });
    }
    Ok(Orchestrator {
        model_id: model_id.to_string(),
        stages,
        spare_endpoints,
        target_replication,
        session_kx: std::sync::Arc::new(KeyExchange::generate()),
        last_forward_compute_ms: 0,
        last_forward_network_ms: 0,
        chain_enabled: true,
        session_prefill: None,
    })
}

fn daemon_to_compute_endpoint(daemon_endpoint: &str) -> String {
    if let Some((host, port_str)) = daemon_endpoint.rsplit_once(':') {
        if let Ok(port) = port_str.parse::<u16>() {
            return format!("{}:{}", host, port + 1000);
        }
    }
    daemon_endpoint.to_string()
}