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
    },
    Relayed {
        relay_endpoint: String,
        target_node_id: Vec<u8>,
        host_kx_public: [u8; 32],
        label: String,
        alive: bool,
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
    pub session_kx: KeyExchange,
    pub last_forward_compute_ms: u64,
    pub last_forward_network_ms: u64,
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

fn argmax_last_token(logits: &Tensor) -> anyhow::Result<i64> {
    if logits.shape.len() != 3 {
        anyhow::bail!("expected 3D logits, got shape {:?}", logits.shape);
    }
    let seq = logits.shape[1] as usize;
    let vocab = logits.shape[2] as usize;
    let floats: &[f32] = bytemuck::cast_slice(&logits.data);
    let offset = (seq - 1) * vocab;
    let row = &floats[offset..offset + vocab];

    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    Ok(best_idx as i64)
}

impl Stage {
    async fn run_with_failover(
        &mut self,
        model_id: &str,
        session_id: &str,
        input: Tensor,
        use_cache: bool,
        session_kx: &KeyExchange,
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
                    worker
                        .run_slice(model_id, start, end, session_id, 0, input.clone(), use_cache)
                        .await
                        .map(|t| (t, 0u64))
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
                            )
                            .await
                        }
                        None => {
                            match crate::compute::connect_compute(compute_endpoint).await {
                                Ok(mut c) => {
                                    let r = request_slice(
                                        &mut c, host_kx_public, session_kx, model_id,
                                        start, end, session_id, &input,
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
                    ..
                } => {
                    crate::relay::relay_compute(
                        relay_endpoint,
                        target_node_id,
                        host_kx_public,
                        session_kx,
                        model_id,
                        start,
                        end,
                        session_id,
                        &input,
                    )
                    .await
                }
            };

            match attempt {
                Ok((out, compute_ms)) => {
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
                    tracing::warn!(
                        "replica {} for slice {}:{} failed ({}), marking dead, trying next",
                        self.replicas[idx].label(),
                        start,
                        end,
                        e
                    );
                    self.replicas[idx].set_dead();
                }
            }
        }

        anyhow::bail!("all replicas dead for slice {}:{}", start, end)
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

    pub async fn forward(
        &mut self,
        ids: &[i64],
        session_id: &str,
        use_cache: bool,
    ) -> anyhow::Result<Tensor> {
        let mut tensor = ids_to_tensor(ids);
        let model_id = self.model_id.clone();
        let session_kx = std::mem::replace(&mut self.session_kx, KeyExchange::generate());
        let result = async {
            let mut t = tensor.clone();
            let mut compute_sum = 0u64;
            let mut network_sum = 0u64;
            for stage in self.stages.iter_mut() {
                t = stage
                    .run_with_failover(&model_id, session_id, t, use_cache, &session_kx)
                    .await?;
                compute_sum += stage.last_compute_ms;
                network_sum += stage.last_network_ms;
            }
            self.last_forward_compute_ms = compute_sum;
            self.last_forward_network_ms = network_sum;
            Ok::<Tensor, anyhow::Error>(t)
        }
        .await;
        self.session_kx = session_kx;
        let _ = &mut tensor;
        result
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
        let logits = self.forward(&ids, session_id, true).await?;
        let prefill_ms = prefill_start.elapsed().as_millis();

        let mut next = argmax_last_token(&logits)?;
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
            let logits = self.forward(&[next], session_id, true).await?;
            decode_total += tok_start.elapsed();
            compute_total += self.last_forward_compute_ms;
            network_total += self.last_forward_network_ms;
            token_count += 1;

            next = argmax_last_token(&logits)?;
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

        let logits = self.forward(&ids, session_id, true).await?;
        let mut next = argmax_last_token(&logits)?;
        ids.push(next);
        on_token(next);
        if Some(next) == eos_id {
            return Ok(ids);
        }

        for _ in 1..max_new_tokens {
            let logits = self.forward(&[next], session_id, true).await?;
            next = argmax_last_token(&logits)?;
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
        session_kx: KeyExchange::generate(),
        last_forward_compute_ms: 0,
        last_forward_network_ms: 0,
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