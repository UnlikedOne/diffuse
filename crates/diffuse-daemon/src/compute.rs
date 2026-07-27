pub mod pb {
    tonic::include_proto!("diffuse.compute");
}

use std::sync::Arc;

use tokio::sync::Mutex;
use tonic::{Request, Response, Status};

use crate::worker::pb::Tensor;
use crate::worker::WorkerHandle;
use diffuse_trust::transport::{decrypt, encrypt, KeyExchange};

use pb::compute_client::ComputeClient;
use pb::compute_server::Compute;
use pb::{ComputeRequest, ComputeResponse, ClearSessionRequest, ClearSessionResponse};

#[derive(Clone, Default)]
pub struct Patch {
    pub offset: u64,
    pub sequence: u64,
    pub branch: String,
    pub layout: String,
    pub arguments: Vec<(u32, Tensor)>,
}

pub fn arguments_to_bytes(arguments: &[(u32, Tensor)]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(arguments.len() as u32).to_le_bytes());
    for (index, tensor) in arguments {
        buf.extend_from_slice(&index.to_le_bytes());
        let body = tensor_to_bytes(tensor);
        buf.extend_from_slice(&(body.len() as u64).to_le_bytes());
        buf.extend_from_slice(&body);
    }
    buf
}

pub fn bytes_to_arguments(bytes: &[u8]) -> anyhow::Result<Vec<(u32, Tensor)>> {
    if bytes.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    let mut off = 0usize;
    let count = u32::from_le_bytes(bytes[off..off + 4].try_into()?) as usize;
    off += 4;
    for _ in 0..count {
        let index = u32::from_le_bytes(bytes[off..off + 4].try_into()?);
        off += 4;
        let len = u64::from_le_bytes(bytes[off..off + 8].try_into()?) as usize;
        off += 8;
        out.push((index, bytes_to_tensor(&bytes[off..off + len])?));
        off += len;
    }
    Ok(out)
}

fn tensor_to_bytes(t: &Tensor) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());
    for d in &t.shape {
        buf.extend_from_slice(&d.to_le_bytes());
    }
    let dtype = t.dtype.as_bytes();
    buf.extend_from_slice(&(dtype.len() as u32).to_le_bytes());
    buf.extend_from_slice(dtype);
    buf.extend_from_slice(&t.data);
    buf
}

fn bytes_to_tensor(b: &[u8]) -> anyhow::Result<Tensor> {
    let mut off = 0;
    let read_u32 = |b: &[u8], off: &mut usize| -> u32 {
        let v = u32::from_le_bytes(b[*off..*off + 4].try_into().unwrap());
        *off += 4;
        v
    };
    let ndim = read_u32(b, &mut off) as usize;
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        let d = i64::from_le_bytes(b[off..off + 8].try_into().unwrap());
        off += 8;
        shape.push(d);
    }
    let dlen = read_u32(b, &mut off) as usize;
    let dtype = String::from_utf8(b[off..off + dlen].to_vec())?;
    off += dlen;
    let data = b[off..].to_vec();
    Ok(Tensor { shape, dtype, data })
}

pub fn argmax_last_row(logits: &Tensor) -> anyhow::Result<i64> {
    if logits.shape.len() != 3 {
        anyhow::bail!("expected 3D logits, got shape {:?}", logits.shape);
    }
    let seq = logits.shape[1] as usize;
    let vocab = logits.shape[2] as usize;
    let offset = (seq - 1) * vocab;

    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    let mut consider = |i: usize, v: f32| {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    };

    match logits.dtype.as_str() {
        "float32" => {
            let floats: &[f32] = bytemuck::cast_slice(&logits.data);
            if floats.len() < offset + vocab {
                anyhow::bail!("logits payload too short for shape {:?}", logits.shape);
            }
            for (i, &v) in floats[offset..offset + vocab].iter().enumerate() {
                consider(i, v);
            }
        }
        "bfloat16" => {
            let raw: &[u16] = bytemuck::cast_slice(&logits.data);
            if raw.len() < offset + vocab {
                anyhow::bail!("logits payload too short for shape {:?}", logits.shape);
            }
            for (i, &v) in raw[offset..offset + vocab].iter().enumerate() {
                consider(i, f32::from_bits((v as u32) << 16));
            }
        }
        other => anyhow::bail!("unsupported logits dtype {}", other),
    }
    Ok(best_idx as i64)
}

fn decode_positions(secret: &[u8; 32], payload: &[u8]) -> anyhow::Result<Option<Tensor>> {
    if payload.is_empty() {
        return Ok(None);
    }
    Ok(Some(bytes_to_tensor(&decrypt(secret, payload)?)?))
}

pub fn tensor_to_bytes_pub(t: &Tensor) -> Vec<u8> {
    tensor_to_bytes(t)
}
pub fn bytes_to_tensor_pub(b: &[u8]) -> anyhow::Result<Tensor> {
    bytes_to_tensor(b)
}

#[derive(Clone, Default)]
pub struct HopClients {
    clients: Arc<Mutex<std::collections::HashMap<String, ComputeClient<tonic::transport::Channel>>>>,
}

impl HopClients {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn get(
        &self,
        endpoint: &str,
    ) -> anyhow::Result<ComputeClient<tonic::transport::Channel>> {
        if let Some(client) = self.clients.lock().await.get(endpoint) {
            return Ok(client.clone());
        }
        let client = connect_compute(endpoint).await?;
        self.clients
            .lock()
            .await
            .insert(endpoint.to_string(), client.clone());
        Ok(client)
    }

    pub async fn forget(&self, endpoint: &str) {
        self.clients.lock().await.remove(endpoint);
    }
}

pub struct ComputeService {
    pub identity_kx: Arc<KeyExchange>,
    pub worker: Arc<Mutex<WorkerHandle>>,
    pub model_id: String,
    pub hops: HopClients,
}

#[tonic::async_trait]
impl Compute for ComputeService {
    async fn run_slice(
        &self,
        request: Request<ComputeRequest>,
    ) -> Result<Response<ComputeResponse>, Status> {
        let req = request.into_inner();
        let resp = process_chained_request(&self.identity_kx, &self.worker, &req, &self.hops)
            .await
            .map_err(|e| Status::internal(format!("compute failed: {}", e)))?;
        Ok(Response::new(resp))
    }

    async fn clear_session(
        &self,
        request: Request<ClearSessionRequest>,
    ) -> Result<Response<ClearSessionResponse>, Status> {
        let session_id = request.into_inner().session_id;
        let mut worker = self.worker.lock().await.clone();
        let _ = worker.clear_session(&session_id).await;
        Ok(Response::new(ClearSessionResponse { ok: true }))
    }
}

pub async fn connect_compute(
    endpoint: &str,
) -> anyhow::Result<ComputeClient<tonic::transport::Channel>> {
    let client = ComputeClient::connect(endpoint.to_string())
        .await?
        .max_decoding_message_size(128 * 1024 * 1024)
        .max_encoding_message_size(128 * 1024 * 1024);
    Ok(client)
}

pub async fn request_slice(
    client: &mut ComputeClient<tonic::transport::Channel>,
    host_kx_public: &[u8; 32],
    my_kx: &KeyExchange,
    model_id: &str,
    start: u32,
    end: u32,
    session_id: &str,
    activations: &Tensor,
    top_k: u32,
    accepts_bf16: bool,
    position_ids: Option<&Tensor>,
    encoder_memory: Option<&Tensor>,
    patch: Option<&Patch>,
    draw: crate::worker::Draw,
) -> anyhow::Result<(Tensor, u64, u32)> {
    let secret = my_kx.shared_secret(host_kx_public);
    let plain = tensor_to_bytes(activations);
    let encrypted = encrypt(&secret, &plain)?;
    let encrypted_position_ids = match position_ids {
        Some(p) => encrypt(&secret, &tensor_to_bytes(p))?,
        None => Vec::new(),
    };
    let encrypted_encoder_memory = match encoder_memory {
        Some(m) => encrypt(&secret, &tensor_to_bytes(m))?,
        None => Vec::new(),
    };
    let encrypted_arguments = match patch {
        Some(p) if !p.arguments.is_empty() => {
            encrypt(&secret, &arguments_to_bytes(&p.arguments))?
        }
        _ => Vec::new(),
    };
    let response = client
        .run_slice(ComputeRequest {
            requester_kx_public: my_kx.public_bytes().to_vec(),
            model_id: model_id.to_string(),
            start_layer: start,
            end_layer: end,
            session_id: session_id.to_string(),
            encrypted_activations: encrypted,
            top_k,
            route: Vec::new(),
            accepts_bf16,
            encrypted_position_ids,
            encrypted_encoder_memory,
            patch_offset: patch.map(|p| p.offset).unwrap_or(0),
            patch_sequence: patch.map(|p| p.sequence).unwrap_or(0),
            branch: patch.map(|p| p.branch.clone()).unwrap_or_default(),
            layout: patch.map(|p| p.layout.clone()).unwrap_or_default(),
            encrypted_arguments,
            guidance: draw.guidance,
            sample: draw.sample,
            temperature: draw.temperature,
            top_p: draw.top_p,
            seed: draw.seed,
        })
        .await?
        .into_inner();
    if !response.ok {
        anyhow::bail!("remote compute failed: {}", response.error);
    }
    let plain_out = decrypt(&secret, &response.encrypted_activations)?;
    let tensor = bytes_to_tensor(&plain_out)?;
    Ok((tensor, response.compute_ms, response.protocol_version))
}

pub async fn request_slice_chained(
    client: &mut ComputeClient<tonic::transport::Channel>,
    host_kx_public: &[u8; 32],
    my_kx: &KeyExchange,
    model_id: &str,
    start: u32,
    end: u32,
    session_id: &str,
    activations: &Tensor,
    top_k: u32,
    route: Vec<pb::Hop>,
    accepts_bf16: bool,
    position_ids: Option<&Tensor>,
    encoder_memory: Option<&Tensor>,
    patch: Option<&Patch>,
    draw: crate::worker::Draw,
) -> anyhow::Result<(Tensor, u64, u32)> {
    let secret = my_kx.shared_secret(host_kx_public);
    let plain = tensor_to_bytes(activations);
    let encrypted = encrypt(&secret, &plain)?;
    let encrypted_position_ids = match position_ids {
        Some(p) => encrypt(&secret, &tensor_to_bytes(p))?,
        None => Vec::new(),
    };
    let encrypted_encoder_memory = match encoder_memory {
        Some(m) => encrypt(&secret, &tensor_to_bytes(m))?,
        None => Vec::new(),
    };
    let encrypted_arguments = match patch {
        Some(p) if !p.arguments.is_empty() => {
            encrypt(&secret, &arguments_to_bytes(&p.arguments))?
        }
        _ => Vec::new(),
    };
    let response = client
        .run_slice(ComputeRequest {
            requester_kx_public: my_kx.public_bytes().to_vec(),
            model_id: model_id.to_string(),
            start_layer: start,
            end_layer: end,
            session_id: session_id.to_string(),
            encrypted_activations: encrypted,
            top_k,
            route,
            accepts_bf16,
            encrypted_position_ids,
            encrypted_encoder_memory,
            patch_offset: patch.map(|p| p.offset).unwrap_or(0),
            patch_sequence: patch.map(|p| p.sequence).unwrap_or(0),
            branch: patch.map(|p| p.branch.clone()).unwrap_or_default(),
            layout: patch.map(|p| p.layout.clone()).unwrap_or_default(),
            encrypted_arguments,
            guidance: draw.guidance,
            sample: draw.sample,
            temperature: draw.temperature,
            top_p: draw.top_p,
            seed: draw.seed,
        })
        .await?
        .into_inner();
    if !response.ok {
        anyhow::bail!("chained compute failed: {}", response.error);
    }
    let plain_out = decrypt(&secret, &response.encrypted_activations)?;
    let tensor = bytes_to_tensor(&plain_out)?;
    Ok((tensor, response.compute_ms, response.protocol_version))
}

pub fn spawn_compute_server(
    listen_addr: std::net::SocketAddr,
    identity_kx: Arc<KeyExchange>,
    worker: Arc<Mutex<WorkerHandle>>,
    model_id: String,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let service = ComputeService {
            identity_kx,
            worker,
            model_id,
            hops: HopClients::new(),
        };
        let server = tonic::transport::Server::builder()
            .add_service(
                pb::compute_server::ComputeServer::new(service)
                    .max_decoding_message_size(128 * 1024 * 1024)
                    .max_encoding_message_size(128 * 1024 * 1024),
            )
            .serve(listen_addr);
        if let Err(e) = server.await {
            tracing::error!("compute server error: {}", e);
        }
    })
}

pub fn decode_patch(
    secret: &[u8; 32],
    req: &ComputeRequest,
) -> anyhow::Result<Option<Patch>> {
    if req.patch_sequence == 0 {
        return Ok(None);
    }
    let arguments = if req.encrypted_arguments.is_empty() {
        Vec::new()
    } else {
        bytes_to_arguments(&decrypt(secret, &req.encrypted_arguments)?)?
    };
    Ok(Some(Patch {
        offset: req.patch_offset,
        sequence: req.patch_sequence,
        branch: req.branch.clone(),
        layout: req.layout.clone(),
        arguments,
    }))
}

pub async fn process_compute_request(
    identity_kx: &KeyExchange,
    worker: &Arc<Mutex<WorkerHandle>>,
    req: &ComputeRequest,
) -> anyhow::Result<ComputeResponse> {
    let peer_kx: [u8; 32] = req
        .requester_kx_public
        .as_slice()
        .try_into()
        .map_err(|_| anyhow::anyhow!("bad kx public key"))?;
    let secret = identity_kx.shared_secret(&peer_kx);
    let plain = decrypt(&secret, &req.encrypted_activations)?;
    let tensor = bytes_to_tensor(&plain)?;
    let positions = decode_positions(&secret, &req.encrypted_position_ids)?;
    let memory = decode_positions(&secret, &req.encrypted_encoder_memory)?;
    let patch = decode_patch(&secret, req)?;
    let compute_start = std::time::Instant::now();
    let out = {
        let mut w = worker.lock().await.clone();
        w.run_slice(
            &req.model_id,
            req.start_layer,
            req.end_layer,
            &req.session_id,
            0,
            tensor,
            true,
            req.top_k,
            req.accepts_bf16,
            positions,
            memory,
            patch,
            crate::worker::Draw {
                guidance: req.guidance,
                sample: req.sample,
                temperature: req.temperature,
                top_p: req.top_p,
                seed: req.seed,
            },
        )
        .await?
    };
    let compute_ms = compute_start.elapsed().as_millis() as u64;
    let out_bytes = tensor_to_bytes(&out);
    let encrypted = encrypt(&secret, &out_bytes)?;
    Ok(ComputeResponse {
        encrypted_activations: encrypted,
        ok: true,
        error: String::new(),
        compute_ms,
        protocol_version: crate::registry::WIRE_VERSION,
    })
}

pub async fn process_chained_request(
    identity_kx: &KeyExchange,
    worker: &Arc<Mutex<WorkerHandle>>,
    req: &ComputeRequest,
    hops: &HopClients,
) -> anyhow::Result<ComputeResponse> {
    let peer_kx: [u8; 32] = req
        .requester_kx_public
        .as_slice()
        .try_into()
        .map_err(|_| anyhow::anyhow!("bad kx public key"))?;
    let secret = identity_kx.shared_secret(&peer_kx);
    let plain = decrypt(&secret, &req.encrypted_activations)?;
    let tensor = bytes_to_tensor(&plain)?;
    let positions = decode_positions(&secret, &req.encrypted_position_ids)?;
    let memory = decode_positions(&secret, &req.encrypted_encoder_memory)?;
    let patch = decode_patch(&secret, req)?;

    let compute_start = std::time::Instant::now();
    let out = {
        let mut w = worker.lock().await.clone();
        w.run_slice(
            &req.model_id,
            req.start_layer,
            req.end_layer,
            &req.session_id,
            0,
            tensor,
            true,
            req.top_k,
            req.accepts_bf16,
            positions.clone(),
            memory.clone(),
            patch.clone(),
            crate::worker::Draw {
                guidance: req.guidance,
                sample: req.sample,
                temperature: req.temperature,
                top_p: req.top_p,
                seed: req.seed,
            },
        )
        .await?
    };
    let mut compute_ms = compute_start.elapsed().as_millis() as u64;

    let payload = if req.route.is_empty() {
        tensor_to_bytes(&out)
    } else {
        let next = &req.route[0];
        let next_kx: [u8; 32] = next
            .kx_public
            .as_slice()
            .try_into()
            .map_err(|_| anyhow::anyhow!("hop {} has a bad kx public key", next.compute_endpoint))?;
        let next_secret = identity_kx.shared_secret(&next_kx);
        let forwarded = encrypt(&next_secret, &tensor_to_bytes(&out))?;

        let mut client = hops.get(&next.compute_endpoint).await.map_err(|e| {
            anyhow::anyhow!("chain stalled at [{}]: {}", next.compute_endpoint, e)
        })?;
        let downstream = client
            .run_slice(ComputeRequest {
                patch_offset: req.patch_offset,
                patch_sequence: req.patch_sequence,
                branch: req.branch.clone(),
                layout: req.layout.clone(),
                guidance: req.guidance,
                sample: req.sample,
                temperature: req.temperature,
                top_p: req.top_p,
                seed: req.seed,
                encrypted_arguments: match &patch {
                    Some(p) if !p.arguments.is_empty() => {
                        encrypt(&next_secret, &arguments_to_bytes(&p.arguments))?
                    }
                    _ => Vec::new(),
                },
                requester_kx_public: identity_kx.public_bytes().to_vec(),
                model_id: req.model_id.clone(),
                start_layer: next.start_layer,
                end_layer: next.end_layer,
                session_id: req.session_id.clone(),
                encrypted_activations: forwarded,
                top_k: next.top_k,
                route: req.route[1..].to_vec(),
                accepts_bf16: req.route.get(1).map(|h| h.accepts_bf16).unwrap_or(false),
                encrypted_position_ids: match &positions {
                    Some(p) => encrypt(&next_secret, &tensor_to_bytes(p))?,
                    None => Vec::new(),
                },
                encrypted_encoder_memory: match &memory {
                    Some(m) => encrypt(&next_secret, &tensor_to_bytes(m))?,
                    None => Vec::new(),
                },
            })
            .await;

        let downstream = match downstream {
            Ok(r) => r.into_inner(),
            Err(e) => {
                hops.forget(&next.compute_endpoint).await;
                anyhow::bail!("chain stalled at [{}]: {}", next.compute_endpoint, e);
            }
        };
        if !downstream.ok {
            anyhow::bail!(
                "chain stalled at [{}]: {}",
                next.compute_endpoint,
                downstream.error
            );
        }
        compute_ms += downstream.compute_ms;
        decrypt(&next_secret, &downstream.encrypted_activations)?
    };

    Ok(ComputeResponse {
        encrypted_activations: encrypt(&secret, &payload)?,
        ok: true,
        error: String::new(),
        compute_ms,
        protocol_version: crate::registry::WIRE_VERSION,
    })
}
