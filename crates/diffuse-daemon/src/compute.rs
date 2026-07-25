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
) -> anyhow::Result<(Tensor, u64)> {
    let secret = my_kx.shared_secret(host_kx_public);
    let plain = tensor_to_bytes(activations);
    let encrypted = encrypt(&secret, &plain)?;
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
        })
        .await?
        .into_inner();
    if !response.ok {
        anyhow::bail!("remote compute failed: {}", response.error);
    }
    let plain_out = decrypt(&secret, &response.encrypted_activations)?;
    let tensor = bytes_to_tensor(&plain_out)?;
    Ok((tensor, response.compute_ms))
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
) -> anyhow::Result<(Tensor, u64)> {
    let secret = my_kx.shared_secret(host_kx_public);
    let plain = tensor_to_bytes(activations);
    let encrypted = encrypt(&secret, &plain)?;
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
        })
        .await?
        .into_inner();
    if !response.ok {
        anyhow::bail!("chained compute failed: {}", response.error);
    }
    let plain_out = decrypt(&secret, &response.encrypted_activations)?;
    let tensor = bytes_to_tensor(&plain_out)?;
    Ok((tensor, response.compute_ms))
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
                requester_kx_public: identity_kx.public_bytes().to_vec(),
                model_id: req.model_id.clone(),
                start_layer: next.start_layer,
                end_layer: next.end_layer,
                session_id: req.session_id.clone(),
                encrypted_activations: forwarded,
                top_k: next.top_k,
                route: req.route[1..].to_vec(),
                accepts_bf16: req.route.get(1).map(|h| h.accepts_bf16).unwrap_or(false),
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
    })
}