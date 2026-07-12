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
use pb::{ComputeRequest, ComputeResponse};

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

pub struct ComputeService {
    pub identity_kx: Arc<KeyExchange>,
    pub worker: Arc<Mutex<WorkerHandle>>,
    pub model_id: String,
}

#[tonic::async_trait]
impl Compute for ComputeService {
    async fn run_slice(
        &self,
        request: Request<ComputeRequest>,
    ) -> Result<Response<ComputeResponse>, Status> {
        let req = request.into_inner();

        let peer_kx: [u8; 32] = req
            .requester_kx_public
            .as_slice()
            .try_into()
            .map_err(|_| Status::invalid_argument("bad kx public key"))?;
        let secret = self.identity_kx.shared_secret(&peer_kx);

        let plain = decrypt(&secret, &req.encrypted_activations)
            .map_err(|e| Status::invalid_argument(format!("decrypt failed: {}", e)))?;
        let tensor = bytes_to_tensor(&plain)
            .map_err(|e| Status::internal(format!("bad tensor: {}", e)))?;

        let compute_start = std::time::Instant::now();
        let out = {
            let mut w = self.worker.lock().await;
            w.run_slice(
                &req.model_id,
                req.start_layer,
                req.end_layer,
                &req.session_id,
                0,
                tensor,
                true,
            )
            .await
            .map_err(|e| Status::internal(format!("worker failed: {}", e)))?
        };
        let compute_ms = compute_start.elapsed().as_millis() as u64;

        let out_bytes = tensor_to_bytes(&out);
        let encrypted = encrypt(&secret, &out_bytes)
            .map_err(|e| Status::internal(format!("encrypt failed: {}", e)))?;

        Ok(Response::new(ComputeResponse {
            encrypted_activations: encrypted,
            ok: true,
            error: String::new(),
            compute_ms,
        }))
    }
}

pub async fn request_slice(
    daemon_endpoint: &str,
    host_kx_public: &[u8; 32],
    my_kx: &KeyExchange,
    model_id: &str,
    start: u32,
    end: u32,
    session_id: &str,
    activations: &Tensor,
) -> anyhow::Result<(Tensor, u64)> {
    let secret = my_kx.shared_secret(host_kx_public);
    let plain = tensor_to_bytes(activations);
    let encrypted = encrypt(&secret, &plain)?;

    let mut client = ComputeClient::connect(daemon_endpoint.to_string())
        .await?
        .max_decoding_message_size(128 * 1024 * 1024)
        .max_encoding_message_size(128 * 1024 * 1024);
    let response = client
        .run_slice(ComputeRequest {
            requester_kx_public: my_kx.public_bytes().to_vec(),
            model_id: model_id.to_string(),
            start_layer: start,
            end_layer: end,
            session_id: session_id.to_string(),
            encrypted_activations: encrypted,
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