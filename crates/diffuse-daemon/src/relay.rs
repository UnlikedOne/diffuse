pub mod pb {
    pub use crate::compute::pb::*;
}

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status, Streaming};

use pb::relay_server::Relay;
use pb::{ComputeResponse, RelayComputeRequest, RelayEnvelope, RelayReply};

const REGISTER_MARKER: &str = "register";

type NodeId = Vec<u8>;

#[derive(Clone, Default)]
pub struct RelayState {
    registrations: Arc<Mutex<HashMap<NodeId, mpsc::Sender<RelayEnvelope>>>>,
    pending: Arc<Mutex<HashMap<String, oneshot::Sender<ComputeResponse>>>>,
    registry: Option<Arc<Mutex<crate::registry::PeerRegistry>>>,
}

impl RelayState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_registry(registry: Arc<Mutex<crate::registry::PeerRegistry>>) -> Self {
        Self {
            registrations: Arc::new(Mutex::new(HashMap::new())),
            pending: Arc::new(Mutex::new(HashMap::new())),
            registry: Some(registry),
        }
    }
}

pub struct RelayService {
    pub state: RelayState,
}

#[tonic::async_trait]
impl Relay for RelayService {
    type AttachStream = ReceiverStream<Result<RelayEnvelope, Status>>;

    async fn attach(
        &self,
        request: Request<Streaming<RelayReply>>,
    ) -> Result<Response<Self::AttachStream>, Status> {
        let mut inbound = request.into_inner();

        let first = inbound
            .next()
            .await
            .ok_or_else(|| Status::invalid_argument("empty attach stream"))?
            .map_err(|e| Status::internal(format!("attach recv error: {}", e)))?;

        if first.request_id != REGISTER_MARKER || first.node_id.is_empty() {
            return Err(Status::invalid_argument(
                "first message must be a register marker with node_id",
            ));
        }
        let node_id = first.node_id.clone();

        let (to_node_tx, to_node_rx) = mpsc::channel::<RelayEnvelope>(64);
        {
            let mut regs = self.state.registrations.lock().await;
            regs.insert(node_id.clone(), to_node_tx);
        }
        tracing::info!("relay: node {} attached", hex_short(&node_id));

        let pending = self.state.pending.clone();
        let registrations = self.state.registrations.clone();
        let registry = self.state.registry.clone();
        let node_id_for_replies = node_id.clone();
        tokio::spawn(async move {
            while let Some(msg) = inbound.next().await {
                match msg {
                    Ok(reply) => {
                        if reply.request_id == REGISTER_MARKER {
                            continue;
                        }
                        if let Some(resp) = reply.response {
                            let waiter = {
                                let mut p = pending.lock().await;
                                p.remove(&reply.request_id)
                            };
                            if let Some(tx) = waiter {
                                let _ = tx.send(resp);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "relay: node {} stream error: {}",
                            hex_short(&node_id_for_replies),
                            e
                        );
                        break;
                    }
                }
            }
            {
                let mut regs = registrations.lock().await;
                regs.remove(&node_id_for_replies);
            }
            if let Some(reg) = registry {
                let evicted = {
                    let mut r = reg.lock().await;
                    r.remove_node(&node_id_for_replies)
                };
                if evicted {
                    tracing::info!(
                        "relay: evicted node {} from registry on detach",
                        hex_short(&node_id_for_replies)
                    );
                }
            }
            tracing::info!("relay: node {} detached", hex_short(&node_id_for_replies));
        });

        let (out_tx, out_rx) = mpsc::channel::<Result<RelayEnvelope, Status>>(64);
        let mut to_node_rx = to_node_rx;
        tokio::spawn(async move {
            while let Some(env) = to_node_rx.recv().await {
                if out_tx.send(Ok(env)).await.is_err() {
                    break;
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(out_rx)))
    }

    async fn relay_compute(
        &self,
        request: Request<RelayComputeRequest>,
    ) -> Result<Response<ComputeResponse>, Status> {
        let req = request.into_inner();
        let target = req.target_node_id;
        let compute_req = req
            .request
            .ok_or_else(|| Status::invalid_argument("missing compute request"))?;

        let sender = {
            let regs = self.state.registrations.lock().await;
            regs.get(&target).cloned()
        };
        let sender = sender.ok_or_else(|| {
            Status::unavailable(format!("target {} not attached", hex_short(&target)))
        })?;

        let request_id = new_request_id();
        let (resp_tx, resp_rx) = oneshot::channel::<ComputeResponse>();
        {
            let mut p = self.state.pending.lock().await;
            p.insert(request_id.clone(), resp_tx);
        }

        let envelope = RelayEnvelope {
            request_id: request_id.clone(),
            request: Some(compute_req),
        };
        if sender.send(envelope).await.is_err() {
            let mut p = self.state.pending.lock().await;
            p.remove(&request_id);
            return Err(Status::unavailable("target channel closed"));
        }

        match tokio::time::timeout(std::time::Duration::from_secs(30), resp_rx).await {
            Ok(Ok(resp)) => Ok(Response::new(resp)),
            Ok(Err(_)) => {
                let mut p = self.state.pending.lock().await;
                p.remove(&request_id);
                Err(Status::internal("relay reply channel dropped"))
            }
            Err(_) => {
                let mut p = self.state.pending.lock().await;
                p.remove(&request_id);
                Err(Status::deadline_exceeded("relay compute timed out"))
            }
        }
    }
}

fn new_request_id() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{:x}", now)
}

fn hex_short(id: &[u8]) -> String {
    id.iter().take(4).map(|b| format!("{:02x}", b)).collect()
}

use pb::relay_client::RelayClient;

pub fn spawn_relay_client(
    sentinel_endpoint: String,
    node_id: Vec<u8>,
    identity_kx: Arc<diffuse_trust::transport::KeyExchange>,
    worker: Arc<Mutex<crate::worker::WorkerHandle>>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            match run_relay_client(&sentinel_endpoint, &node_id, &identity_kx, &worker).await {
                Ok(_) => tracing::warn!("relay client stream ended, reconnecting in 3s"),
                Err(e) => tracing::warn!("relay client error: {}, reconnecting in 3s", e),
            }
            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        }
    })
}

async fn run_relay_client(
    sentinel_endpoint: &str,
    node_id: &[u8],
    identity_kx: &Arc<diffuse_trust::transport::KeyExchange>,
    worker: &Arc<Mutex<crate::worker::WorkerHandle>>,
) -> anyhow::Result<()> {
    let mut client = RelayClient::connect(sentinel_endpoint.to_string())
        .await?
        .max_decoding_message_size(128 * 1024 * 1024)
        .max_encoding_message_size(128 * 1024 * 1024);

    let (reply_tx, reply_rx) = mpsc::channel::<RelayReply>(64);

    let register = RelayReply {
        request_id: REGISTER_MARKER.to_string(),
        response: None,
        node_id: node_id.to_vec(),
    };
    reply_tx
        .send(register)
        .await
        .map_err(|_| anyhow::anyhow!("failed to send register"))?;

    let outbound = ReceiverStream::new(reply_rx);
    let response = client.attach(outbound).await?;
    let mut inbound = response.into_inner();

    tracing::info!("relay client attached to {}", sentinel_endpoint);

    while let Some(env) = inbound.next().await {
        let env = env?;
        let request_id = env.request_id.clone();
        let compute_req = match env.request {
            Some(r) => r,
            None => continue,
        };

        let identity_kx = identity_kx.clone();
        let worker = worker.clone();
        let reply_tx = reply_tx.clone();
        tokio::spawn(async move {
            let response = match crate::compute::process_compute_request(
                &identity_kx,
                &worker,
                &compute_req,
            )
            .await
            {
                Ok(r) => r,
                Err(e) => ComputeResponse {
                    encrypted_activations: Vec::new(),
                    ok: false,
                    error: format!("relay compute failed: {}", e),
                    compute_ms: 0,
                },
            };
            let reply = RelayReply {
                request_id,
                response: Some(response),
                node_id: Vec::new(),
            };
            let _ = reply_tx.send(reply).await;
        });
    }

    Ok(())
}

pub async fn connect_relay(
    endpoint: &str,
) -> anyhow::Result<RelayClient<tonic::transport::Channel>> {
    let client = RelayClient::connect(endpoint.to_string())
        .await?
        .max_decoding_message_size(128 * 1024 * 1024)
        .max_encoding_message_size(128 * 1024 * 1024);
    Ok(client)
}

pub async fn relay_compute(
    relay_endpoint: &str,
    client: &mut Option<RelayClient<tonic::transport::Channel>>,
    target_node_id: &[u8],
    host_kx_public: &[u8; 32],
    my_kx: &diffuse_trust::transport::KeyExchange,
    model_id: &str,
    start: u32,
    end: u32,
    session_id: &str,
    activations: &crate::worker::pb::Tensor,
    top_k: u32,
    accepts_bf16: bool,
) -> anyhow::Result<(crate::worker::pb::Tensor, u64)> {
    use pb::ComputeRequest;
    let secret = my_kx.shared_secret(host_kx_public);
    let plain = crate::compute::tensor_to_bytes_pub(activations);
    let encrypted = diffuse_trust::transport::encrypt(&secret, &plain)?;

    if client.is_none() {
        *client = Some(connect_relay(relay_endpoint).await?);
    }
    let relay = client
        .as_mut()
        .expect("relay client was just established above");

    let compute_req = ComputeRequest {
        requester_kx_public: my_kx.public_bytes().to_vec(),
        model_id: model_id.to_string(),
        start_layer: start,
        end_layer: end,
        session_id: session_id.to_string(),
        encrypted_activations: encrypted,
        top_k,
        route: Vec::new(),
        accepts_bf16,
    };

    let response = match relay
        .relay_compute(RelayComputeRequest {
            target_node_id: target_node_id.to_vec(),
            request: Some(compute_req),
        })
        .await
    {
        Ok(r) => r.into_inner(),
        Err(e) => {
            *client = None;
            return Err(anyhow::anyhow!("relay {} failed: {}", relay_endpoint, e));
        }
    };

    if !response.ok {
        anyhow::bail!("relay compute failed: {}", response.error);
    }
    let plain_out = diffuse_trust::transport::decrypt(&secret, &response.encrypted_activations)?;
    let tensor = crate::compute::bytes_to_tensor_pub(&plain_out)?;
    Ok((tensor, response.compute_ms))
}