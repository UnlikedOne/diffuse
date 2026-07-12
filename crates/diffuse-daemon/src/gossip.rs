pub mod pb {
    tonic::include_proto!("diffuse.gossip");
}

use std::sync::Arc;

use tokio::sync::Mutex;
use tonic::{Request, Response, Status};

use crate::registry::{Peer, PeerRegistry};
use diffuse_trust::crypto::verify;
use pb::gossip_client::GossipClient;
use pb::gossip_server::Gossip;
use pb::{GossipRequest, GossipResponse, PeerInfo, ReachabilityRequest, ReachabilityResponse};

pub fn peer_to_info(p: &Peer) -> PeerInfo {
    PeerInfo {
        node_id: p.node_id.clone(),
        daemon_endpoint: p.daemon_endpoint.clone(),
        worker_endpoint: p.worker_endpoint.clone(),
        model_id: p.model_id.clone(),
        start_layer: p.start_layer,
        end_layer: p.end_layer,
        last_seen_ms: p.last_seen_ms,
        signature: p.signature.clone(),
        kx_public: p.kx_public.clone(),
        reachable: p.reachable,
    }
}

pub fn info_to_peer(i: &PeerInfo) -> Peer {
    Peer {
        node_id: i.node_id.clone(),
        daemon_endpoint: i.daemon_endpoint.clone(),
        worker_endpoint: i.worker_endpoint.clone(),
        model_id: i.model_id.clone(),
        start_layer: i.start_layer,
        end_layer: i.end_layer,
        last_seen_ms: i.last_seen_ms,
        signature: i.signature.clone(),
        kx_public: i.kx_public.clone(),
        reachable: i.reachable,
    }
}

fn peer_is_authentic(peer: &Peer) -> bool {
    if peer.node_id.is_empty() || peer.signature.is_empty() {
        return false;
    }
    verify(&peer.node_id, &peer.signable_bytes(), &peer.signature)
}

fn merge_authentic(registry: &mut PeerRegistry, incoming: Vec<Peer>) -> usize {
    let mut accepted = 0;
    for peer in incoming {
        if peer_is_authentic(&peer) {
            registry.upsert(peer);
            accepted += 1;
        } else {
            tracing::warn!(
                "gossip: rejected unauthenticated peer {}",
                peer.daemon_endpoint
            );
        }
    }
    accepted
}

fn extract_port(endpoint: &str) -> Option<u16> {
    let after_scheme = endpoint.strip_prefix("http://").unwrap_or(endpoint);
    after_scheme.rsplit(':').next()?.parse().ok()
}

pub struct GossipService {
    pub registry: Arc<Mutex<PeerRegistry>>,
}

#[tonic::async_trait]
impl Gossip for GossipService {
    async fn exchange(
        &self,
        request: Request<GossipRequest>,
    ) -> Result<Response<GossipResponse>, Status> {
        let req = request.into_inner();

        let incoming: Vec<Peer> = req.known_peers.iter().map(info_to_peer).collect();

        let mut reg = self.registry.lock().await;
        merge_authentic(&mut reg, incoming);
        let outgoing: Vec<PeerInfo> = reg.all().iter().map(peer_to_info).collect();
        drop(reg);

        Ok(Response::new(GossipResponse {
            known_peers: outgoing,
        }))
    }

    async fn check_reachability(
        &self,
        request: Request<ReachabilityRequest>,
    ) -> Result<Response<ReachabilityResponse>, Status> {
        let source_ip = request
            .remote_addr()
            .map(|a| a.ip())
            .ok_or_else(|| Status::internal("cannot determine caller ip"))?;

        let req = request.into_inner();
        let port = extract_port(&req.compute_endpoint).unwrap_or(0);
        if port == 0 {
            return Err(Status::invalid_argument("bad compute endpoint port"));
        }

        let addr = match source_ip {
            std::net::IpAddr::V4(v4) => format!("{}:{}", v4, port),
            std::net::IpAddr::V6(v6) => format!("[{}]:{}", v6, port),
        };

        let reachable = matches!(
            tokio::time::timeout(
                std::time::Duration::from_secs(3),
                tokio::net::TcpStream::connect(&addr),
            )
            .await,
            Ok(Ok(_))
        );
        tracing::info!("reachability check for {}: {}", addr, reachable);
        Ok(Response::new(ReachabilityResponse { reachable }))
    }
}

pub async fn gossip_with(
    endpoint: &str,
    self_node_id: Vec<u8>,
    registry: &Arc<Mutex<PeerRegistry>>,
) -> anyhow::Result<usize> {
    let known: Vec<PeerInfo> = {
        let reg = registry.lock().await;
        reg.all().iter().map(peer_to_info).collect()
    };

    let mut client = GossipClient::connect(endpoint.to_string()).await?;
    let response = client
        .exchange(GossipRequest {
            from_node_id: self_node_id,
            known_peers: known,
        })
        .await?
        .into_inner();

    let received: Vec<Peer> = response.known_peers.iter().map(info_to_peer).collect();

    let mut reg = registry.lock().await;
    let accepted = merge_authentic(&mut reg, received);
    reg.touch(endpoint);
    drop(reg);

    Ok(accepted)
}

pub fn spawn_gossip_server(
    listen_addr: std::net::SocketAddr,
    registry: Arc<Mutex<PeerRegistry>>,
    relay_state: crate::relay::RelayState,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let gossip_service = GossipService { registry };
        let relay_service = crate::relay::RelayService { state: relay_state };
        let server = tonic::transport::Server::builder()
            .add_service(pb::gossip_server::GossipServer::new(gossip_service))
            .add_service(
                crate::relay::pb::relay_server::RelayServer::new(relay_service)
                    .max_decoding_message_size(128 * 1024 * 1024)
                    .max_encoding_message_size(128 * 1024 * 1024),
            )
            .serve(listen_addr);
        if let Err(e) = server.await {
            tracing::error!("gossip/relay server error: {}", e);
        }
    })
}

pub fn spawn_prune_loop(
    registry: Arc<Mutex<PeerRegistry>>,
    interval: std::time::Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(interval).await;
            let removed = {
                let mut reg = registry.lock().await;
                reg.prune()
            };
            if removed > 0 {
                tracing::info!("pruned {} stale peer(s)", removed);
            }
        }
    })
}

pub async fn check_my_reachability(
    sentinel_endpoint: &str,
    my_compute_endpoint: &str,
) -> bool {
    let result = async {
        let mut client = GossipClient::connect(sentinel_endpoint.to_string()).await?;
        let resp = client
            .check_reachability(ReachabilityRequest {
                compute_endpoint: my_compute_endpoint.to_string(),
            })
            .await?
            .into_inner();
        Ok::<bool, anyhow::Error>(resp.reachable)
    }
    .await;
    match result {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("reachability check failed: {}, assuming reachable", e);
            true
        }
    }
}