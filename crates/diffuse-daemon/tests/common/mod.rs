use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use rand::RngCore;

use diffuse_daemon::registry::{now_ms, Peer};
use diffuse_trust::crypto::sign;

pub fn new_key() -> SigningKey {
    let mut secret = [0u8; 32];
    OsRng.fill_bytes(&mut secret);
    SigningKey::from_bytes(&secret)
}

pub fn signed_peer(
    key: &SigningKey,
    daemon_endpoint: &str,
    worker_endpoint: &str,
    model_id: &str,
    start: u32,
    end: u32,
) -> Peer {
    let node_id = key.verifying_key().to_bytes().to_vec();
    let mut peer = Peer {
        node_id,
        daemon_endpoint: daemon_endpoint.to_string(),
        worker_endpoint: worker_endpoint.to_string(),
        model_id: model_id.to_string(),
        start_layer: start,
        end_layer: end,
        total_layers: 0,
        last_seen_ms: now_ms(),
        signature: Vec::new(),
        kx_public: Vec::new(),
        reachable: true,
    };
    peer.signature = sign(key, &peer.signable_bytes());
    peer
}