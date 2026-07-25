use std::collections::HashMap;

/// Wire protocol this build speaks. Bumped when the format of what travels
/// between nodes changes in a way older nodes cannot parse. A peer's version is
/// never taken from gossip: a node that predates a field silently drops it when
/// relaying, and the field would be unsigned and so forgeable. It is learned
/// from the peer's own answer on the encrypted channel instead.
pub const WIRE_VERSION: u32 = 1;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug)]
pub struct Peer {
    pub node_id: Vec<u8>,
    pub daemon_endpoint: String,
    pub worker_endpoint: String,
    pub model_id: String,
    pub start_layer: u32,
    pub end_layer: u32,
    /// Total number of layers in the model, as reported by the worker that
    /// hosts this slice. `0` means "unknown" — a peer running a version that
    /// predates this field, in which case consumers fall back to inferring the
    /// total from the highest advertised `end_layer`.
    pub total_layers: u32,
    pub last_seen_ms: u64,
    pub signature: Vec<u8>,
    pub kx_public: Vec<u8>,
    pub reachable: bool,
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

pub struct PeerRegistry {
    peers: HashMap<Vec<u8>, Peer>,
    stale_after_ms: u64,
}

impl PeerRegistry {
    pub fn new(stale_after_ms: u64) -> Self {
        Self {
            peers: HashMap::new(),
            stale_after_ms,
        }
    }

    pub fn upsert(&mut self, peer: Peer) {
        if peer.node_id.is_empty() {
            return;
        }
        let key = peer.node_id.clone();
        match self.peers.get_mut(&key) {
            Some(existing) => {
                if peer.last_seen_ms >= existing.last_seen_ms {
                    *existing = peer;
                }
            }
            None => {
                self.peers.insert(key, peer);
            }
        }
    }

    pub fn merge(&mut self, incoming: Vec<Peer>) {
        for peer in incoming {
            self.upsert(peer);
        }
    }

    pub fn touch_endpoint(&mut self, daemon_endpoint: &str) {
        for p in self.peers.values_mut() {
            if p.daemon_endpoint == daemon_endpoint {
                p.last_seen_ms = now_ms();
            }
        }
    }

    pub fn touch_node(&mut self, node_id: &[u8]) {
        if let Some(p) = self.peers.get_mut(node_id) {
            p.last_seen_ms = now_ms();
        }
    }

    pub fn prune(&mut self) -> usize {
        let cutoff = now_ms().saturating_sub(self.stale_after_ms);
        let before = self.peers.len();
        self.peers.retain(|_, p| p.last_seen_ms >= cutoff);
        before - self.peers.len()
    }

    pub fn all(&self) -> Vec<Peer> {
        self.peers.values().cloned().collect()
    }

    pub fn live_endpoints(&self) -> Vec<String> {
        let mut endpoints: Vec<String> = self
            .peers
            .values()
            .map(|p| p.daemon_endpoint.clone())
            .collect();
        endpoints.sort();
        endpoints.dedup();
        endpoints
    }

    pub fn len(&self) -> usize {
        self.peers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.peers.is_empty()
    }

    pub fn replicas_for_slice(&self, model_id: &str, start: u32, end: u32) -> Vec<Peer> {
        self.peers
            .values()
            .filter(|p| p.model_id == model_id && p.start_layer == start && p.end_layer == end)
            .cloned()
            .collect()
    }

    pub fn remove_node(&mut self, node_id: &[u8]) -> bool {
        self.peers.remove(node_id).is_some()
    }
}

impl Peer {
    pub fn signable_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&self.node_id);
        buf.push(0);
        buf.extend_from_slice(self.daemon_endpoint.as_bytes());
        buf.push(0);
        buf.extend_from_slice(self.worker_endpoint.as_bytes());
        buf.push(0);
        buf.extend_from_slice(self.model_id.as_bytes());
        buf.push(0);
        buf.extend_from_slice(&self.start_layer.to_le_bytes());
        buf.extend_from_slice(&self.end_layer.to_le_bytes());
        buf.extend_from_slice(&self.kx_public);
        // `total_layers` was added after the original signing scheme. Only fold
        // it into the signed bytes when it is present, so that a peer running an
        // older version (which never sends the field, decoded here as 0) still
        // produces bytes a newer verifier can reproduce and accept. Newer peers
        // always set it, so their signatures cover it.
        if self.total_layers != 0 {
            buf.extend_from_slice(&self.total_layers.to_le_bytes());
        }
        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peer(node_id: u8, endpoint: &str, model: &str, last_seen_ms: u64) -> Peer {
        Peer {
            node_id: vec![node_id; 32],
            daemon_endpoint: endpoint.to_string(),
            worker_endpoint: "http://127.0.0.1:50051".to_string(),
            model_id: model.to_string(),
            start_layer: 0,
            end_layer: 24,
            total_layers: 24,
            last_seen_ms,
            signature: vec![1, 2, 3],
            kx_public: vec![9; 32],
            reachable: true,
        }
    }

    #[test]
    fn signable_bytes_are_backward_compatible_when_total_is_unknown() {
        // A peer that predates the total_layers field reports 0. Its signed
        // bytes must be identical to the pre-field scheme so a newer verifier,
        // recomputing them, still accepts the old signature.
        let mut legacy = peer(1, "http://a", "m", now_ms());
        legacy.total_layers = 0;
        let legacy_bytes = legacy.signable_bytes();

        // Reconstruct the pre-field byte layout by hand.
        let mut expected = Vec::new();
        expected.extend_from_slice(&legacy.node_id);
        expected.push(0);
        expected.extend_from_slice(legacy.daemon_endpoint.as_bytes());
        expected.push(0);
        expected.extend_from_slice(legacy.worker_endpoint.as_bytes());
        expected.push(0);
        expected.extend_from_slice(legacy.model_id.as_bytes());
        expected.push(0);
        expected.extend_from_slice(&legacy.start_layer.to_le_bytes());
        expected.extend_from_slice(&legacy.end_layer.to_le_bytes());
        expected.extend_from_slice(&legacy.kx_public);
        assert_eq!(legacy_bytes, expected, "unknown total must not alter the signed bytes");

        // A peer that does report a total folds it into the signature.
        let mut modern = peer(1, "http://a", "m", now_ms());
        modern.total_layers = 64;
        assert_eq!(
            modern.signable_bytes().len(),
            legacy_bytes.len() + 4,
            "a known total extends the signed bytes by its 4 encoded bytes"
        );
    }

    #[test]
    fn two_nodes_sharing_an_endpoint_stay_distinct() {
        let mut reg = PeerRegistry::new(60_000);
        reg.upsert(peer(1, "http://0.0.0.0:9440", "model-a", now_ms()));
        reg.upsert(peer(2, "http://0.0.0.0:9440", "model-b", now_ms()));
        assert_eq!(reg.len(), 2);
        let models: Vec<String> = reg.all().into_iter().map(|p| p.model_id).collect();
        assert!(models.contains(&"model-a".to_string()));
        assert!(models.contains(&"model-b".to_string()));
    }

    #[test]
    fn same_node_reannouncing_updates_in_place() {
        let mut reg = PeerRegistry::new(60_000);
        let t0 = now_ms();
        reg.upsert(peer(1, "http://1.2.3.4:9440", "model-a", t0));
        reg.upsert(peer(1, "http://1.2.3.4:9440", "model-a", t0 + 5_000));
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.all()[0].last_seen_ms, t0 + 5_000);
    }

    #[test]
    fn a_stale_record_does_not_rejuvenate_a_fresh_peer() {
        let mut reg = PeerRegistry::new(60_000);
        let t0 = now_ms();
        reg.upsert(peer(1, "http://1.2.3.4:9440", "model-a", t0));
        reg.upsert(peer(1, "http://1.2.3.4:9440", "model-a", t0 - 30_000));
        assert_eq!(reg.all()[0].last_seen_ms, t0);
    }

    #[test]
    fn expired_peers_are_pruned() {
        let mut reg = PeerRegistry::new(60_000);
        reg.upsert(peer(1, "http://1.2.3.4:9440", "model-a", now_ms()));
        reg.upsert(peer(2, "http://5.6.7.8:9440", "model-b", now_ms() - 120_000));
        assert_eq!(reg.len(), 2);
        let removed = reg.prune();
        assert_eq!(removed, 1);
        assert_eq!(reg.len(), 1);
        assert_eq!(reg.all()[0].model_id, "model-a");
    }

    #[test]
    fn a_departed_node_is_not_kept_alive_by_gossip_echo() {
        let mut reg = PeerRegistry::new(60_000);
        let departed = peer(1, "http://1.2.3.4:9440", "model-a", now_ms() - 120_000);
        reg.upsert(departed.clone());
        reg.merge(vec![departed.clone(), departed]);
        assert_eq!(reg.prune(), 1);
        assert!(reg.is_empty());
    }

    #[test]
    fn peers_without_a_node_id_are_rejected() {
        let mut reg = PeerRegistry::new(60_000);
        let mut anonymous = peer(1, "http://1.2.3.4:9440", "model-a", now_ms());
        anonymous.node_id = Vec::new();
        reg.upsert(anonymous);
        assert!(reg.is_empty());
    }

    #[test]
    fn replicas_are_selected_by_model_and_slice() {
        let mut reg = PeerRegistry::new(60_000);
        reg.upsert(peer(1, "http://1.1.1.1:9440", "model-a", now_ms()));
        reg.upsert(peer(2, "http://2.2.2.2:9440", "model-a", now_ms()));
        reg.upsert(peer(3, "http://3.3.3.3:9440", "model-b", now_ms()));
        assert_eq!(reg.replicas_for_slice("model-a", 0, 24).len(), 2);
        assert_eq!(reg.replicas_for_slice("model-b", 0, 24).len(), 1);
        assert_eq!(reg.replicas_for_slice("model-a", 0, 32).len(), 0);
    }

    #[test]
    fn live_endpoints_are_deduplicated() {
        let mut reg = PeerRegistry::new(60_000);
        reg.upsert(peer(1, "http://0.0.0.0:9440", "model-a", now_ms()));
        reg.upsert(peer(2, "http://0.0.0.0:9440", "model-b", now_ms()));
        reg.upsert(peer(3, "http://9.9.9.9:9440", "model-c", now_ms()));
        assert_eq!(reg.live_endpoints().len(), 2);
    }
}