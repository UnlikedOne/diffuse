use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug)]
pub struct Peer {
    pub node_id: Vec<u8>,
    pub daemon_endpoint: String,
    pub worker_endpoint: String,
    pub model_id: String,
    pub start_layer: u32,
    pub end_layer: u32,
    pub last_seen_ms: u64,
    pub signature: Vec<u8>,
    pub kx_public: Vec<u8>,
}

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

pub struct PeerRegistry {
    peers: HashMap<String, Peer>,
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
        let key = peer.daemon_endpoint.clone();
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

    pub fn touch(&mut self, daemon_endpoint: &str) {
        if let Some(p) = self.peers.get_mut(daemon_endpoint) {
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
        self.peers.keys().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.peers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.peers.is_empty()
    }

    pub fn replicas_for_slice(
        &self,
        model_id: &str,
        start: u32,
        end: u32,
    ) -> Vec<Peer> {
        self.peers
            .values()
            .filter(|p| p.model_id == model_id && p.start_layer == start && p.end_layer == end)
            .cloned()
            .collect()
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
        buf
    }
}