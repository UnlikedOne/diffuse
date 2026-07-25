use diffuse_daemon::capacity::analyze;
use diffuse_daemon::display::render_network_state;
use diffuse_daemon::registry::{now_ms, Peer, PeerRegistry};

fn peer(model: &str, ep: &str, start: u32, end: u32) -> Peer {
    Peer {
        node_id: vec![1],
        daemon_endpoint: ep.to_string(),
        worker_endpoint: format!("{}-w", ep),
        model_id: model.to_string(),
        start_layer: start,
        end_layer: end,
        total_layers: 0,
        last_seen_ms: now_ms(),
        signature: Vec::new(),
        kx_public: Vec::new(),
        reachable: true,
    }
}

fn main() {
    let mut r = PeerRegistry::new(600_000);

    // A robust model: fully covered, k=2 everywhere.
    r.upsert(peer("Qwen2.5-0.5B", "http://a1", 0, 12));
    r.upsert(peer("Qwen2.5-0.5B", "http://a2", 0, 12));
    r.upsert(peer("Qwen2.5-0.5B", "http://b1", 12, 24));
    r.upsert(peer("Qwen2.5-0.5B", "http://b2", 12, 24));

    // A fragile model: covered but one slice has a single replica.
    r.upsert(peer("Llama-8B", "http://c1", 0, 16));
    r.upsert(peer("Llama-8B", "http://c2", 0, 16));
    r.upsert(peer("Llama-8B", "http://d1", 16, 32));

    // An incomplete model: people want DeepSeek but coverage is partial.
    r.upsert(peer("DeepSeek-V3", "http://e1", 0, 10));
    r.upsert(peer("DeepSeek-V3", "http://f1", 30, 40));

    let caps = analyze(&r);
    render_network_state(&caps, 7, 2);
}