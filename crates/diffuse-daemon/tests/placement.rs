use diffuse_daemon::capacity::{analyze, recommend_placement, Placement};
use diffuse_daemon::registry::{now_ms, Peer, PeerRegistry};

fn peer(model: &str, ep: &str, start: u32, end: u32) -> Peer {
    Peer {
        // Distinct id per peer — the registry keys on node_id, so a shared id
        // would collapse these into a single entry.
        node_id: ep.as_bytes().to_vec(),
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

#[test]
fn new_node_fills_missing_slice_first() {
    let mut r = PeerRegistry::new(600_000);
    // 0:12 covered, 12:24 missing, 24:36 covered. The gap must be filled first.
    r.upsert(peer("m", "http://a", 0, 12));
    r.upsert(peer("m", "http://c", 24, 36));

    let caps = analyze(&r);
    let placement = recommend_placement(&caps, "m", 2);

    assert_eq!(
        placement,
        Placement::FillMissing { start: 12, end: 24 },
        "should direct the node to the missing slice"
    );
}

#[test]
fn new_node_reinforces_weak_slice_when_all_covered() {
    let mut r = PeerRegistry::new(600_000);
    // Fully covered, but 12:24 has only 1 replica while 0:12 has 2.
    r.upsert(peer("m", "http://a1", 0, 12));
    r.upsert(peer("m", "http://a2", 0, 12));
    r.upsert(peer("m", "http://b1", 12, 24));

    let caps = analyze(&r);
    let placement = recommend_placement(&caps, "m", 2);

    assert_eq!(
        placement,
        Placement::ReinforceWeak { start: 12, end: 24, current_replicas: 1 },
        "should reinforce the weakest slice"
    );
}

#[test]
fn unknown_model_returns_not_present() {
    let r = PeerRegistry::new(600_000);
    let caps = analyze(&r);
    assert_eq!(
        recommend_placement(&caps, "ghost", 2),
        Placement::ModelNotPresent
    );
}