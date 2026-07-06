use diffuse_daemon::capacity::{analyze, assign_slice};
use diffuse_daemon::registry::{now_ms, Peer, PeerRegistry};

fn peer(model: &str, ep: &str, start: u32, end: u32) -> Peer {
    Peer {
        node_id: vec![1],
        daemon_endpoint: ep.to_string(),
        worker_endpoint: format!("{}-w", ep),
        model_id: model.to_string(),
        start_layer: start,
        end_layer: end,
        last_seen_ms: now_ms(),
        signature: Vec::new(),
        kx_public: Vec::new(),
    }
}

#[test]
fn absent_model_bootstraps_from_start_bounded_by_capacity() {
    let r = PeerRegistry::new(600_000);
    let caps = analyze(&r);
    // A weak machine (can hold 3 layers) is first to host a 60-layer model.
    let a = assign_slice(&caps, "deepseek", 60, 3, 2).unwrap();
    assert_eq!(a.start, 0);
    assert_eq!(a.end, 3, "weak machine takes only what it can hold");
}

#[test]
fn weak_machine_fills_part_of_a_gap() {
    let mut r = PeerRegistry::new(600_000);
    // Model of 60 layers, only 0:20 is covered. Gap is 20:60.
    r.upsert(peer("deepseek", "http://a", 0, 20));
    let caps = analyze(&r);

    // Weak machine can hold 5 layers -> takes 20:25 (part of the gap).
    let a = assign_slice(&caps, "deepseek", 60, 5, 2).unwrap();
    assert_eq!(a.start, 20);
    assert_eq!(a.end, 25, "takes a 5-layer bite of the gap");
}

#[test]
fn strong_machine_fills_whole_gap() {
    let mut r = PeerRegistry::new(600_000);
    // 60-layer model, 0:50 covered, gap 50:60 (10 layers).
    r.upsert(peer("deepseek", "http://a", 0, 50));
    let caps = analyze(&r);

    // Strong machine can hold 40 layers -> fills the whole 10-layer gap.
    let a = assign_slice(&caps, "deepseek", 60, 40, 2).unwrap();
    assert_eq!(a.start, 50);
    assert_eq!(a.end, 60, "strong machine fills the entire remaining gap");
}

#[test]
fn zero_capacity_gets_no_assignment() {
    let r = PeerRegistry::new(600_000);
    let caps = analyze(&r);
    // A machine that cannot hold even one layer of this model.
    assert!(assign_slice(&caps, "deepseek", 60, 0, 2).is_none());
}