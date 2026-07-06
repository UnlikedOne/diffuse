use diffuse_daemon::capacity::analyze;
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
fn robust_model_is_servable_and_robust() {
    let mut r = PeerRegistry::new(600_000);
    // Model "big": slices 0:12 and 12:24, each with 2 replicas. Fully covered, robust.
    r.upsert(peer("big", "http://a1", 0, 12));
    r.upsert(peer("big", "http://a2", 0, 12));
    r.upsert(peer("big", "http://b1", 12, 24));
    r.upsert(peer("big", "http://b2", 12, 24));

    let caps = analyze(&r);
    let big = caps.iter().find(|c| c.model_id == "big").unwrap();

    assert!(big.servable, "big should be servable");
    assert_eq!(big.total_layers, 24);
    assert_eq!(big.min_replicas, 2);
    assert!(big.is_robust(2), "big should be robust at k=2");
    assert!(big.missing_slices().is_empty());
}

#[test]
fn model_with_missing_slice_is_not_servable() {
    let mut r = PeerRegistry::new(600_000);
    // Model "broken": has 0:12 but nobody holds 12:24. Not servable.
    r.upsert(peer("broken", "http://a1", 0, 12));
    r.upsert(peer("broken", "http://a2", 0, 12));
    // Announce that 12:24 exists as a slice but with no live holder:
    // we simulate this by having a peer that declares total via a higher end,
    // but here we simply omit 12:24, so total_layers is 12 and it IS covered.
    // To truly test a gap, add a peer for 24:36 leaving 12:24 missing.
    r.upsert(peer("broken", "http://c1", 24, 36));

    let caps = analyze(&r);
    let broken = caps.iter().find(|c| c.model_id == "broken").unwrap();

    assert_eq!(broken.total_layers, 36);
    assert!(!broken.servable, "broken has a gap at 12:24, not servable");
}

#[test]
fn fragile_model_is_servable_but_not_robust() {
    let mut r = PeerRegistry::new(600_000);
    // Model "fragile": fully covered but slice 12:24 has only 1 replica.
    r.upsert(peer("fragile", "http://a1", 0, 12));
    r.upsert(peer("fragile", "http://a2", 0, 12));
    r.upsert(peer("fragile", "http://b1", 12, 24));

    let caps = analyze(&r);
    let f = caps.iter().find(|c| c.model_id == "fragile").unwrap();

    assert!(f.servable, "fragile is fully covered, so servable");
    assert_eq!(f.min_replicas, 1, "weakest slice has 1 replica");
    assert!(!f.is_robust(2), "not robust at k=2");
    assert_eq!(f.weakest_slice, Some((12, 24)), "12:24 is the weak point");
}