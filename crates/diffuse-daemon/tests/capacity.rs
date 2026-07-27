use diffuse_daemon::capacity::analyze;
use diffuse_daemon::registry::{now_ms, Peer, PeerRegistry};

fn peer(model: &str, ep: &str, start: u32, end: u32) -> Peer {
    peer_with_total(model, ep, start, end, 0)
}

fn peer_with_total(model: &str, ep: &str, start: u32, end: u32, total: u32) -> Peer {
    Peer {
        node_id: ep.as_bytes().to_vec(),
        daemon_endpoint: ep.to_string(),
        worker_endpoint: format!("{}-w", ep),
        model_id: model.to_string(),
        start_layer: start,
        end_layer: end,
        total_layers: total,
        last_seen_ms: now_ms(),
        signature: Vec::new(),
        kx_public: Vec::new(),
        reachable: true,
    }
}

#[test]
fn robust_model_is_servable_and_robust() {
    let mut r = PeerRegistry::new(600_000);
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
    r.upsert(peer("broken", "http://a1", 0, 12));
    r.upsert(peer("broken", "http://a2", 0, 12));
    r.upsert(peer("broken", "http://c1", 24, 36));

    let caps = analyze(&r);
    let broken = caps.iter().find(|c| c.model_id == "broken").unwrap();

    assert_eq!(broken.total_layers, 36);
    assert!(!broken.servable, "broken has a gap at 12:24, not servable");
}

#[test]
fn partial_model_reports_true_total_and_is_incomplete() {
    let mut r = PeerRegistry::new(600_000);
    r.upsert(peer_with_total("big", "http://a1", 0, 6, 64));

    let caps = analyze(&r);
    let cap = caps.iter().find(|c| c.model_id == "big").unwrap();

    assert_eq!(cap.total_layers, 64, "total comes from the worker, not end_layer");
    assert!(!cap.servable, "0:6 of a 64-layer model is not servable");
    assert_eq!(
        cap.coverage_gaps(),
        vec![(6, 64)],
        "everything past layer 6 is missing"
    );
}

#[test]
fn falls_back_to_end_layer_when_total_is_unknown() {
    let mut r = PeerRegistry::new(600_000);
    r.upsert(peer("legacy", "http://a1", 0, 12));
    r.upsert(peer("legacy", "http://b1", 12, 24));

    let caps = analyze(&r);
    let cap = caps.iter().find(|c| c.model_id == "legacy").unwrap();

    assert_eq!(cap.total_layers, 24, "inferred from the highest end_layer");
    assert!(cap.servable, "fully covered under the inferred total");
    assert!(cap.coverage_gaps().is_empty());
}

#[test]
fn coverage_gaps_reports_interior_holes() {
    let mut r = PeerRegistry::new(600_000);
    r.upsert(peer_with_total("swiss", "http://a1", 0, 12, 36));
    r.upsert(peer_with_total("swiss", "http://c1", 24, 36, 36));

    let caps = analyze(&r);
    let cap = caps.iter().find(|c| c.model_id == "swiss").unwrap();

    assert_eq!(cap.total_layers, 36);
    assert!(!cap.servable);
    assert_eq!(cap.coverage_gaps(), vec![(12, 24)]);
}

#[test]
fn fragile_model_is_servable_but_not_robust() {
    let mut r = PeerRegistry::new(600_000);
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
