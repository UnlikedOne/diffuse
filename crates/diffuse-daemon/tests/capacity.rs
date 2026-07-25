use diffuse_daemon::capacity::analyze;
use diffuse_daemon::registry::{now_ms, Peer, PeerRegistry};

fn peer(model: &str, ep: &str, start: u32, end: u32) -> Peer {
    peer_with_total(model, ep, start, end, 0)
}

fn peer_with_total(model: &str, ep: &str, start: u32, end: u32, total: u32) -> Peer {
    Peer {
        // The registry keys peers by node_id, so distinct peers must carry
        // distinct ids or they collapse into one. Derive it from the endpoint.
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
        protocol_version: diffuse_daemon::registry::WIRE_VERSION,
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
fn partial_model_reports_true_total_and_is_incomplete() {
    let mut r = PeerRegistry::new(600_000);
    // A single node serving layers 0:6 of a 64-layer model. The worker knows the
    // model has 64 layers and gossips that in total_layers.
    r.upsert(peer_with_total("big", "http://a1", 0, 6, 64));

    let caps = analyze(&r);
    let cap = caps.iter().find(|c| c.model_id == "big").unwrap();

    // Without the propagated total this would infer 6 and look complete (6/6).
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
    // Legacy peers that predate the total_layers field (report 0). Behaviour
    // must match the old inference: total = highest end_layer.
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
    // Holds 0:12 and 24:36 of a 36-layer model, leaving 12:24 open in the middle.
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