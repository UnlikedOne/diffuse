use diffuse_daemon::capacity::{analyze, best_servable_model, fallback_after_loss};
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
fn best_servable_is_the_largest_fully_covered() {
    let mut r = PeerRegistry::new(600_000);
    // small: 0:12, fully servable (12 layers).
    r.upsert(peer("small", "http://s", 0, 12));
    // big: 0:24 covered, but 24:48 has no holder -> gap -> NOT servable.
    r.upsert(peer("big", "http://a", 0, 24));
    r.upsert(peer("big", "http://b", 48, 60));

    let caps = analyze(&r);
    let best = best_servable_model(&caps).unwrap();
    assert_eq!(best.model_id, "small", "big has a gap (24:48), only small is servable");
}

#[test]
fn no_fallback_while_current_is_servable() {
    let mut r = PeerRegistry::new(600_000);
    r.upsert(peer("current", "http://a", 0, 12));

    let caps = analyze(&r);
    assert!(
        fallback_after_loss(&caps, "current").is_none(),
        "no fallback needed while current is servable"
    );
}

#[test]
fn falls_back_when_current_becomes_unservable() {
    let mut r = PeerRegistry::new(600_000);
    // "big" lost a slice: only 0:12 remains of a 0:24 model -> not servable.
    r.upsert(peer("big", "http://a", 0, 12));
    // announce big also has a 12:24 slice-holder that is gone (we just omit it)
    r.upsert(peer("big", "http://a2", 24, 36)); // creates a gap, big not servable
    // "small" is fully servable as a fallback.
    r.upsert(peer("small", "http://s", 0, 12));

    let caps = analyze(&r);
    let fb = fallback_after_loss(&caps, "big").expect("should fall back");
    assert_eq!(fb.model_id, "small", "network falls back to the servable model");
}