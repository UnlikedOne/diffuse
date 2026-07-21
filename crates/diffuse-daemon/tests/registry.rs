use diffuse_daemon::registry::{now_ms, Peer, PeerRegistry};

fn peer(endpoint: &str, last_seen_ms: u64) -> Peer {
    Peer {
        // Identity follows the endpoint: distinct endpoints are distinct nodes,
        // while re-announcing the same endpoint keeps the same node_id (which is
        // what the registry keys on).
        node_id: endpoint.as_bytes().to_vec(),
        daemon_endpoint: endpoint.to_string(),
        worker_endpoint: format!("{}-worker", endpoint),
        model_id: "test-model".to_string(),
        start_layer: 0,
        end_layer: 12,
        total_layers: 0,
        last_seen_ms,
        signature: Vec::new(),
        kx_public: Vec::new(),
        reachable: true,
    }
}

#[test]
fn merge_combines_two_views() {
    let mut a = PeerRegistry::new(60_000);
    a.upsert(peer("http://node1", now_ms()));
    a.upsert(peer("http://node2", now_ms()));

    let incoming = vec![peer("http://node2", now_ms()), peer("http://node3", now_ms())];
    a.merge(incoming);

    assert_eq!(a.len(), 3, "should know node1, node2, node3 after merge");
}

#[test]
fn upsert_keeps_most_recent() {
    let mut r = PeerRegistry::new(60_000);
    r.upsert(peer("http://node1", 1000));
    r.upsert(peer("http://node1", 5000));

    let peers = r.all();
    assert_eq!(peers.len(), 1);
    assert_eq!(peers[0].last_seen_ms, 5000, "should keep the newer timestamp");
}

#[test]
fn prune_removes_stale_peers() {
    let mut r = PeerRegistry::new(10_000);
    let fresh = now_ms();
    let old = now_ms().saturating_sub(20_000);

    r.upsert(peer("http://fresh", fresh));
    r.upsert(peer("http://old", old));
    assert_eq!(r.len(), 2);

    let removed = r.prune();
    assert_eq!(removed, 1, "one stale peer should be pruned");
    assert_eq!(r.len(), 1);
    assert_eq!(r.live_endpoints(), vec!["http://fresh".to_string()]);
}

#[test]
fn replicas_for_slice_filters_correctly() {
    let mut r = PeerRegistry::new(60_000);

    let mut p_a1 = peer("http://a1", now_ms());
    p_a1.start_layer = 0;
    p_a1.end_layer = 12;
    r.upsert(p_a1);

    let mut p_a2 = peer("http://a2", now_ms());
    p_a2.start_layer = 0;
    p_a2.end_layer = 12;
    r.upsert(p_a2);

    let mut p_b1 = peer("http://b1", now_ms());
    p_b1.start_layer = 12;
    p_b1.end_layer = 24;
    r.upsert(p_b1);

    let stage_a = r.replicas_for_slice("test-model", 0, 12);
    assert_eq!(stage_a.len(), 2, "two replicas hold slice 0:12");

    let stage_b = r.replicas_for_slice("test-model", 12, 24);
    assert_eq!(stage_b.len(), 1, "one replica holds slice 12:24");

    let none = r.replicas_for_slice("other-model", 0, 12);
    assert_eq!(none.len(), 0, "no replica for a different model");
}