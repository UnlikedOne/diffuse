mod common;

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;

use common::{new_key, signed_peer};
use diffuse_daemon::gossip::{gossip_with, spawn_gossip_server};
use diffuse_daemon::registry::{now_ms, Peer, PeerRegistry};

#[tokio::test]
async fn two_daemons_converge_via_authenticated_gossip() {
    let reg_a = Arc::new(Mutex::new(PeerRegistry::new(60_000)));
    let reg_b = Arc::new(Mutex::new(PeerRegistry::new(60_000)));

    let key_a1 = new_key();
    let key_a2 = new_key();
    let key_b1 = new_key();
    let key_b2 = new_key();

    {
        let mut a = reg_a.lock().await;
        a.upsert(signed_peer(&key_a1, "http://127.0.0.1:60001", "http://w1", "m", 0, 12));
        a.upsert(signed_peer(&key_a2, "http://only-a-knows", "http://w2", "m", 0, 12));
    }
    {
        let mut b = reg_b.lock().await;
        b.upsert(signed_peer(&key_b1, "http://127.0.0.1:60002", "http://w3", "m", 12, 24));
        b.upsert(signed_peer(&key_b2, "http://only-b-knows", "http://w4", "m", 12, 24));
    }

    let addr_b: std::net::SocketAddr = "127.0.0.1:60002".parse().unwrap();
    let _server_b = spawn_gossip_server(addr_b, Arc::clone(&reg_b), diffuse_daemon::relay::RelayState::new());

    tokio::time::sleep(Duration::from_millis(300)).await;

    let accepted = gossip_with("http://127.0.0.1:60002", vec![1], &reg_a)
        .await
        .expect("gossip should succeed");

    assert!(accepted >= 2, "A should accept B's authentic peers");

    {
        let a = reg_a.lock().await;
        let eps = a.live_endpoints();
        assert!(
            eps.iter().any(|e| e == "http://only-b-knows"),
            "A should now know what only B knew"
        );
        assert!(
            eps.iter().any(|e| e == "http://only-a-knows"),
            "A should still know its own peers"
        );
    }
}

#[tokio::test]
async fn unsigned_peer_is_rejected() {
    let reg_a = Arc::new(Mutex::new(PeerRegistry::new(60_000)));
    let reg_b = Arc::new(Mutex::new(PeerRegistry::new(60_000)));

    let forged = Peer {
        node_id: vec![4, 2, 4, 2],
        daemon_endpoint: "http://malicious".to_string(),
        worker_endpoint: "http://evil-worker".to_string(),
        model_id: "m".to_string(),
        start_layer: 0,
        end_layer: 12,
        total_layers: 0,
        last_seen_ms: now_ms(),
        signature: Vec::new(),
        kx_public: Vec::new(),
        reachable: true,
    };
    {
        let mut b = reg_b.lock().await;
        b.upsert(forged);
    }

    let addr_b: std::net::SocketAddr = "127.0.0.1:60012".parse().unwrap();
    let _server_b = spawn_gossip_server(addr_b, Arc::clone(&reg_b), diffuse_daemon::relay::RelayState::new());
    tokio::time::sleep(Duration::from_millis(300)).await;

    let accepted = gossip_with("http://127.0.0.1:60012", vec![1], &reg_a)
        .await
        .expect("gossip call should succeed");

    assert_eq!(accepted, 0, "no unsigned peer should be accepted");

    let a = reg_a.lock().await;
    assert!(
        !a.live_endpoints().iter().any(|e| e == "http://malicious"),
        "the forged peer must never enter the registry"
    );
}

#[tokio::test]
async fn tampered_peer_is_rejected() {
    let reg_a = Arc::new(Mutex::new(PeerRegistry::new(60_000)));
    let reg_b = Arc::new(Mutex::new(PeerRegistry::new(60_000)));

    let key = new_key();
    let mut tampered = signed_peer(&key, "http://victim", "http://w", "m", 0, 12);
    tampered.end_layer = 24;
    {
        let mut b = reg_b.lock().await;
        b.upsert(tampered);
    }

    let addr_b: std::net::SocketAddr = "127.0.0.1:60013".parse().unwrap();
    let _server_b = spawn_gossip_server(addr_b, Arc::clone(&reg_b), diffuse_daemon::relay::RelayState::new());
    tokio::time::sleep(Duration::from_millis(300)).await;

    let accepted = gossip_with("http://127.0.0.1:60013", vec![1], &reg_a)
        .await
        .expect("gossip call should succeed");

    assert_eq!(accepted, 0, "a peer whose data was tampered after signing must be rejected");

    let a = reg_a.lock().await;
    assert!(
        !a.live_endpoints().iter().any(|e| e == "http://victim"),
        "the tampered peer must not enter the registry"
    );
}
