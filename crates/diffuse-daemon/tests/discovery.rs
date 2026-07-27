mod common;

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;

use common::{new_key, signed_peer};
use diffuse_daemon::discovery::spawn_gossip_loop;
use diffuse_daemon::gossip::spawn_gossip_server;
use diffuse_daemon::registry::PeerRegistry;
use ed25519_dalek::SigningKey;

async fn make_node(
    self_key: &SigningKey,
    endpoint: &str,
    port: u16,
    known: Vec<(&SigningKey, &str)>,
) -> Arc<Mutex<PeerRegistry>> {
    let reg = Arc::new(Mutex::new(PeerRegistry::new(600_000)));
    {
        let mut r = reg.lock().await;
        r.upsert(signed_peer(self_key, endpoint, &format!("{}-w", endpoint), "m", 0, 12));
        for (key, ep) in known {
            r.upsert(signed_peer(key, ep, &format!("{}-w", ep), "m", 0, 12));
        }
    }
    let addr: std::net::SocketAddr = format!("127.0.0.1:{}", port).parse().unwrap();
    spawn_gossip_server(addr, Arc::clone(&reg), diffuse_daemon::relay::RelayState::new());
    reg
}

#[tokio::test]
async fn three_nodes_fully_discover_via_transitivity() {
    let ep1 = "http://127.0.0.1:60101";
    let ep2 = "http://127.0.0.1:60102";
    let ep3 = "http://127.0.0.1:60103";

    let key1 = new_key();
    let key2 = new_key();
    let key3 = new_key();

    let reg1 = make_node(&key1, ep1, 60101, vec![(&key2, ep2)]).await;
    let reg2 = make_node(&key2, ep2, 60102, vec![(&key3, ep3)]).await;
    let reg3 = make_node(&key3, ep3, 60103, vec![(&key1, ep1)]).await;

    tokio::time::sleep(Duration::from_millis(300)).await;

    spawn_gossip_loop(vec![1], ep1.to_string(), Arc::clone(&reg1), Duration::from_millis(200), 4);
    spawn_gossip_loop(vec![2], ep2.to_string(), Arc::clone(&reg2), Duration::from_millis(200), 4);
    spawn_gossip_loop(vec![3], ep3.to_string(), Arc::clone(&reg3), Duration::from_millis(200), 4);

    tokio::time::sleep(Duration::from_secs(3)).await;

    for (name, reg) in [("node1", &reg1), ("node2", &reg2), ("node3", &reg3)] {
        let r = reg.lock().await;
        let eps = r.live_endpoints();
        assert!(eps.iter().any(|e| e == ep1), "{} should know node1", name);
        assert!(eps.iter().any(|e| e == ep2), "{} should know node2", name);
        assert!(eps.iter().any(|e| e == ep3), "{} should know node3", name);
    }
}
