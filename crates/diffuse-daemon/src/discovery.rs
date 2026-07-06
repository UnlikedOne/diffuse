use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;

use crate::gossip::gossip_with;
use crate::registry::PeerRegistry;

pub async fn bootstrap(
    sentinels: &[String],
    self_node_id: Vec<u8>,
    registry: &Arc<Mutex<PeerRegistry>>,
) {
    for sentinel in sentinels {
        match gossip_with(sentinel, self_node_id.clone(), registry).await {
            Ok(n) => {
                tracing::info!("bootstrap: got {} peers from sentinel {}", n, sentinel);
            }
            Err(e) => {
                tracing::warn!("bootstrap: sentinel {} unreachable ({})", sentinel, e);
            }
        }
    }
}

pub fn spawn_gossip_loop(
    self_node_id: Vec<u8>,
    self_daemon_endpoint: String,
    registry: Arc<Mutex<PeerRegistry>>,
    interval: Duration,
    fanout: usize,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(interval).await;

            let targets: Vec<String> = {
                let reg = registry.lock().await;
                reg.live_endpoints()
                    .into_iter()
                    .filter(|e| e != &self_daemon_endpoint)
                    .take(fanout)
                    .collect()
            };

            for target in targets {
                match gossip_with(&target, self_node_id.clone(), &registry).await {
                    Ok(n) => {
                        tracing::debug!("gossip: exchanged with {}, got {} peers", target, n);
                    }
                    Err(e) => {
                        tracing::debug!("gossip: {} unreachable ({})", target, e);
                    }
                }
            }

            let pruned = {
                let mut reg = registry.lock().await;
                reg.prune()
            };

            let size = {
                let reg = registry.lock().await;
                reg.len()
            };

            if pruned > 0 {
                tracing::info!("gossip: pruned {} stale peers, registry size {}", pruned, size);
            } else {
                tracing::debug!("gossip: registry size {}", size);
            }
        }
    })
}