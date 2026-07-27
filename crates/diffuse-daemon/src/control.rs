use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;

use crate::orchestrator::Orchestrator;

pub struct ControlPlane {
    pub orchestrator: Arc<Mutex<Orchestrator>>,
}

impl ControlPlane {
    pub fn new(orchestrator: Orchestrator) -> Self {
        Self {
            orchestrator: Arc::new(Mutex::new(orchestrator)),
        }
    }

    pub fn spawn_health_loop(&self, interval: Duration) -> tokio::task::JoinHandle<()> {
        let orch = Arc::clone(&self.orchestrator);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(interval).await;
                let mut guard = orch.lock().await;
                guard.health_sweep().await;
                let coverage_before = guard.coverage();
                if !coverage_before.iter().all(|&c| c >= guard.target_replication) {
                    let repaired = guard.repair().await;
                    if repaired > 0 {
                        tracing::info!(
                            "control: repaired {} slice(s), coverage now {:?}",
                            repaired,
                            guard.coverage()
                        );
                    }
                }
                let coverage = guard.coverage();
                let servable = guard.is_servable();
                drop(guard);
                tracing::info!(
                    "control: sweep done, coverage {:?}, servable {}",
                    coverage,
                    servable
                );
            }
        })
    }
}
