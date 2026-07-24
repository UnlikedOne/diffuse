use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::Duration;

use diffuse_daemon::orchestrator::{Orchestrator, Replica, Stage};
use diffuse_daemon::worker::WorkerHandle;

const MODEL: &str = "Qwen/Qwen2.5-0.5B-Instruct";

fn worker_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("worker")
        .canonicalize()
        .expect("worker dir")
}

fn python_bin() -> PathBuf {
    worker_dir().join(".venv").join("bin").join("python")
}

struct WorkerProc {
    child: Child,
    port: u16,
}

impl WorkerProc {
    fn spawn(port: u16) -> Self {
        let child = Command::new(python_bin())
            .arg("-m")
            .arg("diffuse_worker")
            .env("DIFFUSE_WORKER_PORT", port.to_string())
            .current_dir(worker_dir())
            .spawn()
            .expect("failed to spawn worker");
        WorkerProc { child, port }
    }

    fn endpoint(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for WorkerProc {
    fn drop(&mut self) {
        self.kill();
    }
}

async fn wait_for_worker(endpoint: &str) -> anyhow::Result<WorkerHandle> {
    for _ in 0..40 {
        if let Ok(w) = WorkerHandle::connect(endpoint.to_string()).await {
            return Ok(w);
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    anyhow::bail!("worker {} never came up", endpoint)
}

async fn connect_loaded(endpoint: String, start: u32, end: u32) -> WorkerHandle {
    let mut worker = wait_for_worker(&endpoint).await.expect("worker up");
    worker.load_slice(MODEL, start, end, "").await.expect("load");
    worker
}

#[tokio::test]
async fn health_sweep_detects_death_proactively() {
    let mut w_a1 = WorkerProc::spawn(50171);
    let w_a2 = WorkerProc::spawn(50172);
    let w_b1 = WorkerProc::spawn(50173);

    let mut probe = wait_for_worker(&w_a1.endpoint()).await.expect("probe up");
    let full = probe.load_slice(MODEL, 0, 12, "").await.expect("probe load");
    let mid = full / 2;

    let a1 = connect_loaded(w_a1.endpoint(), 0, mid).await;
    let a2 = connect_loaded(w_a2.endpoint(), 0, mid).await;
    let b1 = connect_loaded(w_b1.endpoint(), mid, full).await;

    let mut orch = Orchestrator {
        model_id: MODEL.to_string(),
        stages: vec![
            Stage {
                start_layer: 0,
                end_layer: mid,
                replicas: vec![
                    Replica::Local { worker: a1, alive: true },
                    Replica::Local { worker: a2, alive: true },
                ],
                last_compute_ms: 0,
                last_network_ms: 0,
            },
            Stage {
                start_layer: mid,
                end_layer: full,
                replicas: vec![Replica::Local { worker: b1, alive: true }],
                last_compute_ms: 0,
                last_network_ms: 0,
            },
        ],
        spare_endpoints: vec![],
        target_replication: 2,
        session_kx: std::sync::Arc::new(diffuse_trust::transport::KeyExchange::generate()),
        last_forward_compute_ms: 0,
        last_forward_network_ms: 0,
        chain_enabled: false,
    };

    orch.health_sweep().await;
    assert_eq!(orch.coverage(), vec![2, 1], "all replicas should be alive initially");
    assert!(orch.is_servable(), "model should be servable with full coverage");

    w_a1.kill();
    tokio::time::sleep(Duration::from_millis(500)).await;

    orch.health_sweep().await;
    assert_eq!(
        orch.coverage(),
        vec![1, 1],
        "sweep should detect the dead replica proactively, without generating"
    );
    assert!(
        orch.is_servable(),
        "model still servable: one replica left on each stage"
    );

    let _ = w_a2;
    let _ = w_b1;
}