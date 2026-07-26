use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::Duration;

use diffuse_daemon::orchestrator::{Orchestrator, Replica, Stage};
use diffuse_daemon::worker::WorkerHandle;

const MODEL: &str = "Qwen/Qwen2.5-0.5B-Instruct";
const QWEN_EOS: i64 = 151645;

fn worker_dir() -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest)
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
async fn generation_survives_dead_replica() {
    let mut w_a1 = WorkerProc::spawn(50161);
    let w_a2 = WorkerProc::spawn(50162);
    let w_b1 = WorkerProc::spawn(50163);

    let mut probe = wait_for_worker(&w_a1.endpoint())
        .await
        .expect("probe up");
    let full = probe.load_slice(MODEL, 0, 12, "").await.expect("probe load");
    let mid = full / 2;

    let a1 = connect_loaded(w_a1.endpoint(), 0, mid).await;
    let a2 = connect_loaded(w_a2.endpoint(), 0, mid).await;
    let b1 = connect_loaded(w_b1.endpoint(), mid, full).await;

    let stage_a = Stage {
        start_layer: 0,
        end_layer: mid,
        replicas: vec![
            Replica::Local { worker: a1, alive: true },
            Replica::Local { worker: a2, alive: true },
        ],
        last_compute_ms: 0,
        last_network_ms: 0,
    };
    let stage_b = Stage {
        start_layer: mid,
        end_layer: full,
        replicas: vec![Replica::Local { worker: b1, alive: true }],
        last_compute_ms: 0,
        last_network_ms: 0,
    };

    let mut orch = Orchestrator {
        model_id: MODEL.to_string(),
        stages: vec![stage_a, stage_b],
        spare_endpoints: vec![],
        target_replication: 2,
        session_kx: std::sync::Arc::new(diffuse_trust::transport::KeyExchange::generate()),
        last_forward_compute_ms: 0,
        last_forward_network_ms: 0,
        chain_enabled: false,
        session_prefill: None,
        session_positions: None,
    };

    let (ids, _eos) = orch
        .encode_via_any("Say hello in one sentence.", true)
        .await
        .expect("encode");

    let reference = orch
        .generate(&ids, 20, "reference-session", Some(QWEN_EOS))
        .await
        .expect("reference generation with every replica alive");

    let mut produced = 0usize;
    let out = orch
        .generate_streaming(&ids, 20, "chaos-session", Some(QWEN_EOS), |_tok| {
            produced += 1;
            if produced == 5 {
                w_a1.kill();
            }
        })
        .await
        .expect("generation should survive a replica dying mid-session");

    assert!(out.len() > ids.len(), "should have generated new tokens");

    assert_eq!(
        out, reference,
        "a replica dying mid-session must not change the tokens: the standby holds no KV cache \
         for the session, so the prefix has to be replayed rather than silently continued"
    );

    let text = orch
        .decode_via_any(&out[ids.len()..], true)
        .await
        .expect("decode");
    assert!(!text.is_empty(), "decoded text should not be empty");

    let live_a = orch.stages[0].replicas.iter().filter(|r| r.is_alive()).count();
    assert_eq!(live_a, 1, "stage A should have exactly one live replica left");

    let _ = w_a2;
    let _ = w_b1;
}