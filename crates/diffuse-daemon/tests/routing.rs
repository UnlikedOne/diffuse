use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::Duration;

use diffuse_daemon::orchestrator::build_from_registry;
use diffuse_daemon::registry::{now_ms, Peer, PeerRegistry};
use diffuse_daemon::worker::WorkerHandle;

const MODEL: &str = "Qwen/Qwen2.5-0.5B-Instruct";
const QWEN_EOS: i64 = 151645;

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
}

impl Drop for WorkerProc {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

async fn wait_up(endpoint: &str) {
    for _ in 0..40 {
        if WorkerHandle::connect(endpoint.to_string()).await.is_ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    panic!("worker {} never came up", endpoint);
}

fn peer(worker_ep: &str, start: u32, end: u32) -> Peer {
    Peer {
        node_id: vec![7],
        daemon_endpoint: format!("{}-daemon", worker_ep),
        worker_endpoint: worker_ep.to_string(),
        model_id: MODEL.to_string(),
        start_layer: start,
        end_layer: end,
        total_layers: 0,
        last_seen_ms: now_ms(),
        signature: Vec::new(),
        kx_public: Vec::new(),
        reachable: true,
    }
}

#[tokio::test]
async fn orchestrator_builds_from_discovered_registry() {
    let w_a = WorkerProc::spawn(50191);
    let w_b = WorkerProc::spawn(50192);

    wait_up(&w_a.endpoint()).await;
    wait_up(&w_b.endpoint()).await;

    let mut probe = WorkerHandle::connect(w_a.endpoint()).await.expect("probe");
    let full = probe.load_slice(MODEL, 0, 12, "").await.expect("probe load");
    let mid = full / 2;

    let mut registry = PeerRegistry::new(600_000);
    registry.upsert(peer(&w_a.endpoint(), 0, mid));
    registry.upsert(peer(&w_b.endpoint(), mid, full));

    let mut orch = build_from_registry(MODEL, &registry, 2, vec![])
        .await
        .expect("should build orchestrator from registry");

    assert_eq!(orch.stages.len(), 2, "should have discovered two slices");

    let (ids, _eos) = orch
        .encode_via_any("Say hi.", true)
        .await
        .expect("encode");
    let out = orch
        .generate(&ids, 15, "routing-session", Some(QWEN_EOS))
        .await
        .expect("generation over discovered route");

    assert!(out.len() > ids.len(), "should generate tokens over discovered route");

    let text = orch
        .decode_via_any(&out[ids.len()..], true)
        .await
        .expect("decode");
    assert!(!text.is_empty());
}