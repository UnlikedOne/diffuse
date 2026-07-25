use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::Duration;

use std::sync::Arc;

use diffuse_daemon::compute::spawn_compute_server;
use diffuse_daemon::orchestrator::build_from_registry;
use diffuse_daemon::registry::{now_ms, Peer, PeerRegistry, WIRE_VERSION};
use diffuse_daemon::worker::WorkerHandle;
use diffuse_trust::transport::KeyExchange;
use tokio::sync::Mutex;

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

fn peer(
    node_id: u8,
    daemon_port: u16,
    worker_ep: &str,
    kx_public: [u8; 32],
    start: u32,
    end: u32,
    total: u32,
) -> Peer {
    Peer {
        node_id: vec![node_id],
        daemon_endpoint: format!("http://127.0.0.1:{}", daemon_port),
        worker_endpoint: worker_ep.to_string(),
        model_id: MODEL.to_string(),
        start_layer: start,
        end_layer: end,
        total_layers: total,
        last_seen_ms: now_ms(),
        signature: Vec::new(),
        kx_public: kx_public.to_vec(),
        reachable: true,
    }
}

#[tokio::test]
async fn orchestrator_builds_from_discovered_registry() {
    let w_a = WorkerProc::spawn(50191);
    let w_b = WorkerProc::spawn(50192);

    wait_up(&w_a.endpoint()).await;
    wait_up(&w_b.endpoint()).await;

    let mut worker_a = WorkerHandle::connect(w_a.endpoint()).await.expect("probe");
    let full = worker_a
        .load_slice(MODEL, 0, 12, "")
        .await
        .expect("probe load");
    let mid = full / 2;
    worker_a
        .load_slice(MODEL, 0, mid, "")
        .await
        .expect("load stage a");
    let mut worker_b = WorkerHandle::connect(w_b.endpoint()).await.expect("worker b");
    worker_b
        .load_slice(MODEL, mid, full, "")
        .await
        .expect("load stage b");

    // Each peer announces its daemon endpoint; the route derives the compute
    // endpoint from it by adding 1000 to the port, so that is where the
    // encrypted compute servers have to listen.
    let kx_a = Arc::new(KeyExchange::generate());
    let kx_b = Arc::new(KeyExchange::generate());
    let pub_a = kx_a.public_bytes();
    let pub_b = kx_b.public_bytes();
    spawn_compute_server(
        "127.0.0.1:51191".parse().unwrap(),
        Arc::clone(&kx_a),
        Arc::new(Mutex::new(worker_a)),
        MODEL.to_string(),
    );
    spawn_compute_server(
        "127.0.0.1:51192".parse().unwrap(),
        Arc::clone(&kx_b),
        Arc::new(Mutex::new(worker_b)),
        MODEL.to_string(),
    );
    tokio::time::sleep(Duration::from_millis(500)).await;

    let mut registry = PeerRegistry::new(600_000);
    registry.upsert(peer(1, 50191, &w_a.endpoint(), pub_a, 0, mid, full));
    registry.upsert(peer(2, 50192, &w_b.endpoint(), pub_b, mid, full, full));

    let mut orch = build_from_registry(MODEL, &registry, 2, vec![], None)
        .await
        .expect("should build orchestrator from registry");

    assert_eq!(orch.stages.len(), 2, "should have discovered two slices");

    // The route is made of remote replicas, so tokenizing goes through a direct
    // worker connection rather than the orchestrator.
    let mut tokenizer = WorkerHandle::connect(w_a.endpoint()).await.expect("tokenizer");
    let (ids, _eos) = tokenizer.encode("Say hi.", true).await.expect("encode");

    let out = orch
        .generate(&ids, 15, "routing-session", Some(QWEN_EOS))
        .await
        .expect("generation over discovered route");

    assert!(out.len() > ids.len(), "should generate tokens over discovered route");

    let text = tokenizer
        .decode(&out[ids.len()..], true)
        .await
        .expect("decode");
    assert!(!text.is_empty());
}