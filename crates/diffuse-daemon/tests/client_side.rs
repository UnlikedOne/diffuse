use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;

use diffuse_daemon::compute::{request_slice, spawn_compute_server};
use diffuse_daemon::worker::pb::Tensor;
use diffuse_daemon::worker::WorkerHandle;
use diffuse_trust::transport::KeyExchange;

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
}

impl WorkerProc {
    fn spawn(port: u16) -> Self {
        let child = Command::new(python_bin())
            .arg("-m")
            .arg("diffuse_worker")
            .env("DIFFUSE_WORKER_PORT", port.to_string())
            .current_dir(worker_dir())
            .spawn()
            .expect("spawn worker");
        WorkerProc { child }
    }
}

impl Drop for WorkerProc {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

async fn wait_up(endpoint: &str) -> WorkerHandle {
    for _ in 0..40 {
        if let Ok(w) = WorkerHandle::connect(endpoint.to_string()).await {
            return w;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    panic!("worker never came up");
}

#[tokio::test]
async fn prompt_never_leaves_client_only_activations_do() {
    // Client-side worker holds the FIRST slice (embedding + first layers).
    let _client_worker = WorkerProc::spawn(50211);
    // Remote host worker holds the REST of the model.
    let _host_worker = WorkerProc::spawn(50212);

    let mut client_w = wait_up("http://127.0.0.1:50211").await;
    let mut host_w = wait_up("http://127.0.0.1:50212").await;

    let full = client_w.load_slice(MODEL, 0, 2, "").await.expect("probe");
    // Client keeps only layers 0:2 (embedding + 2 blocks), locally.
    client_w.load_slice(MODEL, 0, 2, "").await.expect("client slice");
    // Host holds 2:full.
    host_w.load_slice(MODEL, 2, full, "").await.expect("host slice");

    // The prompt is encoded and embedded LOCALLY on the client.
    let (ids, _eos) = client_w
        .encode("Tell me a secret.", true)
        .await
        .expect("encode locally");

    // Build the input_ids tensor and run the first slice locally.
    let mut data = Vec::new();
    for id in &ids {
        data.extend_from_slice(&id.to_le_bytes());
    }
    let input = Tensor {
        shape: vec![1, ids.len() as i64],
        dtype: "int64".to_string(),
        data,
    };

    // Client executes slice 0:2 locally -> activations. Tokens stay here.
    let local_activations = client_w
        .run_slice(MODEL, 0, 2, "client-session", 0, input, false, 0, false, None)
        .await
        .expect("local first-slice execution");

    // A requester that did not opt into the current wire format must still be
    // answered in float32, the format every released version can parse.
    assert_eq!(
        local_activations.dtype, "float32",
        "a requester that does not announce bf16 support must get float32 back"
    );
    assert_eq!(
        local_activations.shape.len(),
        3,
        "activations are hidden states [1, seq, hidden]"
    );

    // Host daemon exposes an encrypted compute channel over its local worker.
    let host_kx = Arc::new(KeyExchange::generate());
    let host_kx_public = host_kx.public_bytes();
    let host_worker = Arc::new(Mutex::new(host_w));
    let addr: std::net::SocketAddr = "127.0.0.1:50312".parse().unwrap();
    let _server = spawn_compute_server(addr, Arc::clone(&host_kx), host_worker, MODEL.to_string());
    tokio::time::sleep(Duration::from_millis(300)).await;

    // Client sends the ACTIVATIONS (encrypted) to the host for the rest of the model.
    let client_kx = KeyExchange::generate();
    let mut compute_client = diffuse_daemon::compute::connect_compute("http://127.0.0.1:50312")
        .await
        .expect("connect to the encrypted compute channel");
    let (logits, _compute_ms, _peer_version) = request_slice(
        &mut compute_client,
        &host_kx_public,
        &client_kx,
        MODEL,
        2,
        full,
        "client-session",
        &local_activations,
        0,
        false,
        None,
    )
    .await
    .expect("remote completion over encrypted channel");

    // The host returned logits over the full vocabulary: generation is possible
    // even though the host never saw the prompt tokens.
    assert_eq!(logits.shape.len(), 3, "final logits [1, seq, vocab]");
    assert!(logits.shape[2] > 100_000, "vocab dimension for Qwen");
}