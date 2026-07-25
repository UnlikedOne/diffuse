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

fn ids_tensor(ids: &[i64]) -> Tensor {
    let mut data = Vec::new();
    for id in ids {
        data.extend_from_slice(&id.to_le_bytes());
    }
    Tensor {
        shape: vec![1, ids.len() as i64],
        dtype: "int64".to_string(),
        data,
    }
}

#[tokio::test]
async fn daemon_delegates_slice_over_encrypted_channel() {
    let _worker = WorkerProc::spawn(50201);
    let mut w = wait_up("http://127.0.0.1:50201").await;
    let full = w.load_slice(MODEL, 0, 12, "").await.expect("load probe");
    let mid = full / 2;
    w.load_slice(MODEL, 0, mid, "").await.expect("load slice");

    // Host daemon identity and compute server, driving its local worker.
    let host_kx = Arc::new(KeyExchange::generate());
    let host_kx_public = host_kx.public_bytes();
    let worker = Arc::new(Mutex::new(w));
    let addr: std::net::SocketAddr = "127.0.0.1:50301".parse().unwrap();
    let _server = spawn_compute_server(addr, Arc::clone(&host_kx), worker, MODEL.to_string());
    tokio::time::sleep(Duration::from_millis(300)).await;

    // Requester daemon: its own kx identity, delegates a slice to the host.
    let my_kx = KeyExchange::generate();
    let input = ids_tensor(&[9707, 11, 1879]);

    let mut client = diffuse_daemon::compute::connect_compute("http://127.0.0.1:50301")
        .await
        .expect("connect to the encrypted compute channel");
    let (out, _compute_ms, peer_version) = request_slice(
        &mut client,
        &host_kx_public,
        &my_kx,
        MODEL,
        0,
        mid,
        "enc-session",
        &input,
        0,
        false,
    )
    .await
    .expect("encrypted delegation should succeed");

    assert_eq!(
        peer_version,
        diffuse_daemon::registry::WIRE_VERSION,
        "a host must announce the wire it speaks in its own encrypted answer, \
         which is the only statement about itself that cannot be forged or \
         stripped by a node relaying gossip"
    );
    assert!(!out.data.is_empty(), "should receive activations back");
    assert_eq!(out.shape.len(), 3, "output should be hidden states [1, seq, hidden]");
}