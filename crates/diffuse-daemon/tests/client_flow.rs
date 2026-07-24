use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;

use diffuse_daemon::client_flow::{ClientSession, RemoteStage};
use diffuse_daemon::compute::spawn_compute_server;
use diffuse_daemon::worker::WorkerHandle;
use diffuse_trust::transport::KeyExchange;

const MODEL: &str = "Qwen/Qwen2.5-0.5B-Instruct";
const QWEN_EOS: i64 = 151645;

fn worker_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..").join("..").join("worker")
        .canonicalize().expect("worker dir")
}
fn python_bin() -> PathBuf { worker_dir().join(".venv").join("bin").join("python") }

struct WorkerProc { child: Child }
impl WorkerProc {
    fn spawn(port: u16) -> Self {
        let child = Command::new(python_bin())
            .arg("-m").arg("diffuse_worker")
            .env("DIFFUSE_WORKER_PORT", port.to_string())
            .current_dir(worker_dir()).spawn().expect("spawn");
        WorkerProc { child }
    }
}
impl Drop for WorkerProc {
    fn drop(&mut self) { let _ = self.child.kill(); let _ = self.child.wait(); }
}

async fn wait_up(ep: &str) -> WorkerHandle {
    for _ in 0..40 {
        if let Ok(w) = WorkerHandle::connect(ep.to_string()).await { return w; }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    panic!("worker never up");
}

#[tokio::test]
async fn client_generates_with_network_blind_to_prompt_and_output() {
    let _cw = WorkerProc::spawn(50221);
    let _hw = WorkerProc::spawn(50222);

    let mut client_w = wait_up("http://127.0.0.1:50221").await;
    let mut host_w = wait_up("http://127.0.0.1:50222").await;

    let full = client_w.load_slice(MODEL, 0, 2, "").await.expect("probe");
    client_w.load_slice(MODEL, 0, 2, "").await.expect("client slice");
    host_w.load_slice(MODEL, 2, full, "").await.expect("host slice");

    // Prompt encoded locally on the client. Tokens never leave.
    let (ids, _eos) = client_w.encode("Name a color.", true).await.expect("encode");

    // Host exposes encrypted compute over its local worker.
    let host_kx = Arc::new(KeyExchange::generate());
    let host_pub = host_kx.public_bytes();
    let addr: std::net::SocketAddr = "127.0.0.1:50322".parse().unwrap();
    let _srv = spawn_compute_server(addr, Arc::clone(&host_kx), Arc::new(Mutex::new(host_w)), MODEL.to_string());
    tokio::time::sleep(Duration::from_millis(300)).await;

    let mut session = ClientSession {
        local_worker: client_w,
        client_kx: KeyExchange::generate(),
        model_id: MODEL.to_string(),
        local_start: 0,
        local_end: 2,
        remote: RemoteStage {
            daemon_endpoint: "http://127.0.0.1:50322".to_string(),
            host_kx_public: host_pub,
            start_layer: 2,
            end_layer: full,
        },
        remote_client: None,
    };

    let out = session.generate(&ids, 10, "priv-session", Some(QWEN_EOS)).await.expect("gen");
    assert!(out.len() > ids.len(), "should generate new tokens");

    // Decoding happens locally on the client: the output text also never touches the network.
    let text = session.local_worker.decode(&out[ids.len()..], true).await.expect("decode");
    assert!(!text.is_empty(), "client decodes the answer locally");
    println!("\n=== privacy-max generation ===\nanswer: {}\n", text);
}