use std::sync::Arc;
use std::time::Duration;

use owo_colors::OwoColorize;

use crate::capacity::{analyze, assign_slice, ModelCapacity};
use crate::discovery::bootstrap;
use crate::identity::Identity;
use crate::registry::PeerRegistry;
use crate::worker::pb::Tensor;
use crate::worker::WorkerHandle;
use tokio::sync::Mutex;
use crate::gossip::spawn_gossip_server;
use crate::compute::spawn_compute_server;
use crate::discovery::spawn_gossip_loop;
use crate::registry::{now_ms, Peer};
use diffuse_trust::crypto::sign;


pub(crate) fn incomplete_model_message(cap: &ModelCapacity) -> String {
    let gaps = cap.coverage_gaps();
    if gaps.is_empty() {
        return format!("model {} is present but not fully servable", cap.model_id);
    }
    let missing = gaps
        .iter()
        .map(|(start, end)| format!("{}:{}", start, end))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "model {} is incomplete: no peer serves layers {} (of {} total). \
         Waiting for those slices to come online, or host one yourself with `diffuse join`.",
        cap.model_id, missing, cap.total_layers
    )
}

fn human_bytes(b: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut value = b as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", b, UNITS[unit])
    } else {
        format!("{:.1} {}", value, UNITS[unit])
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_generative(
    orch: &mut crate::orchestrator::Orchestrator,
    worker: &mut WorkerHandle,
    session_id: &str,
    first: crate::worker::pb::Tensor,
    memory: Option<crate::worker::pb::Tensor>,
    streams: u32,
    kind: &str,
    attachments: &[Attachment],
    shortlist: u32,
    draw: crate::worker::Draw,
) -> anyhow::Result<String> {
    for item in attachments {
        println!(
            "  {} {} {} stays on this machine, only activations leave",
            "◆".bright_green(),
            item.kind.bright_white(),
            item.label.dimmed()
        );
    }
    println!(
        "  {} this model answers with {} on {} stream{}",
        "→".bright_blue(),
        kind.bright_white(),
        streams.to_string().bright_white(),
        if streams > 1 { "s" } else { "" }
    );
    if memory.is_some() {
        println!(
            "  {} the prompt was read here; its encoding is what travels",
            "→".bright_blue()
        );
    }
    if draw.guidance > 1.0 {
        println!(
            "  {} guided by a factor of {}, so every step runs twice",
            "→".bright_blue(),
            format!("{:.1}", draw.guidance).bright_white()
        );
    }
    if draw.sample {
        println!(
            "  {} sampled on the last slice, where the whole distribution is",
            "→".bright_blue()
        );
    }
    println!("  {} generating over encrypted channel...", "→".bright_blue());
    println!();

    let mut step = first;
    let mut produced = 0usize;
    loop {
        let out = orch
            .forward_streams(step, memory.clone(), session_id, shortlist, draw)
            .await?;
        match worker.advance_generation(session_id, out).await? {
            Some(next) => {
                step = next;
                produced += 1;
            }
            None => break,
        }
    }
    orch.clear_session(session_id).await;

    let (data, mime, text) = worker.finish_generation(session_id).await?;
    if !text.is_empty() {
        println!("  {}", "answer:".bright_green().bold());
        println!("  {}", text);
        println!();
        return Ok(text);
    }

    let extension = mime.rsplit('/').next().unwrap_or("bin");
    let path = std::env::current_dir()
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .join(format!("diffuse-{}.{}", &session_id[..8.min(session_id.len())], extension));
    std::fs::write(&path, &data)?;
    println!("  {}", "answer:".bright_green().bold());
    println!(
        "  {} of {} after {} steps",
        human_bytes(data.len() as u64).bright_white(),
        mime.bright_white(),
        produced.to_string().bright_white()
    );
    println!("  {}", path.display().to_string().bright_green().bold());
    println!();
    Ok(format!(
        "{} of {} written to {}",
        human_bytes(data.len() as u64),
        mime,
        path.display()
    ))
}

async fn run_diffusion(
    orch: &mut crate::orchestrator::Orchestrator,
    worker: &mut WorkerHandle,
    session_id: &str,
    started: crate::worker::DiffusionStart,
    patches: usize,
) -> anyhow::Result<String> {
    let crate::worker::DiffusionStart {
        mut hidden,
        mut patch,
        mut sequence,
        blocks,
        kind,
        max_patches,
    } = started;
    let mut allowed = if max_patches == 0 { patches } else { patches.min(max_patches) };

    println!(
        "  {} this model answers with {} by diffusion",
        "→".bright_blue(),
        kind.bright_white()
    );
    println!(
        "  {} {} blocks, cut into {} patch{} per step",
        "→".bright_blue(),
        blocks.to_string().bright_white(),
        allowed.to_string().bright_white(),
        if allowed > 1 { "es" } else { "" }
    );
    if allowed < patches {
        println!(
            "  {} this model carries two streams through its blocks, so a step stays whole",
            "→".bright_blue()
        );
    }
    println!("  {} generating over encrypted channel...", "→".bright_blue());
    println!();

    let started = std::time::Instant::now();
    let mut calls = 0usize;
    let bar = start_spinner("denoising");
    loop {
        let count = if calls == 0 { 1 } else { allowed };
        let size = (sequence as usize).div_ceil(count.max(1));
        let mut pieces: Vec<Tensor> = Vec::with_capacity(count);
        let mut carried = std::mem::take(&mut patch.arguments);
        let layout = std::mem::take(&mut patch.layout);
        for index in 0..count {
            let start = index * size;
            let stop = ((index + 1) * size).min(sequence as usize);
            if start >= stop {
                continue;
            }
            let mut step = patch.clone();
            step.offset = start as u64;
            step.arguments = std::mem::take(&mut carried);
            step.layout = if index == 0 { layout.clone() } else { String::new() };
            let slice = slice_rows(&hidden, start, stop)?;
            pieces.push(orch.denoise_patch(slice, step, session_id).await?);
        }
        let joined = join_rows(&pieces, &hidden)?;
        calls += 1;
        match worker.advance_diffusion(session_id, joined).await? {
            Some((next, next_patch, ceiling)) => {
                hidden = next;
                sequence = next_patch.sequence;
                patch = next_patch;
                if ceiling > 0 {
                    allowed = allowed.min(ceiling);
                }
            }
            None => break,
        }
    }
    bar.finish_and_clear();
    orch.clear_session(session_id).await;

    let (data, mime) = worker.finish_diffusion(session_id).await?;
    let extension = mime.rsplit('/').next().unwrap_or("bin");
    let path = std::env::current_dir()
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .join(format!("diffuse-{}.{}", &session_id[..8.min(session_id.len())], extension));
    std::fs::write(&path, &data)?;
    println!("  {}", "answer:".bright_green().bold());
    println!(
        "  {} of {} after {} transformer passes in {:.1}s",
        human_bytes(data.len() as u64).bright_white(),
        mime.bright_white(),
        calls.to_string().bright_white(),
        started.elapsed().as_secs_f64()
    );
    println!("  {}", path.display().to_string().bright_green().bold());
    println!();
    Ok(format!(
        "{} of {} written to {}",
        human_bytes(data.len() as u64),
        mime,
        path.display()
    ))
}

fn slice_rows(tensor: &Tensor, start: usize, stop: usize) -> anyhow::Result<Tensor> {
    if tensor.shape.len() != 3 {
        anyhow::bail!("expected [batch, sequence, width] hidden states");
    }
    let width = tensor.shape[2] as usize;
    let unit = element_size(&tensor.dtype)?;
    let row = width * unit;
    Ok(Tensor {
        shape: vec![tensor.shape[0], (stop - start) as i64, tensor.shape[2]],
        dtype: tensor.dtype.clone(),
        data: tensor.data[start * row..stop * row].to_vec(),
    })
}

fn join_rows(pieces: &[Tensor], like: &Tensor) -> anyhow::Result<Tensor> {
    let mut data = Vec::with_capacity(like.data.len());
    let mut rows = 0i64;
    for piece in pieces {
        data.extend_from_slice(&piece.data);
        rows += piece.shape[1];
    }
    Ok(Tensor {
        shape: vec![like.shape[0], rows, like.shape[2]],
        dtype: pieces
            .first()
            .map(|p| p.dtype.clone())
            .unwrap_or_else(|| like.dtype.clone()),
        data,
    })
}

fn element_size(dtype: &str) -> anyhow::Result<usize> {
    match dtype {
        "float32" | "int32" => Ok(4),
        "float16" | "bfloat16" => Ok(2),
        "int64" | "float64" => Ok(8),
        other => anyhow::bail!("unsupported dtype {}", other),
    }
}

pub async fn plan(
    model: &str,
    worker_endpoint: &str,
    overhead: f64,
    bootstrap_sentinels: &[String],
    identity: &Identity,
) -> anyhow::Result<()> {
    crate::tui::header(crate::tui::sym("🔎", "?"), "capacity plan");
    println!("  node {}", identity.short_id().dimmed());
    println!();

    let mut worker = WorkerHandle::connect(worker_endpoint.to_string()).await?;

    println!("  {} profiling this machine...", "→".bright_blue());
    let profile = worker.profile_model(model, overhead).await?;

    println!(
        "    device: {}   available: {}",
        profile.device.bright_white(),
        human_bytes(profile.available_bytes).bright_white()
    );
    println!(
        "    model {}: {} layers, ~{} per layer",
        model.bright_white(),
        profile.total_layers.to_string().bright_white(),
        human_bytes(profile.avg_layer_bytes).bright_white()
    );
    println!(
        "    this machine can hold up to {} layers",
        profile.max_layers.to_string().bright_green().bold()
    );
    println!();

    let registry = Arc::new(Mutex::new(PeerRegistry::new(60_000)));
    if !bootstrap_sentinels.is_empty() {
        println!("  {} contacting sentinels for network view...", "→".bright_blue());
        bootstrap(bootstrap_sentinels, identity.signing_public().to_vec(), &registry).await;
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    let caps = {
        let reg = registry.lock().await;
        analyze(&reg)
    };

    let assignment = assign_slice(
        &caps,
        model,
        profile.total_layers,
        profile.max_layers,
        2,
    );

    match assignment {
        Some(a) => {
            println!("  {} recommended assignment:", "◆".bright_green().bold());
            println!(
                "    hold slice {}   ({})",
                format!("{}:{}", a.start, a.end).bright_green().bold(),
                a.reason.dimmed()
            );
        }
        None => {
            println!(
                "  {} this machine cannot hold any slice of {} (too large)",
                "✗".red().bold(),
                model
            );
        }
    }
    println!();
    Ok(())
}

pub async fn demo(
    stage_a: Vec<String>,
    stage_b: Vec<String>,
    model: String,
    prompt: Option<String>,
    spares: Vec<String>,
    identity: Identity,
) -> anyhow::Result<()> {
    crate::lifecycle::run_demo(stage_a, stage_b, model, prompt, spares, identity).await
}

pub async fn host(
    model: Option<&str>,
    worker_endpoint: &str,
    listen: &str,
    bootstrap_sentinels: &[String],
    overhead: f64,
    spawn_worker: bool,
    public_addr: Option<String>,
    identity: Identity,
) -> anyhow::Result<()> {
    crate::tui::header(crate::tui::sym("📡", "^"), "joining the network");
    println!("  node {}", identity.short_id().dimmed());

    let _worker_child = if spawn_worker {
        let port: u16 = worker_endpoint
            .rsplit(':')
            .next()
            .and_then(|p| p.parse().ok())
            .unwrap_or(50051);
        println!("  {} starting local worker on port {}...", "→".bright_blue(), port);
        Some(spawn_local_worker(port)?)
    } else {
        None
    };

    let mut worker = if spawn_worker {
        match connect_worker(worker_endpoint, Duration::from_secs(30)).await {
            Ok(w) => w,
            Err(e) => anyhow::bail!(
                "the local worker did not become ready within 30 seconds. It imports torch and transformers before opening its gRPC port, which can take longer on a loaded machine or a cold disk cache. This is a local worker startup problem, not a network issue. Underlying error: {}",
                e
            ),
        }
    } else {
        connect_worker(worker_endpoint, Duration::from_secs(30)).await?
    };

    let registry = Arc::new(Mutex::new(PeerRegistry::new(60_000)));
    let sentinels = crate::config::resolve_sentinels(bootstrap_sentinels);
    if !sentinels.is_empty() {
        crate::discovery::bootstrap(&sentinels, identity.signing_public().to_vec(), &registry).await;
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    let caps = { analyze(&*registry.lock().await) };

    let model = match model {
        Some(m) => m.to_string(),
        None => loop {
            let Some(picked) = crate::marketplace::browse(&mut worker, &caps).await? else {
                println!();
                crate::tui::note("no model selected, nothing to host");
                return Ok(());
            };
            if crate::marketplace::confirm_selection(&mut worker, &picked, overhead, &caps).await? {
                break picked;
            }
        },
    };
    let model = model.as_str();

    let profile = worker.profile_model(model, overhead).await?;
    println!(
        "  {} machine holds up to {} layers of {}",
        "→".bright_blue(),
        profile.max_layers.to_string().bright_green().bold(),
        model.bright_white()
    );
    let assignment = assign_slice(&caps, model, profile.total_layers, profile.max_layers, 2)
        .ok_or_else(|| anyhow::anyhow!("machine too small to hold any slice of {}", model))?;

    println!(
        "  {} taking slice {}   ({})",
        "◆".bright_green().bold(),
        format!("{}:{}", assignment.start, assignment.end)
            .bright_green()
            .bold(),
        assignment.reason.dimmed()
    );

    worker
        .load_slice(model, assignment.start, assignment.end, "")
        .await?;
    println!("  {} slice loaded and ready", "✓".bright_green());

    let self_node_id = identity.signing_public().to_vec();
    let self_kx_public = identity.kx_public().to_vec();
    let signing_key = identity.signing_key.clone();
    let shared_worker = Arc::new(Mutex::new(worker));
    let identity_kx = Arc::new(identity.key_exchange);
    let addr: std::net::SocketAddr = listen.parse()?;
    let compute_port = addr.port() + 1000;

    let relay_state = crate::relay::RelayState::with_registry(Arc::clone(&registry));
    spawn_gossip_server(addr, Arc::clone(&registry), relay_state.clone());
    crate::gossip::spawn_prune_loop(Arc::clone(&registry), std::time::Duration::from_secs(30));

    let compute_addr: std::net::SocketAddr =
        format!("{}:{}", addr.ip(), compute_port).parse()?;
    spawn_compute_server(
        compute_addr,
        Arc::clone(&identity_kx),
        Arc::clone(&shared_worker),
        model.to_string(),
    );

    let announce_addr = public_addr.as_deref().unwrap_or(listen);
    let announce_ip = announce_addr.split(':').next().unwrap_or("127.0.0.1");
    let probe_endpoint = format!("http://{}:{}", announce_ip, compute_port);
    let probe_daemon_endpoint = format!("http://{}", announce_addr);

    tokio::time::sleep(std::time::Duration::from_millis(300)).await;

    let (self_reachable, observed_ip) = {
        let probes: Vec<String> = sentinels
            .iter()
            .filter(|s| s.as_str() != probe_daemon_endpoint)
            .cloned()
            .collect();
        if probes.is_empty() {
            tracing::info!("no external sentinel to probe, assuming reachable");
            (true, None)
        } else {
            let mut reachable = false;
            let mut observed: Option<String> = None;
            for s in &probes {
                let (ok, seen) = crate::gossip::check_my_reachability(s, &probe_endpoint).await;
                if observed.is_none() {
                    observed = seen;
                }
                if ok {
                    reachable = true;
                    break;
                }
            }
            (reachable, observed)
        }
    };

    let advertised_ip = match (public_addr.as_deref(), observed_ip.as_deref()) {
        (Some(explicit), _) => explicit.split(':').next().unwrap_or(announce_ip).to_string(),
        (None, Some(seen)) => seen.to_string(),
        (None, None) => announce_ip.to_string(),
    };

    let daemon_endpoint = format!("http://{}:{}", advertised_ip, addr.port());
    let compute_endpoint = format!("http://{}:{}", advertised_ip, compute_port);

    if self_reachable {
        println!(
            "  {} reachable at {} (serving directly)",
            "✓".bright_green(),
            compute_endpoint.bright_white()
        );
    } else {
        println!(
            "  {} behind NAT, seen from outside as {}",
            "⚠".bright_yellow(),
            advertised_ip.bright_white()
        );
        if let Some(sentinel) = sentinels
            .iter()
            .find(|s| s.as_str() != daemon_endpoint)
            .cloned()
        {
            crate::relay::spawn_relay_client(
                sentinel,
                self_node_id.clone(),
                Arc::clone(&identity_kx),
                Arc::clone(&shared_worker),
            );
            println!("  {} relaying compute through sentinel", "✓".bright_green());
        }
    }

    let mut self_peer = Peer {
        node_id: self_node_id.clone(),
        daemon_endpoint: daemon_endpoint.clone(),
        worker_endpoint: worker_endpoint.to_string(),
        model_id: model.to_string(),
        start_layer: assignment.start,
        end_layer: assignment.end,
        total_layers: profile.total_layers,
        last_seen_ms: now_ms(),
        signature: Vec::new(),
        kx_public: self_kx_public.clone(),
        reachable: self_reachable,
    };
    self_peer.signature = sign(&signing_key, &self_peer.signable_bytes());
    registry.lock().await.upsert(self_peer);

    println!(
        "  {} serving — gossip {} / compute {}",
        "✓".bright_green(),
        daemon_endpoint.bright_white(),
        compute_endpoint.bright_white()
    );

    spawn_gossip_loop(
        self_node_id.clone(),
        daemon_endpoint.clone(),
        Arc::clone(&registry),
        Duration::from_secs(5),
        3,
    );

    println!();
    println!("  {} node is live. Press Ctrl+C to leave.", "●".bright_green());
    println!();

    let mut ticker = tokio::time::interval(Duration::from_secs(10));
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                println!();
                println!("  {} leaving the network", "◆".bright_yellow());
                break;
            }
            _ = ticker.tick() => {
                {
                    let mut reg = registry.lock().await;
                    let mut fresh = Peer {
                        node_id: self_node_id.clone(),
                        daemon_endpoint: daemon_endpoint.clone(),
                        worker_endpoint: worker_endpoint.to_string(),
                        model_id: model.to_string(),
                        start_layer: assignment.start,
                        end_layer: assignment.end,
                        total_layers: profile.total_layers,
                        last_seen_ms: now_ms(),
                        signature: Vec::new(),
                        kx_public: self_kx_public.clone(),
                        reachable: self_reachable,
                    };
                    fresh.signature = sign(&signing_key, &fresh.signable_bytes());
                    reg.upsert(fresh);
                }
                let caps = { analyze(&*registry.lock().await) };
                let n = registry.lock().await.len();
                crate::display::render_network_state(&caps, n, 2);
            }
        }
    }

    Ok(())
}

/// Where a virtual environment keeps its interpreter, which is not the same
/// place on every system: Windows puts it in `Scripts` and gives it a suffix.
pub(crate) fn venv_python(worker_dir: &std::path::Path) -> std::path::PathBuf {
    if cfg!(windows) {
        worker_dir.join(".venv").join("Scripts").join("python.exe")
    } else {
        worker_dir.join(".venv").join("bin").join("python")
    }
}

fn home_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(std::path::PathBuf::from)
}

fn find_worker_dir() -> anyhow::Result<String> {
    if let Ok(dir) = std::env::var("DIFFUSE_WORKER_DIR") {
        return Ok(dir);
    }
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Some(home) = home_dir() {
        candidates.push(home.join(".diffuse").join("worker"));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("worker"));
        }
    }
    candidates.push(std::path::PathBuf::from("worker"));
    for c in &candidates {
        if venv_python(c).exists() {
            return Ok(c.to_string_lossy().into_owned());
        }
    }
    anyhow::bail!(
        "could not find the Diffuse worker. Install it (see install.sh) or set DIFFUSE_WORKER_DIR"
    )
}

pub struct WorkerGuard {
    child: std::process::Child,
}

impl WorkerGuard {
    pub fn new(child: std::process::Child) -> Self {
        Self { child }
    }
}

impl Drop for WorkerGuard {
    fn drop(&mut self) {
        let pid = self.child.id();
        tracing::info!("stopping local worker (pid {})", pid);
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub(crate) fn spawn_local_worker(port: u16) -> anyhow::Result<WorkerGuard> {
    let worker_dir = find_worker_dir()?;
    let python = venv_python(std::path::Path::new(&worker_dir));
    let child = std::process::Command::new(python)
        .arg("-m")
        .arg("diffuse_worker")
        .env("DIFFUSE_WORKER_PORT", port.to_string())
        .current_dir(&worker_dir)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()?;
    Ok(WorkerGuard::new(child))
}

pub(crate) async fn connect_worker(endpoint: &str, timeout: Duration) -> anyhow::Result<WorkerHandle> {
    if !endpoint.starts_with("http://") && !endpoint.starts_with("https://") {
        anyhow::bail!(
            "the worker address {} has no scheme; write it as http://{}",
            endpoint,
            endpoint
        );
    }
    let start = std::time::Instant::now();
    loop {
        match WorkerHandle::connect(endpoint.to_string()).await {
            Ok(worker) => return Ok(worker),
            Err(e) => {
                if start.elapsed() >= timeout {
                    anyhow::bail!(
                        "could not reach the inference worker at {} after {} seconds (last error: {})",
                        endpoint,
                        timeout.as_secs(),
                        e
                    );
                }
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
        }
    }
}

pub(crate) async fn start_local_tokenizer(port: u16) -> anyhow::Result<(WorkerGuard, WorkerHandle)> {
    let guard = spawn_local_worker(port)?;
    let endpoint = format!("http://127.0.0.1:{}", port);
    match connect_worker(&endpoint, Duration::from_secs(30)).await {
        Ok(worker) => Ok((guard, worker)),
        Err(e) => anyhow::bail!(
            "the local tokenizer worker on port {} did not become ready within 30 seconds. It imports torch and transformers before opening its gRPC port, which can take longer on a loaded machine or a cold disk cache. This is a local worker startup problem, not a network issue. Underlying error: {}",
            port,
            e
        ),
    }
}

#[derive(Debug)]
pub struct Attachment {
    pub kind: String,
    pub data: Vec<u8>,
    pub mime: String,
    pub label: String,
}

fn kind_from_extension(path: &std::path::Path) -> (&'static str, &'static str) {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "png" => ("image", "image/png"),
        "jpg" | "jpeg" => ("image", "image/jpeg"),
        "gif" => ("image", "image/gif"),
        "webp" => ("image", "image/webp"),
        "bmp" => ("image", "image/bmp"),
        "tif" | "tiff" => ("image", "image/tiff"),
        "wav" => ("audio", "audio/wav"),
        "mp3" => ("audio", "audio/mpeg"),
        "flac" => ("audio", "audio/flac"),
        "ogg" | "oga" => ("audio", "audio/ogg"),
        "m4a" | "aac" => ("audio", "audio/aac"),
        "mp4" => ("video", "video/mp4"),
        "mov" => ("video", "video/quicktime"),
        "mkv" => ("video", "video/x-matroska"),
        "webm" => ("video", "video/webm"),
        "avi" => ("video", "video/x-msvideo"),
        _ => ("", ""),
    }
}

fn read_attachment(path: &str, forced_kind: Option<&str>) -> anyhow::Result<Attachment> {
    let p = std::path::Path::new(path);
    let (guessed, mime) = kind_from_extension(p);
    let kind = match forced_kind {
        Some(k) => k,
        None if !guessed.is_empty() => guessed,
        None => anyhow::bail!(
            "cannot tell what kind of media {} is; pass it with --image, --audio or --video",
            path
        ),
    };
    let data = std::fs::read(p)
        .map_err(|e| anyhow::anyhow!("cannot read attachment {}: {}", path, e))?;
    if data.is_empty() {
        anyhow::bail!("attachment {} is empty", path);
    }
    Ok(Attachment {
        kind: kind.to_string(),
        mime: if mime.is_empty() {
            "application/octet-stream".to_string()
        } else {
            mime.to_string()
        },
        label: p
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(path)
            .to_string(),
        data,
    })
}

pub fn collect_attachments(
    images: &[String],
    audio: &[String],
    video: &[String],
    media: &[String],
) -> anyhow::Result<Vec<Attachment>> {
    let mut out = Vec::new();
    for path in images {
        out.push(read_attachment(path, Some("image"))?);
    }
    for path in audio {
        out.push(read_attachment(path, Some("audio"))?);
    }
    for path in video {
        out.push(read_attachment(path, Some("video"))?);
    }
    for path in media {
        out.push(read_attachment(path, None)?);
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub async fn query(
    model: &str,
    prompt: &str,
    attachments: Vec<Attachment>,
    bootstrap_sentinels: &[String],
    max_tokens: usize,
    steps: usize,
    patches: usize,
    seed: u64,
    identity: Identity,
) -> anyhow::Result<()> {
    crate::tui::header(crate::tui::sym("🔮", ">"), "query");
    println!("  node {}", identity.short_id().dimmed());

    let sentinels = crate::config::resolve_sentinels(bootstrap_sentinels);

    let registry = Arc::new(Mutex::new(PeerRegistry::new(60_000)));

    println!("  {} discovering network...", "→".bright_blue());
    crate::discovery::bootstrap(
        &sentinels,
        identity.signing_public().to_vec(),
        &registry,
    )
    .await;

    spawn_gossip_loop(
        identity.signing_public().to_vec(),
        "http://127.0.0.1:0".to_string(),
        Arc::clone(&registry),
        Duration::from_secs(2),
        3,
    );
    tokio::time::sleep(Duration::from_secs(3)).await;

    let caps = { analyze(&*registry.lock().await) };
    let cap = caps
        .iter()
        .find(|c| c.model_id == model)
        .ok_or_else(|| anyhow::anyhow!("model {} not found on the network", model))?;

    if !cap.servable {
        anyhow::bail!("{}", incomplete_model_message(cap));
    }

    println!(
        "  {} {} is servable, starting local tokenizer...",
        "✓".bright_green(),
        model.bright_white()
    );

    let tok_port: u16 = 50099;
    let (_tok_child, mut tokenizer_worker) = start_local_tokenizer(tok_port).await?;
    tokenizer_worker.load_slice(model, 0, 0, "").await?;

    println!("  {} building encrypted route...", "→".bright_blue());
    let mut orch = {
        let reg = registry.lock().await;
        crate::orchestrator::build_from_registry(model, &reg, 2, Vec::new(), sentinels.first().cloned()).await?
    };

    let session_id = format!("query-{}", uuid::Uuid::new_v4());

    match tokenizer_worker
        .begin_diffusion(&session_id, model, prompt, steps as u32, 0, 0, 0, 0.0, seed)
        .await
    {
        Ok(started) => {
            run_diffusion(&mut orch, &mut tokenizer_worker, &session_id, started, patches).await?;
            return Ok(());
        }
        Err(e) => tracing::debug!("not a diffusion pipeline: {:#}", e),
    }

    let media: Vec<(String, Vec<u8>, String)> = attachments
        .iter()
        .map(|a| (a.kind.clone(), a.data.clone(), a.mime.clone()))
        .collect();
    let generative = tokenizer_worker
        .begin_generation(&session_id, prompt, media.clone(), max_tokens as u32, seed)
        .await;

    let generative = match generative {
        Ok(started) => Some(started),
        Err(e) => {
            tracing::warn!("could not start a generative session: {:#}", e);
            None
        }
    };
    if let Some(started) = generative.filter(|s| s.first.is_some()) {
        if started.kind != "text" || started.memory.is_some() {
            let draw = started.draw();
            run_generative(
                &mut orch,
                &mut tokenizer_worker,
                &session_id,
                started.first.clone().unwrap(),
                started.memory.clone(),
                started.streams,
                &started.kind,
                &attachments,
                started.shortlist,
                draw,
            )
            .await?;
            return Ok(());
        }
        let _ = tokenizer_worker.finish_generation(&session_id).await;
    }

    let new_ids = if attachments.is_empty() {
        let (ids, eos) = tokenizer_worker.encode(prompt, true).await?;
        println!("  {} generating over encrypted channel...", "→".bright_blue());
        println!();
        let out = orch.generate(&ids, max_tokens, &session_id, Some(eos)).await?;
        out[ids.len()..].to_vec()
    } else {
        for item in &attachments {
            println!(
                "  {} {} {} stays on this machine, only activations leave",
                "◆".bright_green(),
                item.kind.bright_white(),
                item.label.dimmed()
            );
        }
        let eos = tokenizer_worker.encode(prompt, true).await.map(|(_, e)| e)?;
        let media: Vec<(String, Vec<u8>, String)> = attachments
            .iter()
            .map(|a| (a.kind.clone(), a.data.clone(), a.mime.clone()))
            .collect();
        let (embeddings, token_count, positions) = tokenizer_worker
            .embed_media(prompt, media, true, true)
            .await?;
        if positions.is_some() {
            println!(
                "  {} carrying multi-axis positions for the media layout",
                "→".bright_blue()
            );
        }
        println!(
            "  {} embedded into {} hidden states",
            "→".bright_blue(),
            token_count.to_string().bright_white()
        );
        println!("  {} generating over encrypted channel...", "→".bright_blue());
        println!();
        orch.generate_from_embeddings(
            embeddings,
            positions,
            max_tokens,
            &session_id,
            Some(eos),
            |_| {},
        )
        .await?
    };

    orch.clear_session(&session_id).await;
    let text = tokenizer_worker.decode(&new_ids, true).await?;

    println!("  {}", "answer:".bright_green().bold());
    println!("  {}", text);
    println!();
    Ok(())
}

async fn discover_network(
    bootstrap_sentinels: &[String],
    identity: &Identity,
) -> anyhow::Result<Arc<Mutex<PeerRegistry>>> {
    let sentinels = crate::config::resolve_sentinels(bootstrap_sentinels);
    if sentinels.is_empty() {
        anyhow::bail!("no sentinels available to find the network");
    }
    let registry = Arc::new(Mutex::new(PeerRegistry::new(60_000)));
    crate::discovery::bootstrap(
        &sentinels,
        identity.signing_public().to_vec(),
        &registry,
    )
    .await;
    spawn_gossip_loop(
        identity.signing_public().to_vec(),
        "http://127.0.0.1:0".to_string(),
        Arc::clone(&registry),
        Duration::from_secs(2),
        3,
    );
    tokio::time::sleep(Duration::from_secs(3)).await;
    Ok(registry)
}

pub async fn models(bootstrap_sentinels: &[String], identity: Identity) -> anyhow::Result<()> {
    crate::tui::header(crate::tui::sym("🌐", "::"), "models on the network");
    println!();

    let registry = {
        let sp = crate::tui::spinner("scanning the mesh");
        let r = discover_network(bootstrap_sentinels, &identity).await;
        sp.finish_and_clear();
        r?
    };
    let caps = { analyze(&*registry.lock().await) };
    let n = registry.lock().await.len();
    crate::display::render_network_state(&caps, n, 2);
    Ok(())
}

pub async fn chat(
    bootstrap_sentinels: &[String],
    memory: bool,
    max_tokens: usize,
    identity: Identity,
) -> anyhow::Result<()> {
    use std::io::Write;

    print!("\x1b[2J\x1b[H");
    let _ = std::io::stdout().flush();

    let registry = {
        let spinner = start_spinner("connecting to the network");
        let r = discover_network(bootstrap_sentinels, &identity).await;
        spinner.finish_and_clear();
        r?
    };

    let caps = { analyze(&*registry.lock().await) };
    let servable: Vec<String> = caps
        .iter()
        .filter(|c| c.servable)
        .map(|c| c.model_id.clone())
        .collect();

    if servable.is_empty() {
        println!("  {}", "No servable models on the network right now.".truecolor(220, 120, 120));
        return Ok(());
    }

    let model = if servable.len() == 1 {
        servable[0].clone()
    } else {
        let labels: Vec<String> = servable
            .iter()
            .map(|m| {
                let cap = caps.iter().find(|c| &c.model_id == m);
                let layers = cap.map(|c| c.total_layers).unwrap_or(0);
                let status = cap
                    .map(|c| if c.is_robust(2) { "robust" } else { "fragile" })
                    .unwrap_or("fragile");
                format!("{}   ·   {} layers · {}", m, layers, status)
            })
            .collect();
        let picked = inquire::Select::new(
            &format!("{} choose a model", crate::tui::sym("🧩", ">")),
            labels.clone(),
        )
        .prompt()
        .map_err(|_| anyhow::anyhow!("no model selected"))?;
        let idx = labels.iter().position(|l| l == &picked).unwrap_or(0);
        servable[idx].clone()
    };

    render_banner(env!("DIFFUSE_VERSION"), &model);
    println!(
        "  {}   {}",
        format!("{} connected", crate::tui::lock()).truecolor(80, 220, 160),
        "type a message, or /help for commands".truecolor(120, 130, 150)
    );
    println!();

    let tok_port: u16 = 50099;
    let (_tok_child, mut tokenizer_worker) = {
        let spinner = start_spinner("warming up");
        let result = start_local_tokenizer(tok_port).await;
        spinner.finish_and_clear();
        result?
    };
    tokenizer_worker.load_slice(&model, 0, 0, "").await?;

    let sentinels = crate::config::resolve_sentinels(bootstrap_sentinels);
    let mut orch = {
        let reg = registry.lock().await;
        crate::orchestrator::build_from_registry(&model, &reg, 2, Vec::new(), sentinels.first().cloned()).await?
    };

    let mut history: Vec<(String, String)> = Vec::new();
    let mut transcript: Vec<(String, String)> = Vec::new();
    let mut pending: Vec<Attachment> = Vec::new();
    let mut total_tokens: usize = 0;
    let session_prefix = format!("{:016x}", now_ms());
    let mut turn = 0u64;

    loop {
        use std::io::Write as _;
        print!(
            "  {} {} ",
            crate::tui::human().truecolor(240, 200, 60),
            "›".truecolor(240, 200, 60).bold()
        );
        let _ = std::io::stdout().flush();
        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).unwrap_or(0) == 0 {
            break;
        }

        let msg = input.trim().to_string();
        if msg.is_empty() {
            continue;
        }
        if msg == "/quit" || msg == "/exit" {
            println!();
            println!("  {} {}", crate::tui::sym("👋", "*"), "goodbye.".truecolor(140, 140, 160));
            break;
        }
        if msg == "/reset" {
            history.clear();
            pending.clear();
            crate::tui::note("conversation memory cleared.");
            continue;
        }
        if let Some((command, rest)) = split_attach_command(&msg) {
            match rest {
                "" => crate::tui::error(&format!("{} needs a file path", command)),
                path => match read_attachment(path, forced_kind(command)) {
                    Ok(item) => {
                        crate::tui::ok(
                            &format!("attached {}", item.kind),
                            &format!("{} · {}", item.label, human_bytes(item.data.len() as u64)),
                        );
                        pending.push(item);
                    }
                    Err(e) => crate::tui::error(&format!("{}", e)),
                },
            }
            continue;
        }
        if msg == "/files" {
            print_pending(&pending);
            continue;
        }
        if msg == "/detach" {
            pending.clear();
            crate::tui::note("attachments cleared.");
            continue;
        }
        if msg == "/clear" {
            print!("\x1b[2J\x1b[H");
            let _ = std::io::stdout().flush();
            continue;
        }
        if msg == "/help" {
            print_chat_help();
            continue;
        }
        if msg == "/stats" {
            print_chat_stats(turn, total_tokens);
            continue;
        }
        if let Some(rest) = msg.strip_prefix("/save") {
            let path = rest.trim();
            let path = if path.is_empty() { "diffuse-chat.md" } else { path };
            match save_transcript(path, &model, &transcript) {
                Ok(_) => crate::tui::ok("saved", path),
                Err(e) => crate::tui::error(&format!("save failed: {}", e)),
            }
            continue;
        }

        turn += 1;
        let session = format!("{}-{}", session_prefix, turn);

        let media: Vec<(String, Vec<u8>, String)> = pending
            .iter()
            .map(|a| (a.kind.clone(), a.data.clone(), a.mime.clone()))
            .collect();

        match tokenizer_worker
            .begin_generation(&session, &msg, media.clone(), max_tokens as u32, 0)
            .await
        {
            Ok(started) => match started.first.clone() {
                Some(first)
                    if started.kind != "text" || started.memory.is_some() =>
                {
                    let draw = started.draw();
                    println!();
                    match run_generative(
                        &mut orch,
                        &mut tokenizer_worker,
                        &session,
                        first,
                        started.memory.clone(),
                        started.streams,
                        &started.kind,
                        &pending,
                        started.shortlist,
                        draw,
                    )
                    .await
                    {
                        Ok(summary) => {
                            transcript.push((msg.clone(), summary.clone()));
                            if memory {
                                history.push(("user".to_string(), msg.clone()));
                                history.push(("assistant".to_string(), summary));
                            }
                        }
                        Err(e) => crate::tui::error(&format!("generation failed: {}", e)),
                    }
                    pending.clear();
                    continue;
                }
                _ => {
                    let _ = tokenizer_worker.finish_generation(&session).await;
                }
            },
            Err(e) => tracing::warn!("could not start a generative session: {:#}", e),
        }

        let messages = if memory {
            let mut m = history.clone();
            m.push(("user".to_string(), msg.clone()));
            m
        } else {
            vec![("user".to_string(), msg.clone())]
        };

        println!();
        let seal = crate::tui::phase_spinner("sealing prompt");
        let (ids, eos) = match tokenizer_worker.encode_messages(messages).await {
            Ok(v) => v,
            Err(e) => {
                seal.finish_and_clear();
                println!("  {} {}", "encode error:".truecolor(220, 120, 120), e);
                continue;
            }
        };
        seal.finish_and_clear();
        crate::tui::phase_done("prompt sealed", "X25519 · ChaCha20");
        tokio::time::sleep(Duration::from_millis(140)).await;

        let embedded = if pending.is_empty() {
            None
        } else {
            for item in &pending {
                crate::tui::phase_done(
                    &format!("{} read locally", item.kind),
                    &format!("{} · only activations leave", item.label),
                );
            }
            match tokenizer_worker.embed_media(&msg, media, true, true).await {
                Ok((prefill, count, positions)) => {
                    crate::tui::phase_done(
                        "embedded",
                        &format!("{} hidden states", count),
                    );
                    Some((prefill, positions))
                }
                Err(e) => {
                    crate::tui::error(&format!("could not read the attachments: {}", e));
                    pending.clear();
                    continue;
                }
            }
        };
        let prompt_len = if embedded.is_some() { 0 } else { ids.len() };

        let hops = orch.stages.len();
        let route = crate::tui::phase_spinner("routing through the network");
        tokio::time::sleep(Duration::from_millis(240)).await;
        route.finish_and_clear();
        crate::tui::phase_done(
            "routed",
            &format!("{} encrypted hop{}", hops, if hops == 1 { "" } else { "s" }),
        );
        tokio::time::sleep(Duration::from_millis(120)).await;

        let gen_start = std::time::Instant::now();
        let think = crate::tui::phase_spinner("thinking");

        let (tok_tx, mut tok_rx) = tokio::sync::mpsc::unbounded_channel::<i64>();

        let mut interrupted = false;
        let out: anyhow::Result<Vec<i64>> = {
            let emit = move |tid: i64| {
                let _ = tok_tx.send(tid);
            };
            type Streamed<'a> =
                std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<Vec<i64>>> + 'a>>;
            let gen_fut: Streamed<'_> = match embedded {
                Some((prefill, positions)) => Box::pin(orch.generate_from_embeddings(
                    prefill,
                    positions,
                    max_tokens,
                    &session,
                    Some(eos),
                    emit,
                )),
                None => {
                    Box::pin(orch.generate_streaming(&ids, max_tokens, &session, Some(eos), emit))
                }
            };

            let mut printed = String::new();
            let mut collected: Vec<i64> = Vec::new();
            let mut token_count = 0usize;
            let mut live: Option<crate::tui::LiveMeter> = None;

            tokio::pin!(gen_fut);
            let result: anyhow::Result<Vec<i64>> = loop {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {
                        match &live {
                            Some(m) => m.clear(),
                            None => think.finish_and_clear(),
                        }
                        interrupted = true;
                        break Ok(collected.clone());
                    }
                    maybe_tid = tok_rx.recv() => {
                        if let Some(tid) = maybe_tid {
                            if tid == eos {
                                continue;
                            }
                            collected.push(tid);
                            if let Ok(text) = tokenizer_worker.decode(&collected, true).await {
                                if text.len() > printed.len() && text.starts_with(&printed) {
                                    let delta = text[printed.len()..].to_string();
                                    printed = text;
                                    token_count += 1;
                                    let rate =
                                        token_count as f64 / gen_start.elapsed().as_secs_f64().max(0.001);
                                    if live.is_none() {
                                        think.finish_and_clear();
                                        live = Some(start_answer());
                                    }
                                    let m = live.as_mut().unwrap();
                                    m.push(&delta);
                                    m.tick(&crate::tui::meter_text(rate, token_count));
                                }
                            }
                        }
                    }
                    res = &mut gen_fut => {
                        while let Ok(tid) = tok_rx.try_recv() {
                            if tid == eos {
                                continue;
                            }
                            collected.push(tid);
                        }
                        if let Ok(text) = tokenizer_worker.decode(&collected, true).await {
                            if text.len() > printed.len() && text.starts_with(&printed) {
                                let delta = text[printed.len()..].to_string();
                                printed = text;
                                if live.is_none() {
                                    think.finish_and_clear();
                                    live = Some(start_answer());
                                }
                                live.as_mut().unwrap().push(&delta);
                            }
                        }
                        match &live {
                            Some(m) => m.clear(),
                            None => think.finish_and_clear(),
                        }
                        break res;
                    }
                }
            };
            result
        };

        if interrupted {
            println!();
            println!("  {}", "[interrupted]".truecolor(150, 150, 160));
        }

        let out = match out {
            Ok(o) => o,
            Err(e) => {
                println!();
                println!("  {} {}", "network error:".truecolor(220, 120, 120), e);
                continue;
            }
        };

        let answer = if out.len() > prompt_len {
            tokenizer_worker
                .decode(&out[prompt_len..], true)
                .await
                .unwrap_or_default()
        } else {
            String::new()
        };
        println!();

        let new_tokens = out.len().saturating_sub(prompt_len);
        total_tokens += new_tokens;
        let secs = gen_start.elapsed().as_secs_f64().max(0.001);
        let tok_s = new_tokens as f64 / secs;
        let compute = orch.last_forward_compute_ms;
        let network = orch.last_forward_network_ms;
        let compute_pct = if compute + network > 0 {
            ((compute as f64 / (compute + network) as f64) * 100.0).round() as u32
        } else {
            50
        };
        crate::tui::hud(hops, tok_s, compute_pct);
        transcript.push((msg.clone(), answer.trim().to_string()));
        println!();
        pending.clear();

        if memory {
            history.push(("user".to_string(), msg));
            history.push(("assistant".to_string(), answer.trim().to_string()));
        }
    }

    Ok(())
}

fn start_answer() -> crate::tui::LiveMeter {
    use std::io::Write;
    crate::tui::role(crate::tui::robot(), "diffuse", crate::tui::ACCENT, "");
    let rule = crate::tui::sym("─", "-").repeat(46);
    println!(
        "  {}",
        rule.truecolor(crate::tui::FAINT.0, crate::tui::FAINT.1, crate::tui::FAINT.2)
    );
    println!();
    print!("  ");
    let _ = std::io::stdout().flush();
    crate::tui::LiveMeter::new()
}

fn split_attach_command(msg: &str) -> Option<(&str, &str)> {
    for command in ["/attach", "/image", "/audio", "/video"] {
        if msg == command {
            return Some((command, ""));
        }
        if let Some(rest) = msg.strip_prefix(command) {
            if rest.starts_with(char::is_whitespace) {
                return Some((command, rest.trim()));
            }
        }
    }
    None
}

fn forced_kind(command: &str) -> Option<&'static str> {
    match command {
        "/image" => Some("image"),
        "/audio" => Some("audio"),
        "/video" => Some("video"),
        _ => None,
    }
}

fn print_pending(pending: &[Attachment]) {
    println!();
    if pending.is_empty() {
        crate::tui::note("nothing attached; /attach <file> adds one.");
        println!();
        return;
    }
    crate::tui::section(crate::tui::sym("📎", "@"), "attached to the next message");
    for item in pending {
        crate::tui::ok(
            &item.kind,
            &format!("{} · {}", item.label, human_bytes(item.data.len() as u64)),
        );
    }
    println!();
}

fn print_chat_help() {
    println!();
    crate::tui::section(crate::tui::sym("⌘", "/"), "commands");
    let items = [
        ("/help", "show this list"),
        ("/attach <file>", "attach a file, kind guessed from its extension"),
        ("/image <file>", "attach a picture"),
        ("/audio <file>", "attach a sound"),
        ("/video <file>", "attach a clip"),
        ("/files", "list what is attached to the next message"),
        ("/detach", "drop the attachments"),
        ("/reset", "clear conversation memory"),
        ("/clear", "clear the screen"),
        ("/stats", "session stats"),
        ("/save [file]", "save the transcript as markdown"),
        ("/quit", "leave the chat"),
    ];
    for (cmd, desc) in items {
        println!(
            "  {}  {}",
            format!("{:<14}", cmd).truecolor(
                crate::tui::GOLD.0,
                crate::tui::GOLD.1,
                crate::tui::GOLD.2
            ),
            desc.truecolor(crate::tui::MUTED.0, crate::tui::MUTED.1, crate::tui::MUTED.2)
        );
    }
    println!();
}

fn print_chat_stats(turns: u64, total_tokens: usize) {
    println!();
    crate::tui::section(crate::tui::sym("📊", "#"), "session");
    crate::tui::ok("turns", &turns.to_string());
    crate::tui::ok("tokens generated", &total_tokens.to_string());
    println!();
}

fn save_transcript(path: &str, model: &str, transcript: &[(String, String)]) -> anyhow::Result<()> {
    use std::io::Write;
    let mut out = String::new();
    out.push_str(&format!("# Diffuse chat · {}\n\n", model));
    for (user, ai) in transcript {
        out.push_str(&format!("**You:** {}\n\n", user));
        out.push_str(&format!("**Diffuse:** {}\n\n", ai));
    }
    let mut f = std::fs::File::create(path)?;
    f.write_all(out.as_bytes())?;
    Ok(())
}

fn start_spinner(msg: &str) -> indicatif::ProgressBar {
    let pb = indicatif::ProgressBar::new_spinner();
    pb.set_style(
        indicatif::ProgressStyle::with_template("  {spinner:.cyan} {msg}")
            .unwrap()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    pb.set_message(msg.to_string());
    pb.enable_steady_tick(std::time::Duration::from_millis(80));
    pb
}

fn stream_print(text: &str) {
    use std::io::Write;
    for c in text.chars() {
        print!("{}", c.to_string().truecolor(245, 220, 130));
        let _ = std::io::stdout().flush();
        std::thread::sleep(std::time::Duration::from_millis(7));
    }
}

fn render_banner(version: &str, model: &str) {
    let cyan = (90u8, 210u8, 235u8);
    let cyan_dim = (60u8, 150u8, 175u8);
    let red = (235u8, 70u8, 90u8);
    let frame = (70u8, 80u8, 100u8);
    let soft = (150u8, 160u8, 180u8);
    let width: usize = 64;

    let logo = [
        "        ╭─────────────────╮        ",
        "     ╭──┤  ◆   ◆   ◆   ◆  ├──╮     ",
        "   ╭─┤  ╰─────────────────╯  ├─╮   ",
        "   │ │  ·  D I F F U S E  ·  │ │   ",
        "   ╰─┤  ╭─────────────────╮  ├─╯   ",
        "     ╰──┤  ◆   ◆   ◆   ◆  ├──╯     ",
        "        ╰─────────────────╯        ",
    ];

    let line = |s: &str, color: (u8, u8, u8), bold: bool| {
        let len = s.chars().count();
        let pad = width.saturating_sub(len);
        let left = pad / 2;
        let right = pad - left;
        print!("  {}", "│".truecolor(frame.0, frame.1, frame.2));
        print!("{}", " ".repeat(left));
        if bold {
            print!("{}", s.truecolor(color.0, color.1, color.2).bold());
        } else {
            print!("{}", s.truecolor(color.0, color.1, color.2));
        }
        print!("{}", " ".repeat(right));
        println!("{}", "│".truecolor(frame.0, frame.1, frame.2));
    };

    println!();
    println!(
        "  {}{}{}",
        "╭".truecolor(frame.0, frame.1, frame.2),
        "─".repeat(width).truecolor(frame.0, frame.1, frame.2),
        "╮".truecolor(frame.0, frame.1, frame.2)
    );
    line("", cyan, false);
    for (i, l) in logo.iter().enumerate() {
        if i == 3 {
            line(l, cyan, true);
        } else {
            line(l, cyan_dim, true);
        }
    }
    line("", cyan, false);
    line("DECENTRALIZED PRIVATE INFERENCE", red, true);
    line("", cyan, false);
    line("no servers · no surveillance · no logs", cyan, true);
    line("", cyan, false);
    line(&format!("version {}", version), soft, false);
    line(&format!("model   {}", model), soft, false);
    line("", cyan, false);
    println!(
        "  {}{}{}",
        "╰".truecolor(frame.0, frame.1, frame.2),
        "─".repeat(width).truecolor(frame.0, frame.1, frame.2),
        "╯".truecolor(frame.0, frame.1, frame.2)
    );
    println!();
}
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn connect_worker_reports_endpoint_and_cause_on_failure() {
        let endpoint = "http://127.0.0.1:59677";
        let err = match connect_worker(endpoint, Duration::from_secs(1)).await {
            Ok(_) => panic!("expected a connection failure"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(msg.contains("59677"), "message names the port: {}", msg);
        assert!(
            msg.contains("could not reach the inference worker"),
            "message names the worker as the cause: {}",
            msg
        );
    }

    #[test]
    fn worker_guard_kills_child_on_drop() {
        let child = std::process::Command::new("sleep")
            .arg("30")
            .spawn()
            .expect("spawn sleep");
        let pid = child.id();
        let guard = WorkerGuard::new(child);
        drop(guard);
        let alive = std::path::Path::new(&format!("/proc/{}", pid)).exists();
        assert!(!alive, "child {} should be gone after the guard drops", pid);
    }
}
