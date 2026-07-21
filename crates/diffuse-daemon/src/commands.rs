use std::sync::Arc;
use std::time::Duration;

use owo_colors::OwoColorize;

use crate::capacity::{analyze, assign_slice, ModelCapacity};
use crate::discovery::bootstrap;
use crate::identity::Identity;
use crate::registry::PeerRegistry;
use crate::worker::WorkerHandle;
use tokio::sync::Mutex;
use crate::gossip::spawn_gossip_server;
use crate::compute::spawn_compute_server;
use crate::discovery::spawn_gossip_loop;
use crate::registry::{now_ms, Peer};
use diffuse_trust::crypto::sign;


/// Explain why an incomplete model can't be served, naming the exact layer
/// ranges the network is missing so the user knows what to spin up rather than
/// letting a query build a route that dead-ends partway through the model.
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
    let gb = b as f64 / 1_073_741_824.0;
    if gb >= 1.0 {
        format!("{:.1} GB", gb)
    } else {
        format!("{:.0} MB", b as f64 / 1_048_576.0)
    }
}

pub async fn plan(
    model: &str,
    worker_endpoint: &str,
    overhead: f64,
    bootstrap_sentinels: &[String],
    identity: &Identity,
) -> anyhow::Result<()> {
    println!();
    println!("{}", "  ◆ DIFFUSE — capacity plan".bright_cyan().bold());
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
    model: &str,
    worker_endpoint: &str,
    listen: &str,
    bootstrap_sentinels: &[String],
    overhead: f64,
    spawn_worker: bool,
    public_addr: Option<String>,
    identity: Identity,
) -> anyhow::Result<()> {
    println!();
    println!("{}", "  ◆ DIFFUSE — joining network".bright_cyan().bold());
    println!("  node {}", identity.short_id().dimmed());

    let _worker_child = if spawn_worker {
        let port: u16 = worker_endpoint
            .rsplit(':')
            .next()
            .and_then(|p| p.parse().ok())
            .unwrap_or(50051);
        println!("  {} starting local worker on port {}...", "→".bright_blue(), port);
        let child = spawn_local_worker(port)?;
        tokio::time::sleep(Duration::from_secs(6)).await;
        Some(child)
    } else {
        None
    };

    let mut worker = {
        let mut attempts = 0;
        loop {
            match WorkerHandle::connect(worker_endpoint.to_string()).await {
                Ok(w) => break w,
                Err(e) => {
                    attempts += 1;
                    if attempts >= 30 {
                        return Err(anyhow::anyhow!("worker never came up: {}", e));
                    }
                    tokio::time::sleep(Duration::from_secs(1)).await;
                }
            }
        }
    };

    let profile = worker.profile_model(model, overhead).await?;
    println!(
        "  {} machine holds up to {} layers of {}",
        "→".bright_blue(),
        profile.max_layers.to_string().bright_green().bold(),
        model.bright_white()
    );

    let registry = Arc::new(Mutex::new(PeerRegistry::new(60_000)));
    let sentinels = crate::config::resolve_sentinels(bootstrap_sentinels);
    if !sentinels.is_empty() {
        crate::discovery::bootstrap(&sentinels, identity.signing_public().to_vec(), &registry).await;
        tokio::time::sleep(Duration::from_millis(500)).await;
    }

    let caps = { analyze(&*registry.lock().await) };
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

fn find_worker_dir() -> anyhow::Result<String> {
    if let Ok(dir) = std::env::var("DIFFUSE_WORKER_DIR") {
        return Ok(dir);
    }
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Some(home) = std::env::var_os("HOME") {
        candidates.push(std::path::PathBuf::from(&home).join(".diffuse/worker"));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("worker"));
        }
    }
    candidates.push(std::path::PathBuf::from("worker"));
    for c in &candidates {
        if c.join(".venv/bin/python").exists() {
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
    let python = format!("{}/.venv/bin/python", worker_dir);
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

pub async fn query(
    model: &str,
    prompt: &str,
    bootstrap_sentinels: &[String],
    max_tokens: usize,
    identity: Identity,
) -> anyhow::Result<()> {
    println!();
    println!("{}", "  ◆ DIFFUSE — query".bright_cyan().bold());
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

    // Local lightweight tokenizer worker (privacy: tokens never leave the client).
    let tok_port: u16 = 50099;
    let _tok_child = spawn_local_worker(tok_port)?;
    tokio::time::sleep(Duration::from_secs(4)).await;
    let tok_endpoint = format!("http://127.0.0.1:{}", tok_port);
    let mut tokenizer_worker = WorkerHandle::connect(tok_endpoint).await?;
    tokenizer_worker.load_slice(model, 0, 0, "").await?;

    println!("  {} building encrypted route...", "→".bright_blue());
    let mut orch = {
        let reg = registry.lock().await;
        crate::orchestrator::build_from_registry(model, &reg, 2, Vec::new(), sentinels.first().cloned()).await?
    };

    // Encode locally (tokens stay here).
    let (ids, eos) = tokenizer_worker.encode(prompt, true).await?;

    println!("  {} generating over encrypted channel...", "→".bright_blue());
    println!();

    let session_id = format!("query-{}", uuid::Uuid::new_v4());
    let out = orch.generate(&ids, max_tokens, &session_id, Some(eos)).await?;
    orch.clear_session(&session_id).await;
    tracing::info!("generated {} tokens total, {} new", out.len(), out.len() - ids.len());
    tracing::info!("new token ids: {:?}", &out[ids.len()..]);
    let text = tokenizer_worker.decode(&out[ids.len()..], true).await?;
    tracing::info!("decoded text length: {}", text.len());

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
    println!();
    println!("{}", "  ◆ DIFFUSE — models on the network".bright_cyan().bold());
    println!();

    let registry = discover_network(bootstrap_sentinels, &identity).await?;
    let caps = { analyze(&*registry.lock().await) };
    let n = registry.lock().await.len();
    crate::display::render_network_state(&caps, n, 2);
    Ok(())
}

pub async fn chat(bootstrap_sentinels: &[String], memory: bool, identity: Identity) -> anyhow::Result<()> {
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
        inquire::Select::new("Choose a model", servable.clone())
            .prompt()
            .map_err(|_| anyhow::anyhow!("no model selected"))?
    };

    render_banner(env!("DIFFUSE_VERSION"), &model);
    println!("  {}", "type your message, or /quit to leave".truecolor(120, 130, 150));
    println!();

    // Local tokenizer worker (tokens stay client-side).
    let tok_port: u16 = 50099;
    let _tok_child = {
        let spinner = start_spinner("warming up");
        let c = spawn_local_worker(tok_port);
        tokio::time::sleep(Duration::from_secs(4)).await;
        spinner.finish_and_clear();
        c?
    };
    let tok_endpoint = format!("http://127.0.0.1:{}", tok_port);
    let mut tokenizer_worker = WorkerHandle::connect(tok_endpoint).await?;
    tokenizer_worker.load_slice(&model, 0, 0, "").await?;

    let sentinels = crate::config::resolve_sentinels(bootstrap_sentinels);
    let mut orch = {
        let reg = registry.lock().await;
        crate::orchestrator::build_from_registry(&model, &reg, 2, Vec::new(), sentinels.first().cloned()).await?
    };

    let mut history: Vec<(String, String)> = Vec::new();
    let session_prefix = format!("{:016x}", now_ms());
    let mut turn = 0u64;
    
    loop {
        use std::io::Write as _;
        print!("{} ", "›".truecolor(240, 200, 60).bold());
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
            println!("  {}", "goodbye.".truecolor(140, 140, 160));
            break;
        }
        if msg == "/reset" {
            history.clear();
            println!("  {}", "conversation cleared.".truecolor(140, 140, 160));
            println!();
            continue;
        }

        turn += 1;
        let session = format!("{}-{}", session_prefix, turn);

        // Build the full conversation: past history + this new user message.
        // With memory: send full history. Without (default): each message is standalone.
        let messages = if memory {
            let mut m = history.clone();
            m.push(("user".to_string(), msg.clone()));
            m
        } else {
            vec![("user".to_string(), msg.clone())]
        };

        let (ids, eos) = match tokenizer_worker.encode_messages(messages).await {
            Ok(v) => v,
            Err(e) => {
                println!("  {} {}", "encode error:".truecolor(220, 120, 120), e);
                continue;
            }
        };

        println!();
        print!("  ");
        use std::io::Write as _;
        let _ = std::io::stdout().flush();

        let (tok_tx, mut tok_rx) = tokio::sync::mpsc::unbounded_channel::<i64>();

        let mut interrupted = false;
        let out: anyhow::Result<Vec<i64>> = {
            let gen_fut = orch.generate_streaming(
                &ids,
                4096,
                &session,
                Some(eos),
                move |tid| {
                    let _ = tok_tx.send(tid);
                },
            );

            let mut printed = String::new();
            let mut collected: Vec<i64> = Vec::new();

            tokio::pin!(gen_fut);
            let result: anyhow::Result<Vec<i64>> = loop {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {
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
                                    print!("{}", delta.truecolor(245, 220, 130));
                                    let _ = std::io::stdout().flush();
                                    printed = text;
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
                                print!("{}", delta.truecolor(245, 220, 130));
                                let _ = std::io::stdout().flush();
                            }
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

        let answer = if out.len() > ids.len() {
            tokenizer_worker
                .decode(&out[ids.len()..], true)
                .await
                .unwrap_or_default()
        } else {
            String::new()
        };
        println!();
        println!();
        println!();

        // Persist the exchange so the model "remembers" next turn.
        if memory {
            history.push(("user".to_string(), msg));
            history.push(("assistant".to_string(), answer.trim().to_string()));
        }
    }

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
        // Le titre central (ligne 3) en cyan vif, le reste de l'anneau en cyan doux
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