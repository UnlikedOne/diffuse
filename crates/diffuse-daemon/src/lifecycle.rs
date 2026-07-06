use std::sync::Arc;
use std::time::Duration;

use crate::control::ControlPlane;
use crate::identity::Identity;
use crate::orchestrator::{Orchestrator, Replica, Stage};
use crate::worker::WorkerHandle;
use diffuse_trust::transport::KeyExchange;

async fn build_stage(
    endpoints: &[String],
    model: &str,
    start: u32,
    end: u32,
) -> anyhow::Result<Stage> {
    let mut replicas = Vec::new();
    for ep in endpoints {
        let mut worker = WorkerHandle::connect(ep.clone()).await?;
        worker.load_slice(model, start, end, "").await?;
        tracing::info!("replica {} loaded slice {}:{}", ep, start, end);
        replicas.push(Replica::Local { worker, alive: true });
    }
    Ok(Stage {
        start_layer: start,
        end_layer: end,
        replicas,
    })
}

pub async fn run_demo(
    stage_a: Vec<String>,
    stage_b: Vec<String>,
    model: String,
    prompt: Option<String>,
    spares: Vec<String>,
    identity: Identity,
) -> anyhow::Result<()> {
    tracing::info!("node {} ready", identity.short_id());

    let mut probe = WorkerHandle::connect(stage_a[0].clone()).await?;
    let full = probe.load_slice(&model, 0, 12, "").await?;
    let mid = full / 2;
    tracing::info!("model {} has {} layers, split at {}", model, full, mid);

    let stage_a = build_stage(&stage_a, &model, 0, mid).await?;
    let stage_b = build_stage(&stage_b, &model, mid, full).await?;
    tracing::info!(
        "pipeline ready: stage A ({} replicas), stage B ({} replicas)",
        stage_a.replicas.len(),
        stage_b.replicas.len()
    );

    let orch = Orchestrator {
        model_id: model.clone(),
        stages: vec![stage_a, stage_b],
        spare_endpoints: spares,
        target_replication: 2,
        session_kx: KeyExchange::generate(),
    };

    let control = ControlPlane::new(orch);
    let health_handle = control.spawn_health_loop(Duration::from_secs(3));
    tracing::info!("autonomous health monitoring started (every 3s)");

    let prompt = prompt.unwrap_or_else(|| "Who are you?".to_string());

    let (ids, eos_token_id) = {
        let mut guard = control.orchestrator.lock().await;
        guard.encode_via_any(&prompt, true).await?
    };
    tracing::info!("encoded prompt into {} tokens (eos={})", ids.len(), eos_token_id);

    tracing::info!("generating (autonomous monitoring runs in background)...");
    let out_ids = {
        let mut guard = control.orchestrator.lock().await;
        guard
            .generate(&ids, 80, "cli-session", Some(eos_token_id))
            .await?
    };

    let text = {
        let mut guard = control.orchestrator.lock().await;
        guard.decode_via_any(&out_ids[ids.len()..], true).await?
    };

    health_handle.abort();
    let _ = Arc::strong_count(&control.orchestrator);

    println!("\n=== Diffuse ===\nprompt: {}\nanswer: {}\n", prompt, text);
    Ok(())
}