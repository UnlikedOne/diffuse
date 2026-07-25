use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use owo_colors::OwoColorize;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::{mpsc, Mutex};
use tokio_stream::wrappers::ReceiverStream;

use crate::capacity::{analyze, ModelCapacity};
use crate::commands::{incomplete_model_message, start_local_tokenizer};
use crate::identity::Identity;
use crate::orchestrator::build_from_registry;
use crate::registry::PeerRegistry;
use crate::worker::WorkerHandle;

const TARGET_REPLICATION: usize = 2;
const DEFAULT_MAX_TOKENS: usize = 512;
const TOKENIZER_PORT: u16 = 50099;

struct TokenizerState {
    worker: WorkerHandle,
    loaded_model: Option<String>,
}

#[derive(Clone)]
struct AppState {
    registry: Arc<Mutex<PeerRegistry>>,
    tokenizer: Arc<Mutex<TokenizerState>>,
    default_model: Option<String>,
    sentinels: Vec<String>,
}

#[derive(Deserialize)]
struct ChatMessageIn {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletionRequest {
    model: Option<String>,
    messages: Vec<ChatMessageIn>,
    max_tokens: Option<usize>,
    stream: Option<bool>,
}

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
    error_type: &'static str,
    code: &'static str,
}

impl ApiError {
    fn into_response(self) -> Response {
        let body = json!({
            "error": {
                "message": self.message,
                "type": self.error_type,
                "code": self.code,
            }
        });
        (self.status, Json(body)).into_response()
    }
}

fn model_list_data(caps: &[ModelCapacity], created: u64) -> Vec<Value> {
    caps.iter()
        .filter(|c| c.servable)
        .map(|c| {
            json!({
                "id": c.model_id,
                "object": "model",
                "created": created,
                "owned_by": "diffuse",
            })
        })
        .collect()
}

fn resolve_servable_model(
    caps: &[ModelCapacity],
    requested: Option<String>,
    default: Option<String>,
) -> Result<ModelCapacity, ApiError> {
    let model = match requested.or(default) {
        Some(m) => m,
        None => {
            return Err(ApiError {
                status: StatusCode::BAD_REQUEST,
                message: "no model specified and no default model configured".to_string(),
                error_type: "invalid_request_error",
                code: "model_required",
            })
        }
    };

    let cap = match caps.iter().find(|c| c.model_id == model) {
        Some(c) => c.clone(),
        None => {
            return Err(ApiError {
                status: StatusCode::NOT_FOUND,
                message: format!("model {} is not present on the network", model),
                error_type: "invalid_request_error",
                code: "model_not_found",
            })
        }
    };

    if !cap.servable {
        return Err(ApiError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: incomplete_model_message(&cap),
            error_type: "service_unavailable",
            code: "model_incomplete",
        });
    }

    Ok(cap)
}

fn chunk_json(id: &str, created: u64, model: &str, delta: Value, finish_reason: Value) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    })
}

pub async fn serve(
    host: &str,
    port: u16,
    default_model: Option<String>,
    bootstrap_sentinels: &[String],
    identity: Identity,
) -> anyhow::Result<()> {
    crate::tui::header(crate::tui::sym("⚡", "*"), "OpenAI-compatible server");
    println!("  node {}", identity.short_id().dimmed());

    if !is_loopback(host) {
        println!();
        println!(
            "  {} binding to {} exposes the API beyond this machine.",
            "⚠".bright_yellow(),
            host.bright_white()
        );
        println!(
            "  {}",
            "The API has no authentication. Do not expose it on an untrusted network."
                .truecolor(220, 180, 120)
        );
    }

    let sentinels = crate::config::resolve_sentinels(bootstrap_sentinels);
    let registry = Arc::new(Mutex::new(PeerRegistry::new(60_000)));

    println!("  {} discovering network...", "→".bright_blue());
    crate::discovery::bootstrap(&sentinels, identity.signing_public().to_vec(), &registry).await;

    crate::discovery::spawn_gossip_loop(
        identity.signing_public().to_vec(),
        "http://127.0.0.1:0".to_string(),
        Arc::clone(&registry),
        Duration::from_secs(2),
        3,
    );
    tokio::time::sleep(Duration::from_secs(3)).await;

    let (_tok_child, mut tokenizer_worker) = start_local_tokenizer(TOKENIZER_PORT).await?;

    let mut loaded_model: Option<String> = None;
    if let Some(model) = default_model.as_deref() {
        tokenizer_worker.load_slice(model, 0, 0, "").await?;
        loaded_model = Some(model.to_string());
    }

    let state = AppState {
        registry: Arc::clone(&registry),
        tokenizer: Arc::new(Mutex::new(TokenizerState {
            worker: tokenizer_worker,
            loaded_model,
        })),
        default_model,
        sentinels,
    };

    let app = Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let bind_addr = format!("{}:{}", host, port);
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;

    println!();
    println!(
        "  {} serving OpenAI API on {}",
        "✓".bright_green(),
        format!("http://{}", bind_addr).bright_white()
    );
    println!(
        "  {} tokenization stays local; prompts only leave over the encrypted pipeline",
        "●".bright_green()
    );
    println!();

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

fn is_loopback(host: &str) -> bool {
    host == "127.0.0.1" || host == "::1" || host == "localhost"
}

async fn list_models(State(state): State<AppState>) -> Response {
    let caps = { analyze(&*state.registry.lock().await) };
    let data = model_list_data(&caps, unix_secs());
    Json(json!({ "object": "list", "data": data })).into_response()
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let cap = {
        let caps = analyze(&*state.registry.lock().await);
        resolve_servable_model(&caps, req.model.clone(), state.default_model.clone())
    };
    let cap = match cap {
        Ok(c) => c,
        Err(e) => return e.into_response(),
    };
    let model = cap.model_id.clone();

    let orch = {
        let reg = state.registry.lock().await;
        build_from_registry(&model, &reg, TARGET_REPLICATION, Vec::new(), state.sentinels.first().cloned()).await
    };
    let orch = match orch {
        Ok(o) => o,
        Err(e) => {
            return ApiError {
                status: StatusCode::SERVICE_UNAVAILABLE,
                message: format!("cannot route model {}: {}", model, e),
                error_type: "service_unavailable",
                code: "route_unavailable",
            }
            .into_response()
        }
    };

    let messages: Vec<(String, String)> = req
        .messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();
    let max_tokens = req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);

    if req.stream.unwrap_or(false) {
        streaming_completion(state, orch, model, messages, max_tokens).await
    } else {
        non_streaming_completion(state, orch, model, messages, max_tokens).await
    }
}

async fn ensure_tokenizer(tk: &mut TokenizerState, model: &str) -> anyhow::Result<()> {
    if tk.loaded_model.as_deref() != Some(model) {
        tk.worker.load_slice(model, 0, 0, "").await?;
        tk.loaded_model = Some(model.to_string());
    }
    Ok(())
}

async fn non_streaming_completion(
    state: AppState,
    mut orch: crate::orchestrator::Orchestrator,
    model: String,
    messages: Vec<(String, String)>,
    max_tokens: usize,
) -> Response {
    let session_id = format!("serve-{}", uuid::Uuid::new_v4());
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = unix_secs();

    let mut tk = state.tokenizer.lock().await;
    if let Err(e) = ensure_tokenizer(&mut tk, &model).await {
        return ApiError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: format!("tokenizer load failed: {}", e),
            error_type: "internal_error",
            code: "tokenizer_error",
        }
        .into_response();
    }

    let (ids, eos) = match tk.worker.encode_messages(messages).await {
        Ok(v) => v,
        Err(e) => {
            return ApiError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                message: format!("encode failed: {}", e),
                error_type: "internal_error",
                code: "encode_error",
            }
            .into_response()
        }
    };

    let out = match orch.generate(&ids, max_tokens, &session_id, Some(eos)).await {
        Ok(o) => o,
        Err(e) => {
            orch.clear_session(&session_id).await;
            return ApiError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                message: format!("generation failed: {}", e),
                error_type: "internal_error",
                code: "generation_error",
            }
            .into_response();
        }
    };

    let new_ids: Vec<i64> = out[ids.len()..].to_vec();
    let stopped_on_eos = new_ids.last() == Some(&eos);
    let finish_reason = if stopped_on_eos { "stop" } else { "length" };

    let text = tk.worker.decode(&new_ids, true).await.unwrap_or_default();
    drop(tk);
    orch.clear_session(&session_id).await;

    let prompt_tokens = ids.len();
    let completion_tokens = new_ids.len();
    let body = json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": text },
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    });
    Json(body).into_response()
}

async fn streaming_completion(
    state: AppState,
    mut orch: crate::orchestrator::Orchestrator,
    model: String,
    messages: Vec<(String, String)>,
    max_tokens: usize,
) -> Response {
    let session_id = format!("serve-{}", uuid::Uuid::new_v4());
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let created = unix_secs();

    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(64);

    tokio::spawn(async move {
        let mut tk = state.tokenizer.lock().await;
        if let Err(e) = ensure_tokenizer(&mut tk, &model).await {
            send_stream_error(&tx, &id, created, &model, format!("tokenizer load failed: {}", e)).await;
            return;
        }

        let (ids, eos) = match tk.worker.encode_messages(messages).await {
            Ok(v) => v,
            Err(e) => {
                send_stream_error(&tx, &id, created, &model, format!("encode failed: {}", e)).await;
                return;
            }
        };

        let role_chunk = chunk_json(&id, created, &model, json!({ "role": "assistant" }), Value::Null);
        if tx.send(Ok(Event::default().data(role_chunk.to_string()))).await.is_err() {
            orch.clear_session(&session_id).await;
            return;
        }

        let finish_reason: &str = {
            let (tok_tx, mut tok_rx) = mpsc::unbounded_channel::<i64>();
            let gen_fut = orch.generate_streaming(&ids, max_tokens, &session_id, Some(eos), move |tid| {
                let _ = tok_tx.send(tid);
            });
            tokio::pin!(gen_fut);

            let mut printed = String::new();
            let mut collected: Vec<i64> = Vec::new();

            loop {
                tokio::select! {
                    maybe = tok_rx.recv() => {
                        if let Some(tid) = maybe {
                            if tid == eos {
                                continue;
                            }
                            collected.push(tid);
                            if let Ok(text) = tk.worker.decode(&collected, true).await {
                                if text.len() > printed.len() && text.starts_with(&printed) {
                                    let delta = text[printed.len()..].to_string();
                                    printed = text;
                                    let chunk = chunk_json(&id, created, &model, json!({ "content": delta }), Value::Null);
                                    if tx.send(Ok(Event::default().data(chunk.to_string()))).await.is_err() {
                                        break "stop";
                                    }
                                }
                            }
                        }
                    }
                    res = &mut gen_fut => {
                        let out = match res {
                            Ok(out) => out,
                            Err(_) => break "stop",
                        };
                        let stopped_on_eos = out.last() == Some(&eos);
                        if out.len() > ids.len() {
                            let new_ids: Vec<i64> = out[ids.len()..].iter().copied().filter(|t| *t != eos).collect();
                            if let Ok(text) = tk.worker.decode(&new_ids, true).await {
                                if text.len() > printed.len() && text.starts_with(&printed) {
                                    let delta = text[printed.len()..].to_string();
                                    let chunk = chunk_json(&id, created, &model, json!({ "content": delta }), Value::Null);
                                    let _ = tx.send(Ok(Event::default().data(chunk.to_string()))).await;
                                }
                            }
                        }
                        break if stopped_on_eos { "stop" } else { "length" };
                    }
                }
            }
        };

        let final_chunk = chunk_json(&id, created, &model, json!({}), json!(finish_reason));
        let _ = tx.send(Ok(Event::default().data(final_chunk.to_string()))).await;
        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;

        drop(tk);
        orch.clear_session(&session_id).await;
    });

    Sse::new(ReceiverStream::new(rx))
        .keep_alive(KeepAlive::new())
        .into_response()
}

async fn send_stream_error(
    tx: &mpsc::Sender<Result<Event, Infallible>>,
    id: &str,
    created: u64,
    model: &str,
    message: String,
) {
    let chunk = chunk_json(id, created, model, json!({ "content": format!("[error: {}]", message) }), json!("stop"));
    let _ = tx.send(Ok(Event::default().data(chunk.to_string()))).await;
    let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{now_ms, Peer, PeerRegistry};

    fn peer(model: &str, ep: &str, start: u32, end: u32, total: u32) -> Peer {
        Peer {
            node_id: ep.as_bytes().to_vec(),
            daemon_endpoint: ep.to_string(),
            worker_endpoint: format!("{}-w", ep),
            model_id: model.to_string(),
            start_layer: start,
            end_layer: end,
            total_layers: total,
            last_seen_ms: now_ms(),
            signature: Vec::new(),
            kx_public: Vec::new(),
            reachable: true,
            protocol_version: crate::registry::WIRE_VERSION,
        }
    }

    fn registry_with(peers: Vec<Peer>) -> PeerRegistry {
        let mut r = PeerRegistry::new(600_000);
        for p in peers {
            r.upsert(p);
        }
        r
    }

    #[test]
    fn models_endpoint_lists_only_servable_models() {
        let reg = registry_with(vec![
            peer("full", "http://a", 0, 12, 12),
            peer("partial", "http://b", 0, 6, 64),
        ]);
        let caps = analyze(&reg);
        let data = model_list_data(&caps, 1000);

        assert_eq!(data.len(), 1);
        assert_eq!(data[0]["id"], "full");
        assert_eq!(data[0]["object"], "model");
        assert_eq!(data[0]["owned_by"], "diffuse");
    }

    #[test]
    fn absent_model_resolves_to_404() {
        let reg = registry_with(vec![peer("full", "http://a", 0, 12, 12)]);
        let caps = analyze(&reg);
        let err = resolve_servable_model(&caps, Some("ghost".to_string()), None).unwrap_err();

        assert_eq!(err.status, StatusCode::NOT_FOUND);
        assert_eq!(err.code, "model_not_found");
    }

    #[test]
    fn incomplete_model_resolves_to_503_naming_missing_slices() {
        let reg = registry_with(vec![peer("partial", "http://b", 0, 6, 64)]);
        let caps = analyze(&reg);
        let err = resolve_servable_model(&caps, Some("partial".to_string()), None).unwrap_err();

        assert_eq!(err.status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(err.code, "model_incomplete");
        assert!(err.message.contains("6:64"), "message names the missing range: {}", err.message);
    }

    #[test]
    fn servable_model_resolves_ok() {
        let reg = registry_with(vec![peer("full", "http://a", 0, 12, 12)]);
        let caps = analyze(&reg);
        let cap = resolve_servable_model(&caps, None, Some("full".to_string())).unwrap();

        assert_eq!(cap.model_id, "full");
        assert!(cap.servable);
    }

    #[test]
    fn missing_model_without_default_is_400() {
        let reg = registry_with(vec![peer("full", "http://a", 0, 12, 12)]);
        let caps = analyze(&reg);
        let err = resolve_servable_model(&caps, None, None).unwrap_err();

        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(err.code, "model_required");
    }

    #[test]
    fn is_loopback_detects_local_and_external_hosts() {
        assert!(is_loopback("127.0.0.1"));
        assert!(is_loopback("localhost"));
        assert!(is_loopback("::1"));
        assert!(!is_loopback("0.0.0.0"));
        assert!(!is_loopback("192.168.1.10"));
    }

    #[test]
    fn streaming_chunk_shape_matches_openai() {
        let chunk = chunk_json("chatcmpl-x", 1000, "full", json!({ "content": "hi" }), Value::Null);
        assert_eq!(chunk["object"], "chat.completion.chunk");
        assert_eq!(chunk["choices"][0]["delta"]["content"], "hi");
        assert!(chunk["choices"][0]["finish_reason"].is_null());
    }

    #[tokio::test]
    async fn incomplete_model_error_renders_503_with_openai_body() {
        let reg = registry_with(vec![peer("partial", "http://b", 0, 6, 64)]);
        let caps = analyze(&reg);
        let err = resolve_servable_model(&caps, Some("partial".to_string()), None).unwrap_err();
        let response = err.into_response();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["type"], "service_unavailable");
        assert_eq!(body["error"]["code"], "model_incomplete");
        assert!(body["error"]["message"].as_str().unwrap().contains("6:64"));
    }
}
