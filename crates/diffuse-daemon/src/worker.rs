pub mod pb {
    tonic::include_proto!("diffuse.data");
}

use pb::inference_worker_client::InferenceWorkerClient;
use pb::{ClearSessionRequest, DecodeRequest, EncodeRequest, HealthRequest, LoadSliceRequest, SliceRequest, Tensor};

#[derive(Debug, Clone)]
pub struct WorkerProfile {
    pub total_layers: u32,
    pub max_layers: u32,
    pub max_layers_if_holding_embedding: u32,
    pub avg_layer_bytes: u64,
    pub available_bytes: u64,
    pub device: String,
}

#[derive(Clone)]
pub struct WorkerHandle {
    pub endpoint: String,
    client: InferenceWorkerClient<tonic::transport::Channel>,
}

impl WorkerHandle {
    pub async fn connect(endpoint: String) -> anyhow::Result<Self> {
        let client = InferenceWorkerClient::connect(endpoint.clone())
            .await?
            .max_decoding_message_size(128 * 1024 * 1024)
            .max_encoding_message_size(128 * 1024 * 1024);
        Ok(Self { endpoint, client })
    }

    pub async fn clear_session(&mut self, session_id: &str) -> anyhow::Result<()> {
        self.client
            .clear_session(ClearSessionRequest {
                session_id: session_id.to_string(),
            })
            .await?;
        Ok(())
    }

    pub async fn load_slice(
        &mut self,
        model_id: &str,
        start_layer: u32,
        end_layer: u32,
        hf_token: &str,
    ) -> anyhow::Result<u32> {
        let request = tonic::Request::new(LoadSliceRequest {
            model_id: model_id.to_string(),
            start_layer,
            end_layer,
            hf_token: hf_token.to_string(),
        });
        let response = self.client.load_slice(request).await?.into_inner();
        if !response.ok {
            anyhow::bail!("worker load failed: {}", response.error);
        }
        Ok(response.total_layers)
    }

    pub async fn run_slice(
        &mut self,
        model_id: &str,
        start_layer: u32,
        end_layer: u32,
        session_id: &str,
        position: u64,
        activations: Tensor,
        use_cache: bool,
        top_k: u32,
        accepts_bf16: bool,
    ) -> anyhow::Result<Tensor> {
        let request = tonic::Request::new(SliceRequest {
            model_id: model_id.to_string(),
            start_layer,
            end_layer,
            session_id: session_id.to_string(),
            position,
            activations: Some(activations),
            use_cache,
            top_k,
            accepts_bf16,
        });
        let response = self.client.run_slice(request).await?.into_inner();
        if !response.ok {
            anyhow::bail!("worker run failed: {}", response.error);
        }
        response
            .activations
            .ok_or_else(|| anyhow::anyhow!("worker returned no activations"))
    }

    pub async fn encode(&mut self, text: &str, chat_template: bool) -> anyhow::Result<(Vec<i64>, i64)> {
        let request = tonic::Request::new(EncodeRequest {
            text: text.to_string(),
            apply_chat_template: chat_template,
            messages: Vec::new(),
        });
        let response = self.client.encode(request).await?.into_inner();
        if !response.ok {
            anyhow::bail!("worker encode failed: {}", response.error);
        }
        Ok((response.token_ids, response.eos_token_id))
    }

    pub async fn decode(&mut self, token_ids: &[i64], skip_special: bool) -> anyhow::Result<String> {
        let request = tonic::Request::new(DecodeRequest {
            token_ids: token_ids.to_vec(),
            skip_special_tokens: skip_special,
        });
        let response = self.client.decode(request).await?.into_inner();
        if !response.ok {
            anyhow::bail!("worker decode failed: {}", response.error);
        }
        Ok(response.text)
    }

    pub async fn health(&mut self) -> anyhow::Result<bool> {
        let request = tonic::Request::new(HealthRequest {});
        let response = self.client.health(request).await?.into_inner();
        Ok(response.ok && response.slice_loaded)
    }

    pub async fn profile_model(
        &mut self,
        model_id: &str,
        overhead_fraction: f64,
    ) -> anyhow::Result<WorkerProfile> {
        let request = tonic::Request::new(pb::ProfileRequest {
            model_id: model_id.to_string(),
            hf_token: String::new(),
            overhead_fraction,
        });
        let r = self.client.profile_model(request).await?.into_inner();
        if !r.ok {
            anyhow::bail!("profile failed: {}", r.error);
        }
        Ok(WorkerProfile {
            total_layers: r.total_layers,
            max_layers: r.max_layers,
            max_layers_if_holding_embedding: r.max_layers_if_holding_embedding,
            avg_layer_bytes: r.avg_layer_bytes,
            available_bytes: r.available_bytes,
            device: r.device,
        })
    }

    pub async fn search_models(
        &mut self,
        query: &str,
        limit: u32,
        supported_only: bool,
    ) -> anyhow::Result<(Vec<pb::ModelCard>, u64, String)> {
        let request = tonic::Request::new(pb::SearchModelsRequest {
            query: query.to_string(),
            limit,
            hf_token: String::new(),
            supported_only,
        });
        let response = self.client.search_models(request).await?.into_inner();
        if !response.ok {
            anyhow::bail!("model search failed: {}", response.error);
        }
        Ok((response.models, response.available_bytes, response.device))
    }

    pub async fn embed_media(
        &mut self,
        text: &str,
        media: Vec<(String, Vec<u8>, String)>,
        apply_chat_template: bool,
        accepts_bf16: bool,
    ) -> anyhow::Result<(Tensor, u32)> {
        let attachments: Vec<pb::MediaAttachment> = media
            .into_iter()
            .map(|(kind, data, mime)| pb::MediaAttachment { kind, data, mime })
            .collect();
        let request = tonic::Request::new(pb::EmbedMediaRequest {
            text: text.to_string(),
            messages: Vec::new(),
            media: attachments,
            apply_chat_template,
            accepts_bf16,
        });
        let response = self.client.embed_media(request).await?.into_inner();
        if !response.ok {
            anyhow::bail!("media embedding failed: {}", response.error);
        }
        let embeddings = response
            .embeddings
            .ok_or_else(|| anyhow::anyhow!("worker returned no embeddings"))?;
        Ok((embeddings, response.token_count))
    }

    pub async fn encode_messages(
        &mut self,
        messages: Vec<(String, String)>,
    ) -> anyhow::Result<(Vec<i64>, i64)> {
        let msgs: Vec<pb::ChatMessage> = messages
            .into_iter()
            .map(|(role, content)| pb::ChatMessage { role, content })
            .collect();
        let request = tonic::Request::new(EncodeRequest {
            text: String::new(),
            apply_chat_template: true,
            messages: msgs,
        });
        let response = self.client.encode(request).await?.into_inner();
        if !response.ok {
            anyhow::bail!("worker encode failed: {}", response.error);
        }
        Ok((response.token_ids, response.eos_token_id))
    }
}