use crate::compute::request_slice;
use crate::worker::pb::Tensor;
use crate::worker::WorkerHandle;
use diffuse_trust::transport::KeyExchange;

pub struct RemoteStage {
    pub daemon_endpoint: String,
    pub host_kx_public: [u8; 32],
    pub start_layer: u32,
    pub end_layer: u32,
}

fn ids_to_tensor(ids: &[i64]) -> Tensor {
    let mut data = Vec::with_capacity(ids.len() * 8);
    for id in ids {
        data.extend_from_slice(&id.to_le_bytes());
    }
    Tensor {
        shape: vec![1, ids.len() as i64],
        dtype: "int64".to_string(),
        data,
    }
}

fn argmax_last_token(logits: &Tensor) -> anyhow::Result<i64> {
    if logits.shape.len() != 3 {
        anyhow::bail!("expected 3D logits, got {:?}", logits.shape);
    }
    let seq = logits.shape[1] as usize;
    let vocab = logits.shape[2] as usize;
    let floats: &[f32] = bytemuck::cast_slice(&logits.data);
    let offset = (seq - 1) * vocab;
    let row = &floats[offset..offset + vocab];
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    Ok(best_idx as i64)
}

pub struct ClientSession {
    pub local_worker: WorkerHandle,
    pub client_kx: KeyExchange,
    pub model_id: String,
    pub local_start: u32,
    pub local_end: u32,
    pub remote: RemoteStage,
}

impl ClientSession {
    async fn forward(&mut self, ids: &[i64], session_id: &str) -> anyhow::Result<Tensor> {
        let input = ids_to_tensor(ids);

        let local_activations = self
            .local_worker
            .run_slice(
                &self.model_id,
                self.local_start,
                self.local_end,
                session_id,
                0,
                input,
                false,
            )
            .await?;

        let logits = request_slice(
            &self.remote.daemon_endpoint,
            &self.remote.host_kx_public,
            &self.client_kx,
            &self.model_id,
            self.remote.start_layer,
            self.remote.end_layer,
            session_id,
            &local_activations,
        )
        .await?;

        Ok(logits)
    }
    pub async fn generate(
        &mut self,
        prompt_ids: &[i64],
        max_new_tokens: usize,
        session_id: &str,
        eos_id: Option<i64>,
    ) -> anyhow::Result<Vec<i64>> {
        let mut ids = prompt_ids.to_vec();
        for _ in 0..max_new_tokens {
            let logits = self.forward(&ids, session_id).await?;
            let next = argmax_last_token(&logits)?;
            ids.push(next);
            if Some(next) == eos_id {
                break;
            }
        }
        Ok(ids)
    }
}