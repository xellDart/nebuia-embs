use anyhow::Result;
use candle_core::{Device, Tensor};
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tracing::{error, info};

/// Serialized embedding: list of pages, each page is (seq_len, dims) flattened.
/// Stored as bf16 (2 bytes) for storage, converted to f32 for scoring.
#[derive(Debug, Clone)]
pub struct PageEmbedding {
    pub seq_len: usize,
    pub dims: usize,
    pub data: Vec<f32>,
}

/// Encode-only requests run on the dedicated model thread (single stream of inference).
/// Score requests do NOT go here — they run on the tokio blocking pool sharing
/// only the device handle, so search latency is not blocked behind heavy encode jobs.
enum EncodeRequest {
    Images {
        images: Vec<Vec<u8>>,
        reply: oneshot::Sender<Result<Vec<PageEmbedding>>>,
    },
    Query {
        query: String,
        reply: oneshot::Sender<Result<Vec<PageEmbedding>>>,
    },
}

#[derive(Clone)]
pub struct EmbeddingService {
    /// Bounded sender → applies backpressure to HTTP/NATS callers when the
    /// model is saturated (instead of growing memory unboundedly).
    encode_tx: mpsc::Sender<EncodeRequest>,
    /// Device handle used by score() on a separate (blocking) task.
    /// Same physical GPU as the model; CUDA driver schedules concurrent kernels.
    device: Arc<Device>,
}

impl EmbeddingService {
    /// Spawn the model on a dedicated thread. `queue_capacity` bounds the
    /// in-flight encode queue (caller `send().await` blocks when full).
    pub fn spawn(
        model_path: &str,
        use_cpu: bool,
        use_bf16: bool,
        target_dims: Option<usize>,
        queue_capacity: usize,
    ) -> Result<Self> {
        // Build the score-side device on the calling thread so we can hand it
        // back to the service immediately, even if the model thread is still loading.
        let device = if use_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0)?
        };
        let device = Arc::new(device);

        let (tx, mut rx) = mpsc::channel::<EncodeRequest>(queue_capacity.max(1));
        let path = model_path.to_string();

        std::thread::Builder::new()
            .name("embedding-model".into())
            .spawn(move || {
                info!("Loading ColQwen3 embedding model from: {}", path);
                let mut model = match crane_core::models::colqwen3_emb::ColQwen3Emb::from_local(
                    &path, use_cpu, use_bf16,
                ) {
                    Ok(m) => m,
                    Err(e) => {
                        error!("Failed to load embedding model: {}", e);
                        return;
                    }
                };
                if let Some(dims) = target_dims {
                    model.set_dims(dims);
                    info!("Embedding dims set to {}", dims);
                }
                info!(
                    "Embedding model ready (encode queue capacity: {})",
                    queue_capacity
                );

                while let Some(req) = rx.blocking_recv() {
                    match req {
                        EncodeRequest::Images { images, reply } => {
                            let result = encode_images_from_bytes(&mut model, &images);
                            let _ = reply.send(result);
                        }
                        EncodeRequest::Query { query, reply } => {
                            let result = encode_query(&mut model, &query);
                            let _ = reply.send(result);
                        }
                    }
                }
                info!("Embedding model thread shutting down");
            })?;

        Ok(Self {
            encode_tx: tx,
            device,
        })
    }

    /// Submit images for encoding. **Awaits** if the queue is full → backpressure.
    pub async fn encode_images_from_bytes(&self, images: Vec<Vec<u8>>) -> Result<Vec<PageEmbedding>> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.encode_tx
            .send(EncodeRequest::Images {
                images,
                reply: reply_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("Model thread died"))?;
        reply_rx.await?
    }

    /// Submit a query for encoding. **Awaits** if the queue is full → backpressure.
    pub async fn encode_query(&self, query: String) -> Result<Vec<PageEmbedding>> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.encode_tx
            .send(EncodeRequest::Query {
                query,
                reply: reply_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("Model thread died"))?;
        reply_rx.await?
    }

    /// Score query embeddings against page embeddings.
    ///
    /// Runs on the tokio blocking pool, NOT on the model thread — so a search
    /// request never queues behind a long-running document encode.
    pub async fn score(
        &self,
        query_embs: Vec<PageEmbedding>,
        page_embs: Vec<PageEmbedding>,
    ) -> Result<Vec<f32>> {
        let device = self.device.clone();
        tokio::task::spawn_blocking(move || score_on_device(&device, &query_embs, &page_embs))
            .await?
    }

    pub fn is_alive(&self) -> bool {
        !self.encode_tx.is_closed()
    }

    /// Approximate number of slots free in the encode queue (for metrics).
    pub fn encode_capacity_available(&self) -> usize {
        self.encode_tx.capacity()
    }
}

// ── Model-thread functions (sync, run on the model thread) ──────────

fn tensor_to_page_embedding(t: &Tensor) -> Result<PageEmbedding> {
    let dims = t.dims();
    let (seq_len, d) = match dims.len() {
        2 => (dims[0], dims[1]),
        _ => anyhow::bail!("Expected 2D tensor, got {}D", dims.len()),
    };
    let data: Vec<f32> = t.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1()?;
    Ok(PageEmbedding {
        seq_len,
        dims: d,
        data,
    })
}

fn encode_images_from_bytes(
    model: &mut crane_core::models::colqwen3_emb::ColQwen3Emb,
    images: &[Vec<u8>],
) -> Result<Vec<PageEmbedding>> {
    let mut all = Vec::new();
    for img_bytes in images {
        let tensors = model.encode_images_from_bytes(&[img_bytes.as_slice()])?;
        for t in &tensors {
            all.push(tensor_to_page_embedding(t)?);
        }
    }
    Ok(all)
}

fn encode_query(
    model: &mut crane_core::models::colqwen3_emb::ColQwen3Emb,
    query: &str,
) -> Result<Vec<PageEmbedding>> {
    let tensors = model.encode_queries(&[query])?;
    tensors.iter().map(tensor_to_page_embedding).collect()
}

// ── Score function (runs on tokio blocking pool, no model dependency) ──

fn page_embedding_to_tensor(emb: &PageEmbedding, device: &Device) -> Result<Tensor> {
    Ok(Tensor::from_vec(
        emb.data.clone(),
        (emb.seq_len, emb.dims),
        device,
    )?)
}

fn score_on_device(
    device: &Device,
    query_embs: &[PageEmbedding],
    page_embs: &[PageEmbedding],
) -> Result<Vec<f32>> {
    let qs: Vec<Tensor> = query_embs
        .iter()
        .map(|e| page_embedding_to_tensor(e, device))
        .collect::<Result<_>>()?;

    let ps: Vec<Tensor> = page_embs
        .iter()
        .map(|e| page_embedding_to_tensor(e, device))
        .collect::<Result<_>>()?;

    let scores = crane_core::models::colqwen3_emb::ColQwen3Emb::score(&qs, &ps, 128)?;
    let scores_vec: Vec<f32> = scores.squeeze(0)?.to_vec1()?;

    let mut indexed: Vec<(usize, f32)> = scores_vec.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top: Vec<String> = indexed
        .iter()
        .take(10)
        .map(|(i, s)| format!("p{}={:.1}", i, s))
        .collect();
    info!("Scores top-10: {}", top.join(", "));

    Ok(scores_vec)
}

// ── Serialization for MinIO storage ──────────────────────────────────

/// Serialize page embeddings to raw bytes as bf16:
/// [num_pages(u32), then for each page: seq_len(u32), dims(u32), bf16 data...]
pub fn serialize_embeddings(embeddings: &[PageEmbedding]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(embeddings.len() as u32).to_le_bytes());
    for emb in embeddings {
        buf.extend_from_slice(&(emb.seq_len as u32).to_le_bytes());
        buf.extend_from_slice(&(emb.dims as u32).to_le_bytes());
        for &val in &emb.data {
            buf.extend_from_slice(&half::bf16::from_f32(val).to_le_bytes());
        }
    }
    buf
}

/// Deserialize page embeddings from raw bf16 bytes.
pub fn deserialize_embeddings(data: &[u8]) -> Result<Vec<PageEmbedding>> {
    let mut cursor = 0;

    let read_u32 = |cursor: &mut usize| -> Result<u32> {
        if *cursor + 4 > data.len() {
            anyhow::bail!("Unexpected end of embeddings data");
        }
        let val = u32::from_le_bytes(data[*cursor..*cursor + 4].try_into()?);
        *cursor += 4;
        Ok(val)
    };

    let num_pages = read_u32(&mut cursor)? as usize;
    let mut embeddings = Vec::with_capacity(num_pages);

    for _ in 0..num_pages {
        let seq_len = read_u32(&mut cursor)? as usize;
        let dims = read_u32(&mut cursor)? as usize;
        let num_values = seq_len * dims;
        let byte_len = num_values * 2; // bf16 = 2 bytes

        if cursor + byte_len > data.len() {
            anyhow::bail!("Unexpected end of embeddings data");
        }

        let mut float_data = Vec::with_capacity(num_values);
        for i in 0..num_values {
            let offset = cursor + i * 2;
            let val = half::bf16::from_le_bytes(data[offset..offset + 2].try_into()?);
            float_data.push(val.to_f32());
        }
        cursor += byte_len;

        embeddings.push(PageEmbedding {
            seq_len,
            dims,
            data: float_data,
        });
    }

    Ok(embeddings)
}
