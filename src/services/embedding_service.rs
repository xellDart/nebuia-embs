use anyhow::Result;
use candle_core::{Device, Tensor};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot, Semaphore};
use tracing::{error, info};

/// Serialized embedding: list of pages, each page is (seq_len, dims) flattened.
///
/// Kept as bf16 end-to-end: it's the exact storage format (lossless
/// serialize/deserialize), halves RAM in cache and PCIe transfer per search.
/// The model computes in bf16, so its outputs are bf16-representable and this
/// loses nothing; scoring upcasts to f32 ON DEVICE, which is exact, so scores
/// are bit-identical to expanding on the CPU as before.
#[derive(Debug, Clone)]
pub struct PageEmbedding {
    pub seq_len: usize,
    pub dims: usize,
    pub data: Vec<half::bf16>,
}

/// One document's page embeddings resident on the GPU as ready-to-score f32
/// tensors. `bytes` is the device footprint used for budget accounting.
struct GpuCacheEntry {
    doc_id: String,
    tensors: Arc<Vec<Tensor>>,
    bytes: u64,
    last_used: Instant,
}

/// LRU cache of documents' page tensors in VRAM, bounded by a byte budget and
/// an idle TTL. The typical workload is a burst of ~20-30 searches right after
/// a document is ingested and almost nothing later, so entries live minutes,
/// not hours. VRAM is the scarcest resource on the box: the budget must stay
/// small enough that the encoder's peak allocations are never squeezed.
struct GpuDocCache {
    entries: Vec<GpuCacheEntry>,
    budget_bytes: u64,
    idle: Duration,
}

impl GpuDocCache {
    fn get(&mut self, doc_id: &str) -> Option<Arc<Vec<Tensor>>> {
        self.evict_expired();
        let e = self.entries.iter_mut().find(|e| e.doc_id == doc_id)?;
        e.last_used = Instant::now();
        Some(e.tensors.clone())
    }

    fn insert(&mut self, doc_id: String, tensors: Arc<Vec<Tensor>>, bytes: u64) {
        // A document bigger than the whole budget is never cached.
        if bytes > self.budget_bytes {
            return;
        }
        self.entries.retain(|e| e.doc_id != doc_id);
        let mut used: u64 = self.entries.iter().map(|e| e.bytes).sum();
        while used + bytes > self.budget_bytes {
            let Some((idx, _)) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_used)
            else {
                break;
            };
            used -= self.entries[idx].bytes;
            self.entries.swap_remove(idx);
        }
        self.entries.push(GpuCacheEntry {
            doc_id,
            tensors,
            bytes,
            last_used: Instant::now(),
        });
    }

    fn evict_expired(&mut self) {
        let now = Instant::now();
        self.entries
            .retain(|e| now.duration_since(e.last_used) < self.idle);
    }

    fn invalidate(&mut self, doc_id: &str) {
        self.entries.retain(|e| e.doc_id != doc_id);
    }
}

struct ImagesRequest {
    images: Vec<Vec<u8>>,
    reply: oneshot::Sender<Result<Vec<PageEmbedding>>>,
}

struct QueryRequest {
    query: String,
    reply: oneshot::Sender<Result<Vec<PageEmbedding>>>,
}

#[derive(Clone)]
pub struct EmbeddingService {
    /// Bounded sender → applies backpressure to HTTP/NATS callers when the
    /// model is saturated (instead of growing memory unboundedly).
    images_tx: mpsc::Sender<ImagesRequest>,
    /// Queries take a separate, prioritized lane: the model thread drains all
    /// pending queries before picking up the next image batch, so a search
    /// never waits behind multi-second document encodes.
    query_tx: mpsc::Sender<QueryRequest>,
    /// Device handle used by score() on a separate (blocking) task.
    /// Same physical GPU as the model; CUDA driver schedules concurrent kernels.
    device: Arc<Device>,
    /// Caps concurrent score() calls. Each search holds its page tensors on
    /// the GPU while scoring; unbounded concurrency OOMs the device (seen in
    /// prod: 4 simultaneous searches on a 224-page document).
    score_semaphore: Arc<Semaphore>,
    /// VRAM-resident page tensors per document (None = disabled).
    gpu_cache: Option<Arc<Mutex<GpuDocCache>>>,
}

impl EmbeddingService {
    /// Spawn the model on a dedicated thread. `queue_capacity` bounds the
    /// in-flight image-encode queue (caller `send().await` blocks when full).
    /// `image_batch` is how many images are encoded per model forward
    /// (1 = one at a time; >1 batches the vision tower, more VRAM).
    /// `score_concurrency` caps simultaneous score() calls on the device.
    /// `gpu_cache_mb` > 0 keeps recently-searched documents' page tensors
    /// resident in VRAM up to that budget (0 = upload per search, as before).
    pub fn spawn(
        model_path: &str,
        use_cpu: bool,
        use_bf16: bool,
        target_dims: Option<usize>,
        queue_capacity: usize,
        image_batch: usize,
        score_concurrency: usize,
        gpu_cache_mb: u64,
        gpu_cache_idle_secs: u64,
    ) -> Result<Self> {
        // Build the score-side device on the calling thread so we can hand it
        // back to the service immediately, even if the model thread is still loading.
        let device = if use_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0)?
        };
        let device = Arc::new(device);

        let (images_tx, mut images_rx) = mpsc::channel::<ImagesRequest>(queue_capacity.max(1));
        let (query_tx, mut query_rx) = mpsc::channel::<QueryRequest>(64);
        let path = model_path.to_string();
        let image_batch = image_batch.max(1);

        // Handle used only to await the two channels from the model thread.
        let rt = tokio::runtime::Handle::current();

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
                    "Embedding model ready (image queue: {}, image batch: {})",
                    queue_capacity, image_batch
                );

                loop {
                    // Biased select: pending queries always win over image batches.
                    // A closed channel disables its branch; `else` fires when both
                    // are closed → shutdown.
                    enum Req {
                        Query(QueryRequest),
                        Images(ImagesRequest),
                    }
                    let req = rt.block_on(async {
                        tokio::select! {
                            biased;
                            Some(q) = query_rx.recv() => Some(Req::Query(q)),
                            Some(i) = images_rx.recv() => Some(Req::Images(i)),
                            else => None,
                        }
                    });
                    match req {
                        Some(Req::Query(QueryRequest { query, reply })) => {
                            let result = encode_query(&mut model, &query);
                            let _ = reply.send(result);
                        }
                        Some(Req::Images(ImagesRequest { images, reply })) => {
                            let result = encode_images_from_bytes(&mut model, &images, image_batch);
                            let _ = reply.send(result);
                        }
                        None => break,
                    }
                }
                info!("Embedding model thread shutting down");
            })?;

        let gpu_cache = (gpu_cache_mb > 0).then(|| {
            info!(
                "GPU doc cache enabled: {} MB budget, {}s idle TTL",
                gpu_cache_mb, gpu_cache_idle_secs
            );
            Arc::new(Mutex::new(GpuDocCache {
                entries: Vec::new(),
                budget_bytes: gpu_cache_mb * 1024 * 1024,
                idle: Duration::from_secs(gpu_cache_idle_secs),
            }))
        });

        Ok(Self {
            images_tx,
            query_tx,
            device,
            score_semaphore: Arc::new(Semaphore::new(score_concurrency.max(1))),
            gpu_cache,
        })
    }

    /// Submit images for encoding. **Awaits** if the queue is full → backpressure.
    pub async fn encode_images_from_bytes(&self, images: Vec<Vec<u8>>) -> Result<Vec<PageEmbedding>> {
        // Give the encoder its VRAM back: drop cached docs nobody searched
        // recently before the next multi-image forward allocates its peak.
        if let Some(cache) = &self.gpu_cache {
            cache.lock().unwrap().evict_expired();
        }
        let (reply_tx, reply_rx) = oneshot::channel();
        self.images_tx
            .send(ImagesRequest {
                images,
                reply: reply_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("Model thread died"))?;
        reply_rx.await?
    }

    /// Submit a query for encoding on the prioritized lane.
    pub async fn encode_query(&self, query: String) -> Result<Vec<PageEmbedding>> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.query_tx
            .send(QueryRequest {
                query,
                reply: reply_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("Model thread died"))?;
        reply_rx.await?
    }

    /// Score query embeddings against a document's page embeddings.
    ///
    /// Runs on the tokio blocking pool, NOT on the model thread — so a search
    /// request never queues behind a long-running document encode.
    /// Takes the cached pages by Arc: no deep copy of the embeddings.
    /// With the GPU cache enabled, the document's page tensors are reused
    /// across searches instead of being re-uploaded each time.
    pub async fn score(
        &self,
        document_id: &str,
        query_embs: Vec<PageEmbedding>,
        page_embs: Arc<Vec<PageEmbedding>>,
    ) -> Result<Vec<f32>> {
        let _permit = self.score_semaphore.clone().acquire_owned().await?;
        let device = self.device.clone();
        let gpu_cache = self.gpu_cache.clone();
        let doc_id = document_id.to_string();
        tokio::task::spawn_blocking(move || {
            let cached = gpu_cache.as_ref().and_then(|c| c.lock().unwrap().get(&doc_id));
            let ps = match cached {
                Some(t) => t,
                None => {
                    // Two concurrent misses on the same doc build twice; the
                    // semaphore bounds that transient, so no lock across upload.
                    let tensors: Vec<Tensor> = page_embs
                        .iter()
                        .map(|e| page_embedding_to_tensor(e, &device))
                        .collect::<Result<_>>()?;
                    let tensors = Arc::new(tensors);
                    if let Some(c) = &gpu_cache {
                        // Resident as f32: 4 bytes per value.
                        let bytes: u64 =
                            page_embs.iter().map(|e| (e.data.len() * 4) as u64).sum();
                        c.lock().unwrap().insert(doc_id, tensors.clone(), bytes);
                    }
                    tensors
                }
            };
            score_on_device(&query_embs, &ps, &device)
        })
        .await?
    }

    /// Drop a document's tensors from the GPU cache (on delete/reprocess).
    pub fn invalidate_gpu(&self, document_id: &str) {
        if let Some(c) = &self.gpu_cache {
            c.lock().unwrap().invalidate(document_id);
        }
    }

    /// Pre-upload a document's page tensors right after ingest, so the search
    /// burst that follows `complete` hits a warm cache instead of stampeding
    /// cold misses. No-op when the cache is disabled or the doc doesn't fit.
    pub async fn warm_gpu(
        &self,
        document_id: &str,
        page_embs: Arc<Vec<PageEmbedding>>,
    ) -> Result<()> {
        let Some(cache) = self.gpu_cache.clone() else {
            return Ok(());
        };
        // Warming is a device upload like any search miss: take a score
        // permit so it can't stack on top of concurrent searches.
        let _permit = self.score_semaphore.clone().acquire_owned().await?;
        let device = self.device.clone();
        let doc_id = document_id.to_string();
        tokio::task::spawn_blocking(move || {
            let bytes: u64 = page_embs.iter().map(|e| (e.data.len() * 4) as u64).sum();
            if bytes > cache.lock().unwrap().budget_bytes {
                return Ok(());
            }
            let tensors: Vec<Tensor> = page_embs
                .iter()
                .map(|e| page_embedding_to_tensor(e, &device))
                .collect::<Result<_>>()?;
            cache.lock().unwrap().insert(doc_id, Arc::new(tensors), bytes);
            Ok(())
        })
        .await?
    }

    pub fn is_alive(&self) -> bool {
        !self.images_tx.is_closed()
    }
}

// ── Model-thread functions (sync, run on the model thread) ──────────

fn tensor_to_page_embedding(t: &Tensor) -> Result<PageEmbedding> {
    let dims = t.dims();
    let (seq_len, d) = match dims.len() {
        2 => (dims[0], dims[1]),
        _ => anyhow::bail!("Expected 2D tensor, got {}D", dims.len()),
    };
    // Model outputs are bf16 already (or exactly representable); no-op cast in
    // the common case, and matches the on-disk format bit for bit.
    let data: Vec<half::bf16> = t
        .to_dtype(candle_core::DType::BF16)?
        .flatten_all()?
        .to_vec1()?;
    Ok(PageEmbedding {
        seq_len,
        dims: d,
        data,
    })
}

fn encode_images_from_bytes(
    model: &mut crane_core::models::colqwen3_emb::ColQwen3Emb,
    images: &[Vec<u8>],
    image_batch: usize,
) -> Result<Vec<PageEmbedding>> {
    let profile = std::env::var("CRANE_PROFILE").map(|v| v == "1").unwrap_or(false);
    let mut all = Vec::with_capacity(images.len());
    let (mut t_model, mut t_convert) = (0f64, 0f64);
    for chunk in images.chunks(image_batch) {
        let refs: Vec<&[u8]> = chunk.iter().map(|v| v.as_slice()).collect();
        let t0 = std::time::Instant::now();
        let tensors = model.encode_images_from_bytes(&refs)?;
        let t1 = std::time::Instant::now();
        for t in &tensors {
            all.push(tensor_to_page_embedding(t)?);
        }
        t_model += t1.duration_since(t0).as_secs_f64();
        t_convert += t1.elapsed().as_secs_f64();
    }
    if profile {
        tracing::info!(
            "encode profile: crane {:.0}ms, tensor->PageEmbedding {:.0}ms",
            t_model * 1e3,
            t_convert * 1e3
        );
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
    // Transfer to device as bf16 (half the PCIe traffic), then upcast to f32
    // ON DEVICE. bf16→f32 is exact, so the scoring math is unchanged.
    Ok(Tensor::from_slice(&emb.data, (emb.seq_len, emb.dims), device)?
        .to_dtype(candle_core::DType::F32)?)
}

fn score_on_device(
    query_embs: &[PageEmbedding],
    page_tensors: &[Tensor],
    device: &Device,
) -> Result<Vec<f32>> {
    let qs: Vec<Tensor> = query_embs
        .iter()
        .map(|e| page_embedding_to_tensor(e, device))
        .collect::<Result<_>>()?;

    let scores = crane_core::models::colqwen3_emb::ColQwen3Emb::score(&qs, page_tensors, 128)?;
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
// CPU-bound: callers must run these on the blocking pool (spawn_blocking),
// never directly on an async worker.

/// Serialize page embeddings to raw bytes as bf16:
/// [num_pages(u32), then for each page: seq_len(u32), dims(u32), bf16 data...]
/// Data is already bf16 in memory, so this is a pure byte copy (lossless).
pub fn serialize_embeddings(embeddings: &[PageEmbedding]) -> Vec<u8> {
    let total: usize = 4 + embeddings
        .iter()
        .map(|e| 8 + e.data.len() * 2)
        .sum::<usize>();
    let mut buf = Vec::with_capacity(total);
    buf.extend_from_slice(&(embeddings.len() as u32).to_le_bytes());
    for emb in embeddings {
        buf.extend_from_slice(&(emb.seq_len as u32).to_le_bytes());
        buf.extend_from_slice(&(emb.dims as u32).to_le_bytes());
        for &val in &emb.data {
            buf.extend_from_slice(&val.to_le_bytes());
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

        // Stays bf16 in memory: deserialization is a pure byte copy.
        let mut values = Vec::with_capacity(num_values);
        values.extend(
            data[cursor..cursor + byte_len]
                .chunks_exact(2)
                .map(|b| half::bf16::from_le_bytes([b[0], b[1]])),
        );
        cursor += byte_len;

        embeddings.push(PageEmbedding {
            seq_len,
            dims,
            data: values,
        });
    }

    Ok(embeddings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_deserialize_roundtrip_is_lossless() {
        let values: Vec<half::bf16> = [
            0.0f32, -0.0, 1.0, -1.0, 0.123456, -3.14159, 1e-20, -1e-20, 65504.0, 1e30, -1e30,
            f32::MIN_POSITIVE, 0.999999,
        ]
        .iter()
        .map(|v| half::bf16::from_f32(*v))
        .collect();
        let page = PageEmbedding {
            seq_len: values.len(),
            dims: 1,
            data: values.clone(),
        };
        let raw = serialize_embeddings(&[page]);
        let out = deserialize_embeddings(&raw).unwrap();

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].seq_len, values.len());
        assert_eq!(out[0].dims, 1);
        // bf16 in memory == bf16 on disk: the roundtrip must be bit-exact.
        for (orig, got) in values.iter().zip(out[0].data.iter()) {
            assert_eq!(orig.to_bits(), got.to_bits());
        }
    }

    #[test]
    fn deserialize_rejects_truncated_data() {
        let page = PageEmbedding {
            seq_len: 4,
            dims: 2,
            data: vec![half::bf16::from_f32(1.0); 8],
        };
        let raw = serialize_embeddings(&[page]);
        assert!(deserialize_embeddings(&raw[..raw.len() - 3]).is_err());
    }
}
