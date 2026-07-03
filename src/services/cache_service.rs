use moka::future::Cache;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use crate::models::database::{Document, Page};
use crate::services::embedding_service::PageEmbedding;

/// TTL for the (document, pages) metadata cache. Metadata is immutable once a
/// document is `complete`, but deletes/reprocess can happen on other nodes, so
/// a short TTL bounds staleness instead of relying on local invalidation only.
const META_TTL: Duration = Duration::from_secs(60);

#[derive(Clone)]
pub struct CacheService {
    /// Embeddings per document, weighed by actual byte size.
    embeddings: Cache<String, Arc<Vec<PageEmbedding>>>,
    /// (Document, pages) for completed documents — keeps Postgres off the
    /// search hot path.
    meta: Cache<String, Arc<(Document, Vec<Page>)>>,
}

impl CacheService {
    pub fn new(max_mb: u64, expiry_hours: u64) -> Self {
        let embeddings = Cache::builder()
            .max_capacity(max_mb.saturating_mul(1024 * 1024))
            .weigher(|_k: &String, v: &Arc<Vec<PageEmbedding>>| {
                // Embeddings live in RAM as bf16 (2 bytes per value).
                let bytes: usize = v
                    .iter()
                    .map(|p| p.data.len() * 2 + 48)
                    .sum::<usize>()
                    + 64;
                u32::try_from(bytes).unwrap_or(u32::MAX)
            })
            // Idle-based expiry: documents that keep being searched stay warm;
            // only truly unused ones expire.
            .time_to_idle(Duration::from_secs(expiry_hours * 3600))
            .build();

        let meta = Cache::builder()
            .max_capacity(10_000)
            .time_to_live(META_TTL)
            .build();

        Self { embeddings, meta }
    }

    pub async fn put(&self, document_id: &str, embeddings: Arc<Vec<PageEmbedding>>) {
        self.embeddings
            .insert(document_id.to_string(), embeddings)
            .await;
    }

    /// Drop everything cached for a document (embeddings + metadata).
    pub async fn remove(&self, document_id: &str) {
        self.embeddings.remove(document_id).await;
        self.meta.remove(document_id).await;
    }

    pub async fn get_meta(&self, document_id: &str) -> Option<Arc<(Document, Vec<Page>)>> {
        self.meta.get(document_id).await
    }

    pub async fn put_meta(&self, document_id: &str, meta: Arc<(Document, Vec<Page>)>) {
        self.meta.insert(document_id.to_string(), meta).await;
    }

    /// Singleflight fetch: if N concurrent callers ask for the same `document_id`
    /// while it's NOT in cache, only ONE runs `fetcher`; the others wait and
    /// receive the same Arc. Failures are also shared (as `Arc<anyhow::Error>`).
    pub async fn try_get_or_fetch<F, Fut>(
        &self,
        document_id: &str,
        fetcher: F,
    ) -> anyhow::Result<Arc<Vec<PageEmbedding>>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = anyhow::Result<Vec<PageEmbedding>>>,
    {
        self.embeddings
            .try_get_with(document_id.to_string(), async move {
                fetcher().await.map(Arc::new)
            })
            .await
            .map_err(|arc_err: Arc<anyhow::Error>| anyhow::anyhow!("{}", arc_err))
    }
}
