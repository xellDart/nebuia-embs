use moka::future::Cache;
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use crate::services::embedding_service::PageEmbedding;

#[derive(Clone)]
pub struct CacheService {
    cache: Cache<String, Arc<Vec<PageEmbedding>>>,
}

impl CacheService {
    pub fn new(max_size: u64, expiry_hours: u64) -> Self {
        let cache = Cache::builder()
            .max_capacity(max_size)
            .time_to_live(Duration::from_secs(expiry_hours * 3600))
            .build();
        Self { cache }
    }

    pub async fn get(&self, document_id: &str) -> Option<Arc<Vec<PageEmbedding>>> {
        self.cache.get(document_id).await
    }

    pub async fn put(&self, document_id: &str, embeddings: Vec<PageEmbedding>) {
        self.cache
            .insert(document_id.to_string(), Arc::new(embeddings))
            .await;
    }

    pub async fn remove(&self, document_id: &str) {
        self.cache.remove(document_id).await;
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
        self.cache
            .try_get_with(document_id.to_string(), async move {
                fetcher().await.map(Arc::new)
            })
            .await
            .map_err(|arc_err: Arc<anyhow::Error>| anyhow::anyhow!("{}", arc_err))
    }
}
