use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use sqlx::PgPool;
use tokio::sync::Semaphore;

use crate::config::AppConfig;
use crate::repositories::storage_repository::StorageRepository;
use crate::services::cache_service::CacheService;
use crate::services::embedding_service::EmbeddingService;

pub struct AppState {
    pub config: AppConfig,
    pub db_pool: PgPool,
    pub storage: StorageRepository,
    pub embedding: EmbeddingService,
    pub cache: CacheService,
    pub process_semaphore: Arc<Semaphore>,
    /// Document ids currently being processed, to stop the HTTP endpoint and the
    /// NATS consumer from encoding the same document twice in parallel.
    pub in_flight: Arc<Mutex<HashSet<String>>>,
}

impl AppState {
    /// Claim exclusive processing of `document_id`. Returns `None` if another task
    /// already holds it; the returned guard releases the claim when dropped.
    pub fn try_claim(&self, document_id: &str) -> Option<InFlightGuard> {
        let mut set = self.in_flight.lock().unwrap();
        if !set.insert(document_id.to_string()) {
            return None;
        }
        Some(InFlightGuard {
            set: self.in_flight.clone(),
            id: document_id.to_string(),
        })
    }
}

/// Releases an in-flight claim on drop (covers the success, error and panic paths).
pub struct InFlightGuard {
    set: Arc<Mutex<HashSet<String>>>,
    id: String,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        if let Ok(mut set) = self.set.lock() {
            set.remove(&self.id);
        }
    }
}
