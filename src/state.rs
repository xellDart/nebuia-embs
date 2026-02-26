use sqlx::PgPool;

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
}
