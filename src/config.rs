use std::env;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub database_url: String,
    pub minio_endpoint: String,
    pub minio_access_key: String,
    pub minio_secret_key: String,
    pub minio_bucket: String,
    pub model_path: String,
    pub model_batch_size: usize,
    pub model_dims: Option<usize>,
    pub use_bf16: bool,
    pub cache_max_size: u64,
    pub cache_expiry_hours: u64,
    pub nats_url: String,
    pub nats_enabled: bool,
    pub host: String,
    pub port: u16,
    pub max_retries: usize,
    pub retry_delay_secs: u64,
}

impl AppConfig {
    pub fn from_env() -> Self {
        Self {
            database_url: get("DATABASE_URL", None),
            minio_endpoint: get("MINIO_ENDPOINT", None),
            minio_access_key: get("MINIO_ACCESS_KEY", None),
            minio_secret_key: get("MINIO_SECRET_KEY", None),
            minio_bucket: get("MINIO_BUCKET", None),
            model_path: get("MODEL_PATH", Some("ops-colqwen3-FP8")),
            model_batch_size: get("MODEL_BATCH_SIZE", Some("9")).parse().unwrap_or(9),
            model_dims: std::env::var("MODEL_DIMS").ok().and_then(|v| v.parse().ok()),
            use_bf16: get("MODEL_DTYPE", Some("bfloat16")) == "bfloat16",
            cache_max_size: get("CACHE_MAX_SIZE", Some("10")).parse().unwrap_or(10),
            cache_expiry_hours: get("CACHE_EXPIRY_HOURS", Some("24")).parse().unwrap_or(24),
            nats_url: get("NATS_URL", Some("nats://localhost:4222")),
            nats_enabled: get("NATS_ENABLED", Some("true")) == "true",
            host: get("HOST", Some("0.0.0.0")),
            port: get("PORT", Some("8000")).parse().unwrap_or(8000),
            max_retries: get("MAX_RETRIES", Some("5")).parse().unwrap_or(5),
            retry_delay_secs: get("RETRY_DELAY", Some("3")).parse().unwrap_or(3),
        }
    }
}

fn get(name: &str, default: Option<&str>) -> String {
    match env::var(name) {
        Ok(v) if !v.is_empty() => v,
        _ => default
            .unwrap_or_else(|| panic!("Environment variable '{}' is required", name))
            .to_string(),
    }
}
