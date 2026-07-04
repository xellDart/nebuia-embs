mod config;
mod handlers;
mod models;
mod repositories;
mod routes;
mod services;
mod state;

use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

use config::AppConfig;
use services::cache_service::CacheService;
use services::embedding_service::EmbeddingService;
use state::AppState;

#[derive(Parser, Debug)]
#[command(name = "nebuia-embs", about = "Document embedding service powered by ColQwen3")]
struct Args {
    /// Path to .env file
    #[arg(long, default_value = ".env")]
    env_file: String,

    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(clap::Subcommand, Debug)]
enum Cmd {
    /// Regression harness: record or compare golden search results for a document
    Regression {
        /// Document id (default: most recent complete doc with 10-100 pages)
        #[arg(long)]
        doc: Option<String>,
        /// Write the baseline instead of comparing against it
        #[arg(long)]
        record: bool,
        /// Comma-separated queries (default: built-in set)
        #[arg(long, value_delimiter = ',')]
        queries: Vec<String>,
        /// Baseline directory
        #[arg(long, default_value = "regression")]
        dir: String,
        /// Max allowed |Δ| per embedding value when comparing (0 = bit-exact)
        #[arg(long, default_value_t = 0.0)]
        tol: f32,
        /// Images per encode call (1 = production default path)
        #[arg(long, default_value_t = 1)]
        image_batch: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    // Load environment
    if std::path::Path::new(&args.env_file).exists() {
        dotenvy::from_filename(&args.env_file)?;
    }
    let config = AppConfig::from_env();

    if let Some(Cmd::Regression {
        doc,
        record,
        queries,
        dir,
        tol,
        image_batch,
    }) = args.cmd
    {
        return services::regression::run(config, doc, record, queries, dir, tol, image_batch)
            .await;
    }

    info!("Connecting to database...");

    // Tag connections with our PID — safe for multi-node deployments
    let db_url = format!(
        "{}{}application_name={}",
        config.database_url,
        if config.database_url.contains('?') { "&" } else { "?" },
        services::db_janitor::instance_app_name(),
    );
    let db_pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(5)
        .acquire_timeout(std::time::Duration::from_secs(60))
        // 1 min — aggressively below any NAT/firewall idle-TCP-drop timeout.
        // Some cloud providers drop idle TCP in 2-3 min; 60 s guarantees
        // sqlx closes connections before the NAT forgets about them.
        .idle_timeout(std::time::Duration::from_secs(60))
        .after_connect(|conn, _| Box::pin(async move {
            // Server-side guard: kills queries that hang due to DB slowness.
            // Does NOT help with dead TCP connections (client-side issue handled
            // by idle_timeout above), but catches legitimate slow queries.
            sqlx::query("SET statement_timeout = '15s'")
                .execute(&mut *conn)
                .await?;
            Ok(())
        }))
        .connect(&db_url)
        .await?;

    models::database::create_tables(&db_pool).await?;
    info!("Database connected, tables ready");

    // Kill zombie connections from previous instances, then watch in background
    services::db_janitor::cleanup_on_startup(&db_pool).await;
    services::db_janitor::spawn_janitor(db_pool.clone(), 300, 5);

    info!("Connecting to MinIO at {}...", config.minio_endpoint);
    let storage = repositories::storage_repository::StorageRepository::new(&config).await?;
    info!("MinIO connected");

    info!("Loading embedding model from: {}", config.model_path);
    let use_cpu = std::env::var("MODEL_DEVICE")
        .map(|d| d == "cpu")
        .unwrap_or(false);
    // With multiple images per call, crane decodes/resizes them in parallel on
    // the CPU pool while keeping vision forwards strictly per-image, so results
    // are bit-identical to one-at-a-time encoding (regression-verified).
    let image_batch = if config.batch_image_encode {
        config.batch_image_encode_size.max(1)
    } else {
        1
    };
    let embedding = EmbeddingService::spawn(
        &config.model_path,
        use_cpu,
        config.use_bf16,
        config.model_dims,
        config.model_queue_capacity,
        image_batch,
        config.search_score_concurrency,
        config.gpu_cache_mb,
        config.gpu_cache_idle_secs,
    )?;

    let cache = CacheService::new(config.cache_max_mb, config.cache_expiry_hours);

    let state = Arc::new(AppState {
        config: config.clone(),
        db_pool,
        storage,
        embedding,
        cache,
        process_semaphore: Arc::new(tokio::sync::Semaphore::new(4)),
        in_flight: Arc::new(std::sync::Mutex::new(std::collections::HashSet::new())),
    });

    // Start NATS consumer if enabled
    if config.nats_enabled {
        let nats_state = state.clone();
        tokio::spawn(async move {
            match services::nats_consumer::run_nats_consumer(nats_state).await {
                Ok(()) => info!("NATS consumer stopped"),
                Err(e) => tracing::warn!("NATS consumer failed: {} (running HTTP-only mode)", e),
            }
        });
    } else {
        info!("NATS consumer disabled");
    }

    // Build router with CORS
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = routes::build_router(state.clone()).layer(cors);

    let addr = format!("{}:{}", config.host, config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    let local_addr = listener.local_addr()?;

    let sep = "=".repeat(56);
    println!("\n  {sep}");
    println!("  nebuia-embs v{} ready", env!("CARGO_PKG_VERSION"));
    println!("  {sep}");
    println!("  Model  : {}", config.model_path);
    println!("  Listen : http://{local_addr}");
    println!("  NATS   : {}", if config.nats_enabled { &config.nats_url } else { "disabled" });
    println!("  {sep}");
    println!("  POST   http://{local_addr}/process-pdf");
    println!("  GET    http://{local_addr}/simple/search/{{id}}?query=...");
    println!("  DELETE http://{local_addr}/document/{{id}}");
    println!("  GET    http://{local_addr}/health");
    println!("  {sep}\n");

    axum::serve(listener, app).await?;

    Ok(())
}
