use sqlx::PgPool;
use tracing::{info, warn};

pub fn instance_app_name() -> String {
    format!("nebuia-embs-{}-{}", hostname(), std::process::id())
}

/// Short, stable host identifier so each node only ever cleans up its OWN stale
/// connections (from previous crashed PIDs), never another node's healthy ones.
/// Capped so the full application_name stays under Postgres' 63-byte limit —
/// otherwise truncation would desync our `<>` self-check and we could kill our
/// own connections.
pub fn hostname() -> String {
    let h = std::fs::read_to_string("/proc/sys/kernel/hostname")
        .or_else(|_| std::fs::read_to_string("/etc/hostname"))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    h.chars().take(32).collect()
}

/// LIKE pattern matching only this host's instances.
fn host_prefix() -> String {
    format!("nebuia-embs-{}-%", hostname())
}

async fn terminate_stale(pool: &PgPool, older_than_mins: i32) -> anyhow::Result<i64> {
    let our_app_name = instance_app_name();
    let (killed,): (i64,) = sqlx::query_as(
        "SELECT count(*) FROM (
            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE usename    = current_user
              AND datname    = current_database()
              AND state      = 'idle'
              AND pid       <> pg_backend_pid()
              AND application_name LIKE $3
              AND application_name <> $1
              AND state_change < now() - ($2 || ' minutes')::interval
        ) t",
    )
    .bind(&our_app_name)
    .bind(older_than_mins)
    .bind(host_prefix())
    .fetch_one(pool)
    .await?;
    Ok(killed)
}

pub async fn cleanup_on_startup(pool: &PgPool) {
    match terminate_stale(pool, 2).await {
        Ok(0) => info!("DB janitor: no stale connections found at startup"),
        Ok(n) => info!("DB janitor: terminated {} stale idle connections at startup", n),
        Err(e) => warn!("DB janitor: startup cleanup failed: {}", e),
    }
}

pub fn spawn_janitor(pool: PgPool, interval_secs: u64, older_than_mins: i32) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
        ticker.tick().await;
        loop {
            ticker.tick().await;
            match terminate_stale(&pool, older_than_mins).await {
                Ok(0) => {}
                Ok(n) => info!("DB janitor: terminated {} stale idle connections", n),
                Err(e) => warn!("DB janitor: cleanup failed: {}", e),
            }
        }
    });
}
