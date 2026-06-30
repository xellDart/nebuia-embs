use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use async_nats::jetstream::AckKind;
use futures_util::StreamExt;
use tracing::{error, info, warn};

use crate::state::AppState;

/// How often to tell JetStream we're still working on a message, so it doesn't
/// redeliver while a long encode is in flight. Must be well below `ack_wait`.
const PROGRESS_INTERVAL: Duration = Duration::from_secs(60);
const ACK_WAIT: Duration = Duration::from_secs(300);
const MAX_DELIVER: i64 = 5;

pub async fn run_nats_consumer(state: Arc<AppState>) -> Result<()> {
    let nats_url = &state.config.nats_url;
    info!("Connecting to NATS at {}...", nats_url);

    // Parse credentials from URL: nats://user:pass@host:port → ConnectOptions
    let client = if nats_url.contains('@') {
        let without_scheme = nats_url.strip_prefix("nats://").unwrap_or(nats_url);
        let (creds, host) = without_scheme
            .split_once('@')
            .unwrap_or(("", without_scheme));
        let (user, pass) = creds.split_once(':').unwrap_or((creds, ""));
        info!("NATS authenticating as user '{}'", user);
        async_nats::ConnectOptions::new()
            .user_and_password(user.to_string(), pass.to_string())
            .connect(host)
            .await?
    } else {
        let host = nats_url.strip_prefix("nats://").unwrap_or(nats_url);
        async_nats::connect(host).await?
    };

    let jetstream = async_nats::jetstream::new(client.clone());

    info!("NATS connected, creating pull consumer...");

    // Get or create the consumer on the EMBEDDINGS stream
    let stream = jetstream
        .get_or_create_stream(async_nats::jetstream::stream::Config {
            name: "EMBEDDINGS".to_string(),
            subjects: vec!["embeddings.>".to_string()],
            ..Default::default()
        })
        .await?;

    // A durable consumer that already exists is NOT reconfigured by
    // get_or_create_consumer, so changes to ack_wait/max_deliver in code would
    // silently not apply. Delete it first (ignoring "not found") and recreate so
    // the config below is always authoritative. Unacked messages are redelivered.
    match stream.delete_consumer("embeddings_workers").await {
        Ok(_) => info!("NATS: removed existing consumer to apply current config"),
        Err(e) => info!("NATS: no existing consumer to remove ({})", e),
    }

    let consumer = stream
        .get_or_create_consumer(
            "embeddings_workers",
            async_nats::jetstream::consumer::pull::Config {
                durable_name: Some("embeddings_workers".to_string()),
                filter_subject: "embeddings.process".to_string(),
                ack_wait: ACK_WAIT,
                max_deliver: MAX_DELIVER,
                ..Default::default()
            },
        )
        .await?;

    info!("NATS pull consumer ready, listening for embeddings jobs...");

    // Pull-and-dispatch loop. We acquire a processing permit BEFORE fetching the
    // next message, so we never pull more work than we can run concurrently and
    // each message is handed straight to a task. This lets up to `process_semaphore`
    // documents process in parallel instead of one-at-a-time.
    loop {
        let permit = match state.process_semaphore.clone().acquire_owned().await {
            Ok(p) => p,
            Err(_) => {
                warn!("NATS: process semaphore closed, stopping consumer");
                return Ok(());
            }
        };

        let mut messages = match consumer.fetch().max_messages(1).messages().await {
            Ok(msgs) => msgs,
            Err(e) => {
                warn!("NATS fetch error: {}, retrying...", e);
                drop(permit);
                tokio::time::sleep(Duration::from_secs(1)).await;
                continue;
            }
        };

        let msg = match messages.next().await {
            Some(Ok(m)) => m,
            Some(Err(e)) => {
                warn!("NATS message error: {}", e);
                drop(permit);
                continue;
            }
            None => {
                // No work available right now; release the slot and back off briefly.
                drop(permit);
                tokio::time::sleep(Duration::from_millis(500)).await;
                continue;
            }
        };

        let payload: serde_json::Value = match serde_json::from_slice(&msg.payload) {
            Ok(v) => v,
            Err(e) => {
                error!("Invalid JSON in NATS message: {}", e);
                let _ = msg.ack().await;
                drop(permit);
                continue;
            }
        };

        let document_id = match payload.get("document_id").and_then(|v| v.as_str()) {
            Some(id) => id.to_string(),
            None => {
                error!("NATS message missing document_id");
                let _ = msg.ack().await;
                drop(permit);
                continue;
            }
        };

        let task_state = state.clone();
        let task_js = jetstream.clone();
        tokio::spawn(async move {
            // Permit is released when this task ends, freeing a slot for the loop.
            let _permit = permit;
            process_message(msg, document_id, task_state, task_js).await;
        });
    }
}

/// Process a single document, sending periodic progress acks so JetStream does
/// not redeliver while a long encode is in flight.
async fn process_message(
    msg: async_nats::jetstream::Message,
    document_id: String,
    state: Arc<AppState>,
    jetstream: async_nats::jetstream::Context,
) {
    info!("NATS: processing document {}", document_id);

    // Skip if another task (HTTP endpoint or a duplicate delivery) is already
    // processing this document. NAK with a short delay so the job is retried
    // once the in-flight run finishes, rather than dropped.
    let _claim = match state.try_claim(&document_id) {
        Some(g) => g,
        None => {
            info!("NATS: document {} already in flight, NAK to retry", document_id);
            let _ = msg
                .ack_with(AckKind::Nak(Some(Duration::from_secs(5))))
                .await;
            return;
        }
    };

    let processing = crate::services::pdf_service::process_document_embeddings(
        &document_id,
        &state.db_pool,
        &state.storage,
        &state.embedding,
        &state.cache,
        state.config.model_batch_size,
    );
    tokio::pin!(processing);

    // Drive the work while heartbeating progress every PROGRESS_INTERVAL.
    let result = loop {
        tokio::select! {
            r = &mut processing => break r,
            _ = tokio::time::sleep(PROGRESS_INTERVAL) => {
                if let Err(e) = msg.ack_with(AckKind::Progress).await {
                    warn!("NATS: progress ack failed for {}: {}", document_id, e);
                }
            }
        }
    };

    match result {
        Ok(()) => {
            info!("NATS: completed document {}", document_id);
            let completion = serde_json::json!({
                "document_id": document_id,
                "status": "complete",
            });
            let _ = jetstream
                .publish(
                    "embeddings.completed.complete".to_string(),
                    completion.to_string().into(),
                )
                .await;
            let _ = msg.ack().await;
        }
        Err(e) => {
            let err_str = e.to_string();
            let delivered = msg.info().map(|i| i.delivered).unwrap_or(1);
            if delivered >= MAX_DELIVER {
                error!(
                    "NATS: permanently failed document {} after {} attempts: {}",
                    document_id, delivered, err_str
                );
                let completion = serde_json::json!({
                    "document_id": document_id,
                    "status": "error",
                });
                let _ = jetstream
                    .publish(
                        "embeddings.completed.error".to_string(),
                        completion.to_string().into(),
                    )
                    .await;
                let _ = msg.ack().await;
            } else {
                error!(
                    "NATS: failed document {} (attempt {}), will retry: {}",
                    document_id, delivered, err_str
                );
                // Always back off a little before redelivery to avoid hot-looping.
                let delay = if err_str.contains("pool timed out")
                    || err_str.contains("connection")
                {
                    Duration::from_secs(3)
                } else {
                    Duration::from_secs(2)
                };
                let _ = msg.ack_with(AckKind::Nak(Some(delay))).await;
            }
        }
    }
}
