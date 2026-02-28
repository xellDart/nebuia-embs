use std::sync::Arc;

use anyhow::Result;
use tracing::{error, info, warn};

use crate::state::AppState;

pub async fn run_nats_consumer(state: Arc<AppState>) -> Result<()> {
    let nats_url = &state.config.nats_url;
    info!("Connecting to NATS at {}...", nats_url);

    // Parse credentials from URL: nats://user:pass@host:port → ConnectOptions
    let client = if nats_url.contains('@') {
        let without_scheme = nats_url
            .strip_prefix("nats://")
            .unwrap_or(nats_url);
        let (creds, host) = without_scheme
            .split_once('@')
            .unwrap_or(("", without_scheme));
        let (user, pass) = creds
            .split_once(':')
            .unwrap_or((creds, ""));
        info!("NATS authenticating as user '{}'", user);
        async_nats::ConnectOptions::new()
            .user_and_password(user.to_string(), pass.to_string())
            .connect(host)
            .await?
    } else {
        let host = nats_url
            .strip_prefix("nats://")
            .unwrap_or(nats_url);
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

    let consumer = stream
        .get_or_create_consumer(
            "embeddings_workers",
            async_nats::jetstream::consumer::pull::Config {
                durable_name: Some("embeddings_workers".to_string()),
                filter_subject: "embeddings.process".to_string(),
                ack_wait: std::time::Duration::from_secs(300),
                max_deliver: 5,
                ..Default::default()
            },
        )
        .await?;

    info!("NATS pull consumer ready, listening for embeddings jobs...");

    // Pull messages in a loop
    loop {
        let mut messages = match consumer.fetch().max_messages(1).messages().await {
            Ok(msgs) => msgs,
            Err(e) => {
                warn!("NATS fetch error: {}, retrying...", e);
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                continue;
            }
        };

        use futures_util::StreamExt;
        while let Some(msg_result) = messages.next().await {
            let msg = match msg_result {
                Ok(m) => m,
                Err(e) => {
                    warn!("NATS message error: {}", e);
                    continue;
                }
            };

            let payload: serde_json::Value = match serde_json::from_slice(&msg.payload) {
                Ok(v) => v,
                Err(e) => {
                    error!("Invalid JSON in NATS message: {}", e);
                    let _ = msg.ack().await;
                    continue;
                }
            };

            let document_id = match payload.get("document_id").and_then(|v| v.as_str()) {
                Some(id) => id.to_string(),
                None => {
                    error!("NATS message missing document_id");
                    let _ = msg.ack().await;
                    continue;
                }
            };

            info!("NATS: processing document {}", document_id);

            match crate::services::pdf_service::process_document_embeddings(
                &document_id,
                &state.db_pool,
                &state.storage,
                &state.embedding,
                &state.cache,
                state.config.model_batch_size,
            )
            .await
            {
                Ok(()) => {
                    info!("NATS: completed document {}", document_id);
                    // Publish completion
                    let completion = serde_json::json!({
                        "document_id": document_id,
                        "status": "complete",
                    });
                    let _ = jetstream
                        .publish(
                            format!("embeddings.completed.complete"),
                            completion.to_string().into(),
                        )
                        .await;
                    let _ = msg.ack().await;
                }
                Err(e) => {
                    error!("NATS: failed document {}: {}", document_id, e);
                    let completion = serde_json::json!({
                        "document_id": document_id,
                        "status": "error",
                    });
                    let _ = jetstream
                        .publish(
                            format!("embeddings.completed.error"),
                            completion.to_string().into(),
                        )
                        .await;
                    // Negative ack (will be retried up to max_deliver)
                    let _ = msg.ack().await;
                }
            }
        }
    }
}
