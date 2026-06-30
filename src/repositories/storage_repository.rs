use anyhow::{Context, Result};
use aws_sdk_s3::Client;
use aws_sdk_s3::config::retry::RetryConfig;
use aws_sdk_s3::config::{Credentials, Region, StalledStreamProtectionConfig};
use aws_sdk_s3::primitives::ByteStream;
use std::time::Duration;
use bytes::Bytes;
use tracing::{info, warn};

use crate::config::AppConfig;

/// Hard ceiling per download attempt. The SDK's operation timeout does not cover
/// response-body streaming, so a stalled transfer over a lossy/remote link can
/// hang for minutes. Bound each attempt so a retry kicks in quickly instead.
const ATTEMPT_TIMEOUT: Duration = Duration::from_secs(45);

#[derive(Clone)]
pub struct StorageRepository {
    client: Client,
    bucket: String,
}

impl StorageRepository {
    pub async fn new(config: &AppConfig) -> Result<Self> {
        let creds = Credentials::new(
            &config.minio_access_key,
            &config.minio_secret_key,
            None,
            None,
            "env",
        );

        let endpoint = if config.minio_endpoint.starts_with("http") {
            config.minio_endpoint.clone()
        } else {
            format!("https://{}", config.minio_endpoint)
        };

        let s3_config = aws_sdk_s3::Config::builder()
            .behavior_version_latest()
            .region(Region::new("us-east-1"))
            .endpoint_url(&endpoint)
            .credentials_provider(creds)
            .force_path_style(true)
            // Retry transient failures on a fresh connection. MinIO keep-alive
            // connections that go idle get dropped by NAT/firewall; reusing one
            // surfaces as "connection should not be re-used" → the SDK poisons it
            // and a retry grabs a new connection. Covers list/put/delete/head.
            .retry_config(RetryConfig::standard().with_max_attempts(6))
            // The SDK aborts a transfer if throughput drops below the minimum for
            // the grace period (5s by default). With several concurrent downloads
            // of large images over a remote MinIO, bandwidth contention can stall
            // individual streams past that window, surfacing as "Failed to read
            // image body". A 30s grace period tolerates the slowdown while still
            // protecting against genuinely dead connections.
            .stalled_stream_protection(
                StalledStreamProtectionConfig::enabled()
                    .grace_period(Duration::from_secs(30))
                    .build(),
            )
            .build();

        let client = Client::from_conf(s3_config);

        Ok(Self {
            client,
            bucket: config.minio_bucket.clone(),
        })
    }

    pub async fn get_image(&self, object_name: &str) -> Result<Bytes> {
        // Retry each image independently. A single image failing its body read
        // (transient IO, connection reset, throughput stall) would otherwise
        // fail the whole batch and force NATS to re-download the entire document.
        const MAX_ATTEMPTS: u32 = 4;
        let mut last_err = None;
        for attempt in 1..=MAX_ATTEMPTS {
            let res = match tokio::time::timeout(ATTEMPT_TIMEOUT, self.try_get_image(object_name))
                .await
            {
                Ok(r) => r,
                Err(_) => Err(anyhow::anyhow!("timed out after {}s", ATTEMPT_TIMEOUT.as_secs())),
            };
            match res {
                Ok(bytes) => return Ok(bytes),
                Err(e) => {
                    warn!(
                        "get_image {} attempt {}/{} failed: {}",
                        object_name, attempt, MAX_ATTEMPTS, e
                    );
                    last_err = Some(e);
                    if attempt < MAX_ATTEMPTS {
                        tokio::time::sleep(Duration::from_millis(300 * attempt as u64)).await;
                    }
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("get_image failed: {}", object_name)))
    }

    async fn try_get_image(&self, object_name: &str) -> Result<Bytes> {
        let resp = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(object_name)
            .send()
            .await
            .with_context(|| format!("Failed to get image: {}", object_name))?;

        let bytes = resp
            .body
            .collect()
            .await
            .context("Failed to read image body")?
            .into_bytes();

        Ok(bytes)
    }

    pub async fn list_objects(&self, prefix: &str) -> Result<Vec<String>> {
        let mut keys = Vec::new();
        let mut continuation_token: Option<String> = None;

        loop {
            let mut req = self
                .client
                .list_objects_v2()
                .bucket(&self.bucket)
                .prefix(prefix);

            if let Some(token) = continuation_token.take() {
                req = req.continuation_token(token);
            }

            let resp = req.send().await.context("Failed to list objects")?;

            for obj in resp.contents() {
                if let Some(key) = obj.key() {
                    keys.push(key.to_string());
                }
            }

            if resp.is_truncated() == Some(true) {
                continuation_token = resp.next_continuation_token().map(|s| s.to_string());
            } else {
                break;
            }
        }

        Ok(keys)
    }

    pub async fn upload_embeddings(
        &self,
        document_id: &str,
        data: &[u8],
    ) -> Result<String> {
        let object_name = format!("{}_embeddings.zst", document_id);

        let compressed = zstd::encode_all(data, 3)
            .context("Failed to compress embeddings")?;

        let original_size = data.len();
        let compressed_size = compressed.len();

        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(&object_name)
            .content_type("application/octet-stream")
            .body(ByteStream::from(compressed))
            .send()
            .await
            .with_context(|| format!("Failed to upload embeddings for {}", document_id))?;

        info!(
            "Uploaded embeddings for {}: {} -> {} ({:.1}x)",
            document_id,
            format_size(original_size),
            format_size(compressed_size),
            original_size as f64 / compressed_size as f64,
        );

        Ok(object_name)
    }

    pub async fn get_embeddings(&self, document_id: &str) -> Result<Vec<u8>> {
        // Retry the whole fetch: a stalled/dropped body stream cannot be retried
        // by the SDK once streaming has started, so we re-issue the request.
        const MAX_ATTEMPTS: u32 = 4;
        let mut last_err = None;
        for attempt in 1..=MAX_ATTEMPTS {
            let res = match tokio::time::timeout(
                ATTEMPT_TIMEOUT,
                self.try_get_embeddings(document_id),
            )
            .await
            {
                Ok(r) => r,
                Err(_) => Err(anyhow::anyhow!("timed out after {}s", ATTEMPT_TIMEOUT.as_secs())),
            };
            match res {
                Ok(data) => return Ok(data),
                Err(e) => {
                    warn!(
                        "get_embeddings {} attempt {}/{} failed: {}",
                        document_id, attempt, MAX_ATTEMPTS, e
                    );
                    last_err = Some(e);
                    if attempt < MAX_ATTEMPTS {
                        tokio::time::sleep(Duration::from_millis(300 * attempt as u64)).await;
                    }
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("get_embeddings failed: {}", document_id)))
    }

    async fn try_get_embeddings(&self, document_id: &str) -> Result<Vec<u8>> {
        let object_name = format!("{}_embeddings.zst", document_id);

        let resp = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(&object_name)
            .send()
            .await
            .with_context(|| format!("Failed to get embeddings for {}", document_id))?;

        let compressed = resp
            .body
            .collect()
            .await
            .context("Failed to read embeddings body")?
            .into_bytes();

        let decompressed = zstd::decode_all(compressed.as_ref())
            .context("Failed to decompress embeddings")?;

        Ok(decompressed)
    }

    pub async fn delete_objects(&self, keys: &[String]) -> Result<()> {
        for key in keys {
            match self
                .client
                .delete_object()
                .bucket(&self.bucket)
                .key(key)
                .send()
                .await
            {
                Ok(_) => info!("Deleted: {}", key),
                Err(e) => warn!("Failed to delete {}: {}", key, e),
            }
        }
        Ok(())
    }

    pub async fn health_check(&self) -> bool {
        self.client
            .head_bucket()
            .bucket(&self.bucket)
            .send()
            .await
            .is_ok()
    }
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1 << 20 {
        format!("{:.1} MB", bytes as f64 / (1 << 20) as f64)
    } else if bytes >= 1 << 10 {
        format!("{:.1} KB", bytes as f64 / (1 << 10) as f64)
    } else {
        format!("{} B", bytes)
    }
}
