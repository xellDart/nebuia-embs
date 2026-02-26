use anyhow::{Context, Result};
use aws_sdk_s3::Client;
use aws_sdk_s3::config::{Credentials, Region};
use aws_sdk_s3::primitives::ByteStream;
use bytes::Bytes;
use tracing::{info, warn};

use crate::config::AppConfig;

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
            .build();

        let client = Client::from_conf(s3_config);

        Ok(Self {
            client,
            bucket: config.minio_bucket.clone(),
        })
    }

    pub async fn get_image(&self, object_name: &str) -> Result<Bytes> {
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

        let compressed = zstd::encode_all(data, 9)
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
