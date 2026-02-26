use anyhow::Result;
use tracing::{error, info, warn};

use crate::repositories::{document_repository, storage_repository::StorageRepository};
use crate::services::cache_service::CacheService;
use crate::services::embedding_service::{
    deserialize_embeddings, serialize_embeddings, EmbeddingService, PageEmbedding,
};

/// Process embeddings for a document: download images → encode → compress → upload → save DB.
pub async fn process_document_embeddings(
    document_id: &str,
    pool: &sqlx::PgPool,
    storage: &StorageRepository,
    embedding: &EmbeddingService,
    cache: &CacheService,
    batch_size: usize,
) -> Result<()> {
    // Verify document exists
    let doc = document_repository::get_document(pool, document_id).await?;
    if doc.is_none() {
        warn!("Document {} not found, skipping", document_id);
        return Ok(());
    }

    // Update status
    document_repository::update_document_status(pool, document_id, "processing").await?;

    // Run the actual pipeline, catching errors to reset status
    match do_process(document_id, pool, storage, embedding, cache, batch_size).await {
        Ok(()) => {
            document_repository::update_document_status(pool, document_id, "complete").await?;
            info!("Embeddings complete for document {}", document_id);
            Ok(())
        }
        Err(e) => {
            error!("Processing failed for {}: {}", document_id, e);
            let _ = document_repository::update_document_status(pool, document_id, "error").await;
            Err(e)
        }
    }
}

async fn do_process(
    document_id: &str,
    pool: &sqlx::PgPool,
    storage: &StorageRepository,
    embedding: &EmbeddingService,
    cache: &CacheService,
    batch_size: usize,
) -> Result<()> {
    // List page images from storage
    let prefix = format!("{}_page_", document_id);
    let mut page_objects: Vec<String> = storage
        .list_objects(&prefix)
        .await?
        .into_iter()
        .filter(|k| k.ends_with(".jpeg") || k.ends_with(".jpg") || k.ends_with(".png"))
        .collect();

    if page_objects.is_empty() {
        anyhow::bail!("No images found for document {}", document_id);
    }

    // Sort by page number
    page_objects.sort_by_key(|name| extract_page_number(name));
    info!(
        "Found {} images for document {}",
        page_objects.len(),
        document_id
    );

    // Download and process in batches
    let mut all_embeddings: Vec<PageEmbedding> = Vec::new();

    for (batch_idx, chunk) in page_objects.chunks(batch_size).enumerate() {
        info!(
            "Processing batch {} ({} images) for {}",
            batch_idx + 1,
            chunk.len(),
            document_id
        );

        // Download images and save to temp files
        let mut temp_paths = Vec::new();
        for object_name in chunk {
            let image_bytes = storage.get_image(object_name).await?;
            let tmp = std::env::temp_dir().join(format!(
                "nebuia_{}_{}.jpg",
                document_id,
                extract_page_number(object_name)
            ));
            std::fs::write(&tmp, &image_bytes)?;
            temp_paths.push(tmp);
        }

        // Encode batch
        let batch_embeddings = embedding.encode_images(temp_paths.clone()).await?;
        all_embeddings.extend(batch_embeddings);

        // Cleanup temp files
        for tmp in &temp_paths {
            let _ = std::fs::remove_file(tmp);
        }
    }

    // Verify document still exists
    if document_repository::get_document(pool, document_id)
        .await?
        .is_none()
    {
        warn!(
            "Document {} was deleted during processing, skipping save",
            document_id
        );
        return Ok(());
    }

    // Save page metadata to DB
    for object_name in page_objects.iter() {
        let page_number = extract_page_number(object_name);
        document_repository::save_page(pool, document_id, page_number, object_name).await?;
    }

    // Serialize and upload embeddings (bf16 format)
    let raw = serialize_embeddings(&all_embeddings);
    storage.upload_embeddings(document_id, &raw).await?;

    // Warm cache
    cache.put(document_id, all_embeddings).await;

    Ok(())
}

/// Search document pages by query.
pub async fn search_document(
    document_id: &str,
    query: &str,
    k: usize,
    continues: bool,
    pool: &sqlx::PgPool,
    storage: &StorageRepository,
    embedding: &EmbeddingService,
    cache: &CacheService,
) -> Result<Vec<String>> {
    // Get document + pages
    let (doc, pages) = document_repository::get_document_with_pages(pool, document_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Document {} not found", document_id))?;

    if doc.status != "complete" {
        anyhow::bail!("Document {} is not ready (status: {})", document_id, doc.status);
    }

    if pages.is_empty() {
        anyhow::bail!("No pages found for document {}", document_id);
    }

    // Get stored embeddings (cache or storage)
    let page_embs = match cache.get(document_id).await {
        Some(cached) => {
            info!("Cache hit for document {}", document_id);
            cached.as_ref().clone()
        }
        None => {
            info!("Cache miss for {}, downloading from storage", document_id);
            let raw = storage.get_embeddings(document_id).await?;
            let embs = deserialize_embeddings(&raw)?;
            cache.put(document_id, embs.clone()).await;
            embs
        }
    };

    // Encode query
    let query_embs = embedding.encode_query(query.to_string()).await?;

    // Score
    let scores = embedding.score(query_embs, page_embs).await?;

    // Build results
    let mut indexed: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();

    if continues {
        // Return k consecutive pages starting from the best match
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let best_page = pages
            .get(indexed[0].0)
            .map(|p| p.page_number)
            .unwrap_or(1);

        let mut result = Vec::new();
        for i in 0..k as i32 {
            let target = best_page + i;
            if let Some(page) = pages.iter().find(|p| p.page_number == target) {
                result.push(page.image_path.clone());
            } else {
                break;
            }
        }
        Ok(result)
    } else {
        // Top-k by score
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let result: Vec<String> = indexed
            .iter()
            .take(k)
            .filter_map(|(idx, _)| pages.get(*idx).map(|p| p.image_path.clone()))
            .collect();
        Ok(result)
    }
}

/// Delete a document and all its data.
pub async fn delete_document(
    document_id: &str,
    pool: &sqlx::PgPool,
    storage: &StorageRepository,
    cache: &CacheService,
) -> Result<serde_json::Value> {
    let (_, pages) = document_repository::get_document_with_pages(pool, document_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Document {} not found", document_id))?;

    let image_paths: Vec<String> = pages.iter().map(|p| p.image_path.clone()).collect();
    let num_images = image_paths.len();

    // Delete from storage
    if !image_paths.is_empty() {
        storage.delete_objects(&image_paths).await?;
    }
    let emb_key = format!("{}_embeddings.zst", document_id);
    storage.delete_objects(&[emb_key]).await?;

    // Clear cache
    cache.remove(document_id).await;

    // Delete from DB
    document_repository::delete_document(pool, document_id).await?;

    Ok(serde_json::json!({
        "message": format!("Document {} completely deleted", document_id),
        "deleted_images": num_images,
        "deleted_embeddings": true
    }))
}

fn extract_page_number(name: &str) -> i32 {
    // Format: "{document_id}_page_{N}.jpeg"
    name.rsplit("_page_")
        .next()
        .and_then(|s| s.split('.').next())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}
