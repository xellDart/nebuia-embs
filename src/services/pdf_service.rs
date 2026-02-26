use anyhow::Result;
use futures_util::stream::{self, StreamExt};
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

    // Download and process in batches (parallel downloads, no temp files)
    let mut all_embeddings: Vec<PageEmbedding> = Vec::new();
    let chunks: Vec<&[String]> = page_objects.chunks(batch_size).collect();

    // Pre-fetch first batch
    let mut next_download: Option<tokio::task::JoinHandle<Result<Vec<Vec<u8>>>>> = None;
    if let Some(first_chunk) = chunks.first() {
        let chunk_names: Vec<String> = first_chunk.to_vec();
        let st = storage.clone();
        next_download = Some(tokio::spawn(download_batch(st, chunk_names)));
    }

    for (batch_idx, chunk) in chunks.iter().enumerate() {
        info!(
            "Processing batch {} ({} images) for {}",
            batch_idx + 1,
            chunk.len(),
            document_id
        );

        // Await current batch download
        let images_bytes = next_download
            .take()
            .ok_or_else(|| anyhow::anyhow!("No download task"))?
            .await??;

        // Pre-fetch next batch while encoding
        if batch_idx + 1 < chunks.len() {
            let chunk_names: Vec<String> = chunks[batch_idx + 1].to_vec();
            let st = storage.clone();
            next_download = Some(tokio::spawn(download_batch(st, chunk_names)));
        }

        // Encode from bytes (no temp files)
        let batch_embeddings = embedding.encode_images_from_bytes(images_bytes).await?;
        all_embeddings.extend(batch_embeddings);
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

    // Save page metadata to DB (batch insert)
    let page_numbers: Vec<i32> = page_objects.iter().map(|n| extract_page_number(n)).collect();
    let image_paths: Vec<&str> = page_objects.iter().map(|s| s.as_str()).collect();
    document_repository::save_pages_batch(pool, document_id, &page_numbers, &image_paths).await?;

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
        // Top-k by score, then sort consecutive runs by page number.
        // Example: scores give [p26, p25, p52, p7] → detect 26,25 are consecutive
        // → reorder to [p25, p26, p52, p7] so the reader gets natural page order.
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<(i32, String)> = indexed
            .iter()
            .take(k)
            .filter_map(|(idx, _)| {
                pages.get(*idx).map(|p| (p.page_number, p.image_path.clone()))
            })
            .collect();

        let result = sort_consecutive_runs(top);
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

    let num_pages = pages.len();

    // Delete only embeddings from storage (images belong to the upload service)
    let emb_key = format!("{}_embeddings.zst", document_id);
    storage.delete_objects(&[emb_key]).await?;

    // Clear cache
    cache.remove(document_id).await;

    // Delete from DB (pages cascade-delete via FK)
    document_repository::delete_document(pool, document_id).await?;

    Ok(serde_json::json!({
        "message": format!("Document {} deleted", document_id),
        "deleted_pages": num_pages,
        "deleted_embeddings": true
    }))
}

/// Sort consecutive page runs by page number while preserving group rank.
/// Input ranked by score: [(p26,".."), (p25,".."), (p52,".."), (p7,"..")]
/// Output: [(p25,".."), (p26,".."), (p52,".."), (p7,"..")]
/// Only adjacent entries whose page numbers differ by exactly 1 are grouped.
fn sort_consecutive_runs(ranked: Vec<(i32, String)>) -> Vec<String> {
    if ranked.is_empty() {
        return Vec::new();
    }

    // Group adjacent entries that are consecutive (differ by ±1)
    let mut groups: Vec<Vec<(i32, String)>> = Vec::new();
    let mut current_group = vec![ranked[0].clone()];

    for pair in ranked.iter().skip(1) {
        let prev_page = current_group.last().unwrap().0;
        if (pair.0 - prev_page).abs() == 1 {
            current_group.push(pair.clone());
        } else {
            groups.push(current_group);
            current_group = vec![pair.clone()];
        }
    }
    groups.push(current_group);

    // Sort each group by page number (ascending), then flatten
    groups
        .iter_mut()
        .flat_map(|g| {
            g.sort_by_key(|(page, _)| *page);
            g.iter().map(|(_, path)| path.clone())
        })
        .collect()
}

/// Download a batch of images in parallel (4 concurrent).
async fn download_batch(storage: StorageRepository, names: Vec<String>) -> Result<Vec<Vec<u8>>> {
    let results: Vec<Result<Vec<u8>>> = stream::iter(names.into_iter().map(|name| {
        let st = storage.clone();
        async move { st.get_image(&name).await.map(|b| b.to_vec()) }
    }))
    .buffered(4)
    .collect()
    .await;

    results.into_iter().collect()
}

fn extract_page_number(name: &str) -> i32 {
    // Format: "{document_id}_page_{N}.jpeg"
    name.rsplit("_page_")
        .next()
        .and_then(|s| s.split('.').next())
        .and_then(|s| s.parse().ok())
        .unwrap_or(0)
}
