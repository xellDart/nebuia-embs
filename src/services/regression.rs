//! Regression harness: freeze a golden baseline (embeddings + search results)
//! for a real document, then re-run after model/pipeline changes and compare.
//!
//! Record:  nebuia-embs regression --record [--doc <id>]
//! Compare: nebuia-embs regression --doc <id>
//!
//! The baseline lives in <dir>/<doc_id>/ as the exact serialized embeddings
//! blob plus per-query scores and rankings. Compare re-runs the full pipeline
//! (download → encode → score) with the current build and reports:
//!   - embedding diff vs baseline (bit-exact / max abs diff / #values changed)
//!   - ranking changes per query and max score delta
//! Exit code 1 when top-5 rankings change or the embedding diff exceeds --tol.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::config::AppConfig;
use crate::repositories::storage_repository::StorageRepository;
use crate::services::embedding_service::{
    deserialize_embeddings, serialize_embeddings, EmbeddingService,
};
use crate::services::pdf_service::{download_batch, extract_page_number};

const DEFAULT_QUERIES: &[&str] = &[
    "fecha de firma",
    "monto total",
    "nombre del cliente",
    "dirección",
    "cláusula de terminación",
    "condiciones de pago",
    "tabla resumen",
    "firma del representante legal",
];

#[derive(Serialize, Deserialize)]
struct QueryResult {
    query: String,
    /// Page indices (0-based, in page-number order) sorted by score desc.
    ranking: Vec<usize>,
    scores: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct Baseline {
    document_id: String,
    model_path: String,
    model_dims: Option<usize>,
    pages: Vec<String>,
    queries: Vec<QueryResult>,
}

pub async fn run(
    config: AppConfig,
    doc: Option<String>,
    record: bool,
    queries: Vec<String>,
    dir: String,
    tol: f32,
) -> Result<()> {
    let storage = StorageRepository::new(&config).await?;

    let document_id = match doc {
        Some(d) => d,
        None => pick_document(&config).await?,
    };
    println!("Documento: {}", document_id);

    // ── Download all pages ───────────────────────────────────────
    let prefix = format!("{}_page_", document_id);
    let mut page_objects: Vec<String> = storage
        .list_objects(&prefix)
        .await?
        .into_iter()
        .filter(|k| k.ends_with(".jpeg") || k.ends_with(".jpg") || k.ends_with(".png"))
        .collect();
    if page_objects.is_empty() {
        anyhow::bail!("No hay imágenes para el documento {}", document_id);
    }
    page_objects.sort_by_key(|n| extract_page_number(n));
    println!("Páginas: {}", page_objects.len());

    let images = download_batch(storage.clone(), page_objects.clone()).await?;

    // ── Encode with the production path (image_batch = 1) ───────
    let use_cpu = std::env::var("MODEL_DEVICE").map(|d| d == "cpu").unwrap_or(false);
    let embedding = EmbeddingService::spawn(
        &config.model_path,
        use_cpu,
        config.use_bf16,
        config.model_dims,
        config.model_queue_capacity,
        1,
    )?;

    let started = std::time::Instant::now();
    let page_embs = embedding.encode_images_from_bytes(images).await?;
    println!(
        "Encode: {} páginas en {:.1}s",
        page_embs.len(),
        started.elapsed().as_secs_f64()
    );

    let query_list: Vec<String> = if queries.is_empty() {
        DEFAULT_QUERIES.iter().map(|s| s.to_string()).collect()
    } else {
        queries
    };

    let page_embs_arc = Arc::new(page_embs);
    let mut results: Vec<QueryResult> = Vec::new();
    for q in &query_list {
        let q_embs = embedding.encode_query(q.clone()).await?;
        let scores = embedding.score(q_embs, page_embs_arc.clone()).await?;
        let mut ranking: Vec<usize> = (0..scores.len()).collect();
        ranking.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.push(QueryResult {
            query: q.clone(),
            ranking,
            scores,
        });
    }

    let base_dir = PathBuf::from(&dir).join(&document_id);
    let blob_path = base_dir.join("baseline_embeddings.bin");
    let json_path = base_dir.join("results.json");

    if record {
        std::fs::create_dir_all(&base_dir)?;
        std::fs::write(&blob_path, serialize_embeddings(&page_embs_arc))?;
        let baseline = Baseline {
            document_id: document_id.clone(),
            model_path: config.model_path.clone(),
            model_dims: config.model_dims,
            pages: page_objects,
            queries: results,
        };
        std::fs::write(&json_path, serde_json::to_string_pretty(&baseline)?)?;
        println!("\nBaseline guardado en {}", base_dir.display());
        return Ok(());
    }

    // ── Compare against baseline ─────────────────────────────────
    let baseline: Baseline = serde_json::from_str(
        &std::fs::read_to_string(&json_path)
            .with_context(|| format!("No hay baseline en {} (corre --record primero)", json_path.display()))?,
    )?;
    let baseline_embs = deserialize_embeddings(&std::fs::read(&blob_path)?)?;

    anyhow::ensure!(
        baseline.pages == page_objects,
        "El conjunto de páginas cambió desde el baseline; regraba con --record"
    );

    let mut failed = false;

    // Embedding diff
    println!("\n── Embeddings vs baseline ──");
    if baseline_embs.len() != page_embs_arc.len() {
        println!(
            "FAIL: número de páginas {} vs baseline {}",
            page_embs_arc.len(),
            baseline_embs.len()
        );
        failed = true;
    } else {
        let mut changed_values: u64 = 0;
        let mut total_values: u64 = 0;
        let mut max_abs_diff: f32 = 0.0;
        for (b, n) in baseline_embs.iter().zip(page_embs_arc.iter()) {
            if b.seq_len != n.seq_len || b.dims != n.dims {
                println!("FAIL: shape de página cambió ({}x{} vs {}x{})", n.seq_len, n.dims, b.seq_len, b.dims);
                failed = true;
                continue;
            }
            total_values += b.data.len() as u64;
            for (bv, nv) in b.data.iter().zip(n.data.iter()) {
                if bv.to_bits() != nv.to_bits() {
                    changed_values += 1;
                    max_abs_diff = max_abs_diff.max((bv.to_f32() - nv.to_f32()).abs());
                }
            }
        }
        if changed_values == 0 {
            println!("BIT-EXACT ({} valores)", total_values);
        } else {
            println!(
                "{} de {} valores difieren ({:.4}%), max |Δ| = {:.6}",
                changed_values,
                total_values,
                changed_values as f64 / total_values as f64 * 100.0,
                max_abs_diff
            );
            if max_abs_diff > tol {
                println!("FAIL: max |Δ| {} > tolerancia {}", max_abs_diff, tol);
                failed = true;
            }
        }
    }

    // Ranking diff
    println!("\n── Búsquedas vs baseline ──");
    for new_r in &results {
        let Some(base_r) = baseline.queries.iter().find(|b| b.query == new_r.query) else {
            println!("  \"{}\": no está en el baseline, omitida", new_r.query);
            continue;
        };
        let max_score_delta = base_r
            .scores
            .iter()
            .zip(new_r.scores.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let top5_base: Vec<usize> = base_r.ranking.iter().take(5).copied().collect();
        let top5_new: Vec<usize> = new_r.ranking.iter().take(5).copied().collect();
        if top5_base == top5_new {
            println!(
                "  OK   \"{}\"  top-5 idéntico, max |Δscore| = {:.4}",
                new_r.query, max_score_delta
            );
        } else {
            println!(
                "  FAIL \"{}\"  top-5 cambió: {:?} → {:?}  (max |Δscore| = {:.4})",
                new_r.query, top5_base, top5_new, max_score_delta
            );
            failed = true;
        }
    }

    println!();
    if failed {
        println!("RESULTADO: FAIL — los cambios alteran los resultados");
        std::process::exit(1);
    }
    println!("RESULTADO: OK — sin cambios observables");
    Ok(())
}

/// Pick the most recent complete document with a reasonable page count.
async fn pick_document(config: &AppConfig) -> Result<String> {
    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(1)
        .connect(&config.database_url)
        .await?;
    let row: Option<(String,)> = sqlx::query_as(
        "SELECT d.id FROM documents d
         JOIN pages p ON p.document_id = d.id
         WHERE d.status = 'complete'
         GROUP BY d.id, d.upload_date
         HAVING count(*) BETWEEN 10 AND 100
         ORDER BY d.upload_date DESC
         LIMIT 1",
    )
    .fetch_optional(&pool)
    .await?;
    row.map(|(id,)| id)
        .ok_or_else(|| anyhow::anyhow!("No hay documentos complete con 10-100 páginas; pasa --doc"))
}
