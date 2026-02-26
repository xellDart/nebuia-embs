use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::response::Json;
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use crate::services::pdf_service;
use crate::state::AppState;

// ── Request / Response types ─────────────────────────────────

#[derive(Deserialize)]
pub struct ProcessPdfRequest {
    pub document_id: String,
}

#[derive(Deserialize)]
pub struct SearchQuery {
    pub query: Option<String>,
    pub k: Option<usize>,
    pub continues: Option<bool>,
}

#[derive(Serialize)]
pub struct MessageResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document_id: Option<String>,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

type ApiResult<T> = Result<Json<T>, (StatusCode, Json<ErrorResponse>)>;

fn api_error(status: StatusCode, msg: impl Into<String>) -> (StatusCode, Json<ErrorResponse>) {
    (
        status,
        Json(ErrorResponse {
            error: msg.into(),
        }),
    )
}

// ── Handlers ─────────────────────────────────────────────────

pub async fn process_pdf(
    State(state): State<Arc<AppState>>,
    Json(body): Json<ProcessPdfRequest>,
) -> ApiResult<MessageResponse> {
    let document_id = body.document_id.clone();

    // Check if document exists
    let doc = crate::repositories::document_repository::get_document(
        &state.db_pool,
        &document_id,
    )
    .await
    .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if doc.is_none() {
        return Err(api_error(
            StatusCode::NOT_FOUND,
            format!("Document {} does not exist", document_id),
        ));
    }

    // Spawn background task
    let state_clone = state.clone();
    let doc_id = document_id.clone();
    tokio::spawn(async move {
        if let Err(e) = pdf_service::process_document_embeddings(
            &doc_id,
            &state_clone.db_pool,
            &state_clone.storage,
            &state_clone.embedding,
            &state_clone.cache,
            state_clone.config.model_batch_size,
        )
        .await
        {
            tracing::error!("Error processing document {}: {}", doc_id, e);
        }
    });

    Ok(Json(MessageResponse {
        document_id: Some(document_id),
        message: "Embeddings processing started".into(),
        status: Some("processing".into()),
    }))
}

pub async fn search_document(
    State(state): State<Arc<AppState>>,
    Path(document_id): Path<String>,
    Query(params): Query<SearchQuery>,
) -> ApiResult<serde_json::Value> {
    let query = params
        .query
        .ok_or_else(|| api_error(StatusCode::BAD_REQUEST, "Query parameter is required"))?;
    let k = params.k.unwrap_or(3);
    let continues = params.continues.unwrap_or(false);

    let results = pdf_service::search_document(
        &document_id,
        &query,
        k,
        continues,
        &state.db_pool,
        &state.storage,
        &state.embedding,
        &state.cache,
    )
    .await
    .map_err(|e| {
        let msg = e.to_string();
        if msg.contains("not found") {
            api_error(StatusCode::NOT_FOUND, msg)
        } else {
            api_error(StatusCode::INTERNAL_SERVER_ERROR, msg)
        }
    })?;

    Ok(Json(serde_json::json!(results)))
}

pub async fn delete_document(
    State(state): State<Arc<AppState>>,
    Path(document_id): Path<String>,
) -> ApiResult<serde_json::Value> {
    let result = pdf_service::delete_document(
        &document_id,
        &state.db_pool,
        &state.storage,
        &state.cache,
    )
    .await
    .map_err(|e| {
        let msg = e.to_string();
        if msg.contains("not found") {
            api_error(StatusCode::NOT_FOUND, msg)
        } else {
            api_error(StatusCode::INTERNAL_SERVER_ERROR, msg)
        }
    })?;

    Ok(Json(result))
}

pub async fn health_check(
    State(state): State<Arc<AppState>>,
) -> ApiResult<serde_json::Value> {
    let db_healthy = sqlx::query("SELECT 1")
        .execute(&state.db_pool)
        .await
        .is_ok();

    let minio_healthy = state.storage.health_check().await;
    let model_healthy = state.embedding.is_alive();

    let healthy = db_healthy && minio_healthy && model_healthy;

    let status = serde_json::json!({
        "healthy": healthy,
        "services": {
            "database": { "healthy": db_healthy },
            "minio": { "healthy": minio_healthy },
            "model": { "healthy": model_healthy },
        },
        "timestamp": chrono::Utc::now().to_rfc3339(),
    });

    Ok(Json(status))
}
