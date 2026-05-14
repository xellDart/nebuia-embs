use std::sync::Arc;

use axum::routing::{delete, get, post};
use axum::Router;

use crate::handlers;
use crate::state::AppState;

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/process-pdf", post(handlers::process_pdf))
        .route(
            "/simple/search/{document_id}",
            get(handlers::search_document),
        )
        .route(
            "/document/{document_id}",
            delete(handlers::delete_document),
        )
        .route("/health", get(handlers::health_check))
        .with_state(state)
}
