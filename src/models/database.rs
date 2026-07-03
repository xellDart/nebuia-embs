use sqlx::postgres::PgPool;
use sqlx::FromRow;

// Row models: only the columns the service actually reads. The full table
// schemas live in create_tables() below; sqlx ignores unselected extra columns.
#[derive(Debug, Clone, FromRow)]
pub struct Document {
    pub status: String,
}

#[derive(Debug, Clone, FromRow)]
pub struct Page {
    pub page_number: i32,
    pub image_path: String,
}

pub async fn create_tables(pool: &PgPool) -> sqlx::Result<()> {
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            upload_date TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'processing'
        )
        "#,
    )
    .execute(pool)
    .await?;

    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS pages (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            page_number INTEGER NOT NULL,
            image_path TEXT NOT NULL
        )
        "#,
    )
    .execute(pool)
    .await?;

    // Processing provenance: which node produced the embeddings and how long it took
    for ddl in [
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS processed_by TEXT",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS processed_at TIMESTAMPTZ",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS processing_secs DOUBLE PRECISION",
    ] {
        sqlx::query(ddl).execute(pool).await?;
    }

    // Ensure unique constraint exists for upsert support
    sqlx::query(
        r#"
        CREATE UNIQUE INDEX IF NOT EXISTS idx_pages_doc_page
        ON pages (document_id, page_number)
        "#,
    )
    .execute(pool)
    .await?;

    Ok(())
}
