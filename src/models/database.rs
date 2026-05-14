use sqlx::postgres::PgPool;
use sqlx::FromRow;

#[derive(Debug, FromRow)]
pub struct Document {
    pub id: String,
    pub filename: String,
    pub upload_date: String,
    pub status: String,
}

#[derive(Debug, FromRow)]
pub struct Page {
    pub id: String,
    pub document_id: String,
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
