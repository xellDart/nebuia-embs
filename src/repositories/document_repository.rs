use sqlx::postgres::PgPool;

use crate::models::database::{Document, Page};

pub async fn get_document(pool: &PgPool, document_id: &str) -> anyhow::Result<Option<Document>> {
    let doc = sqlx::query_as::<_, Document>("SELECT * FROM documents WHERE id = $1")
        .bind(document_id)
        .fetch_optional(pool)
        .await?;
    Ok(doc)
}

pub async fn update_document_status(
    pool: &PgPool,
    document_id: &str,
    status: &str,
) -> anyhow::Result<bool> {
    let result = sqlx::query("UPDATE documents SET status = $1 WHERE id = $2")
        .bind(status)
        .bind(document_id)
        .execute(pool)
        .await?;
    Ok(result.rows_affected() > 0)
}

pub async fn get_document_with_pages(
    pool: &PgPool,
    document_id: &str,
) -> anyhow::Result<Option<(Document, Vec<Page>)>> {
    let doc = match get_document(pool, document_id).await? {
        Some(d) => d,
        None => return Ok(None),
    };

    let pages = sqlx::query_as::<_, Page>(
        "SELECT * FROM pages WHERE document_id = $1 ORDER BY page_number",
    )
    .bind(document_id)
    .fetch_all(pool)
    .await?;

    Ok(Some((doc, pages)))
}

pub async fn save_page(
    pool: &PgPool,
    document_id: &str,
    page_number: i32,
    image_path: &str,
) -> anyhow::Result<()> {
    // Upsert: update if exists, insert if not
    sqlx::query(
        r#"
        INSERT INTO pages (id, document_id, page_number, image_path)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (document_id, page_number)
        DO UPDATE SET image_path = EXCLUDED.image_path
        "#,
    )
    .bind(uuid::Uuid::new_v4().to_string())
    .bind(document_id)
    .bind(page_number)
    .bind(image_path)
    .execute(pool)
    .await?;

    Ok(())
}

pub async fn delete_document(pool: &PgPool, document_id: &str) -> anyhow::Result<bool> {
    // Pages cascade-delete via FK
    let result = sqlx::query("DELETE FROM documents WHERE id = $1")
        .bind(document_id)
        .execute(pool)
        .await?;
    Ok(result.rows_affected() > 0)
}
