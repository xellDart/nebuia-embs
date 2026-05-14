# nebuia-embs

High-performance document embedding service powered by **ColQwen3** (ColBERT-style multi-vector embeddings) built in Rust. Replaces the Python embeddings service with a single statically-linked binary.

Uses [crane-core](https://github.com/xellDart/Crane) as the inference backend for the ColQwen3 vision-language model.

## Features

- **ColQwen3 4B** multi-vector embeddings for document page images
- **ColBERT MaxSim** late-interaction scoring for retrieval
- **Matryoshka dimension support** — configurable embedding dims (1280 / 2560)
- **CUDA + Flash Attention** support with automatic detection
- **BF16 inference** for reduced GPU memory
- **S3-compatible storage** (MinIO, UpCloud, AWS S3) for images and embeddings
- **NATS JetStream** consumer for async job processing
- **LRU cache** with TTL for hot document embeddings
- **Zstd compression** for embedding storage
- Built-in **health checks** (DB, S3, model)

## Architecture

```
nebuia-embs/
├── src/
│   ├── main.rs                    # CLI, axum server, startup
│   ├── config.rs                  # Environment configuration
│   ├── state.rs                   # Shared application state
│   ├── routes.rs                  # Router setup
│   ├── handlers.rs                # HTTP handlers
│   ├── models/
│   │   └── database.rs            # SQLx models (Document, Page)
│   ├── services/
│   │   ├── embedding_service.rs   # ColQwen3 model wrapper (dedicated thread)
│   │   ├── pdf_service.rs         # Pipeline: download → encode → upload
│   │   ├── cache_service.rs       # Moka LRU cache
│   │   └── nats_consumer.rs       # NATS JetStream pull consumer
│   └── repositories/
│       ├── document_repository.rs # PostgreSQL CRUD
│       └── storage_repository.rs  # S3 operations
├── nebuia-ctl                     # Service management CLI
├── install.sh                     # One-line installer
├── Cargo.toml
└── .env.example
```

### How it works

1. **Process**: Document page images (JPEG/PNG) are stored in S3. On a `/process-pdf` request, the service downloads each page image, runs it through ColQwen3 to produce multi-vector embeddings (~1225 tokens × dims per page), serializes them as BF16, compresses with zstd, and uploads back to S3.

2. **Search**: On a `/simple/search/{id}?query=...` request, the service encodes the text query through the same model, loads the document's embeddings (from cache or S3), and scores each page using ColBERT MaxSim. Returns top-k page image paths ranked by relevance.

3. **Model thread**: The ColQwen3 model runs on a dedicated `std::thread` (not async) with an `mpsc` channel interface. All GPU inference happens on this single thread, while the async runtime handles HTTP, S3, and DB concurrently.

## Quick Start

### One-line install

```bash
curl -fsSL https://raw.githubusercontent.com/xellDart/nebuia-embs/main/install.sh | bash
```

With options:
```bash
# CPU-only build
curl -fsSL ... | bash -s -- --cpu

# Custom install directory
curl -fsSL ... | bash -s -- --dir /opt/nebuia-embs
```

After install:
```bash
cp .env.example .env       # configure database, S3, model path
nebuia-ctl start           # start as daemon
nebuia-ctl status          # check health
```

### Manual build

```bash
git clone https://github.com/xellDart/nebuia-embs.git
cd nebuia-embs

# Copy and edit configuration
cp .env.example .env
# Edit .env with your database, S3, and model settings

# Build (auto-detect features)
cargo build --release --features cuda          # CUDA
cargo build --release --features flash-attn    # CUDA + Flash Attention
cargo build --release                          # CPU only

# Run
./target/release/nebuia-embs
```

### Model weights

Download ColQwen3-4B weights from HuggingFace:

```bash
huggingface-cli download OpenSearch-AI/Ops-Colqwen3-4B --local-dir ./colqwen3-4b
```

Set `MODEL_PATH` in `.env` to the download directory.

## API

### Process document embeddings

```bash
POST /process-pdf
Content-Type: application/json

{"document_id": "abc-123"}
```

Starts background processing. The service downloads page images from S3 (`{document_id}_page_{N}.jpeg`), generates embeddings, and uploads `{document_id}_embeddings.zst`.

### Search document

```bash
GET /simple/search/{document_id}?query=tabla+de+accionistas&k=3
```

Returns top-k page image paths ranked by similarity:
```json
["abc-123_page_5.jpeg", "abc-123_page_2.jpeg", "abc-123_page_8.jpeg"]
```

Parameters:
- `query` (required) — search text
- `k` (optional, default: 3) — number of results
- `continues` (optional, default: false) — return k consecutive pages from the best match

### Delete document

```bash
DELETE /document/{document_id}
```

Deletes document from DB, S3 (images + embeddings), and cache.

### Health check

```bash
GET /health
```

```json
{
  "healthy": true,
  "services": {
    "database": {"healthy": true},
    "minio": {"healthy": true},
    "model": {"healthy": true}
  },
  "timestamp": "2026-02-25T..."
}
```

## CLI: nebuia-ctl

Service management CLI installed automatically by `install.sh` or available directly from the repo.

```
nebuia-ctl — manage nebuia-embs service

Usage: nebuia-ctl <command> [args]

Commands:
  start              Start service as daemon
  stop               Stop running service
  restart            Restart service
  status             Show PID, memory, GPU, uptime, health
  log [n]            Tail logs (default: last 50 lines)
  health             Quick health check (JSON)
  search <id> <q>    Search document pages
  process <id>       Trigger embedding processing
  help               Show help
```

### Examples

```bash
# Start as daemon, wait for model to load
nebuia-ctl start

# Check status (PID, RAM, GPU memory, threads, uptime, services)
nebuia-ctl status

# Search a document
nebuia-ctl search abc-123 "tabla de accionistas"

# Trigger processing via CLI
nebuia-ctl process def-456

# Tail logs
nebuia-ctl log 100

# Restart
nebuia-ctl restart
```

### Status output

```
  nebuia-embs v0.1.0

  Service running

  PID        12345
  Port       8000
  Uptime     2h 15m 30s
  RAM        354.0 MB
  VIRT       25.3 GB
  Threads    22
  GPU mem    13400 MiB
  Log        /path/to/nebuia-embs.log

  Services   DB  MinIO  Model
```

## Configuration

All configuration via environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | *required* | PostgreSQL connection string |
| `MINIO_ENDPOINT` | *required* | S3-compatible endpoint |
| `MINIO_ACCESS_KEY` | *required* | S3 access key |
| `MINIO_SECRET_KEY` | *required* | S3 secret key |
| `MINIO_BUCKET` | *required* | S3 bucket name |
| `MODEL_PATH` | `ops-colqwen3-FP8` | Path to ColQwen3 model weights |
| `MODEL_DEVICE` | `cuda` | `cuda` or `cpu` |
| `MODEL_BATCH_SIZE` | `9` | Images per batch (reduce if OOM) |
| `MODEL_DTYPE` | `bfloat16` | `bfloat16` or `float32` |
| `MODEL_DIMS` | `1280` | Embedding dimensions (1280 or 2560) |
| `CACHE_MAX_SIZE` | `10` | Max documents in LRU cache |
| `CACHE_EXPIRY_HOURS` | `24` | Cache TTL in hours |
| `NATS_URL` | `nats://localhost:4222` | NATS server URL |
| `NATS_ENABLED` | `true` | Enable NATS JetStream consumer |
| `HOST` | `0.0.0.0` | Listen address |
| `PORT` | `8000` | Listen port |
| `MAX_RETRIES` | `5` | Max retry attempts |
| `RETRY_DELAY` | `3` | Retry delay in seconds |

### Embedding dimensions

ColQwen3 supports Matryoshka representation learning. You can choose between:

- **2560 dims** — full projection, maximum accuracy
- **1280 dims** — half the storage, comparable accuracy for most queries

Storage impact per page (~1225 tokens):
| Dims | Raw (BF16) | Compressed (zstd) |
|------|------------|-------------------|
| 2560 | ~3.1 MB | ~2.6 MB |
| 1280 | ~3.1 MB | ~2.4 MB |

## NATS Integration

When `NATS_ENABLED=true`, the service creates a durable pull consumer on the `EMBEDDINGS` stream:

- **Subject**: `embeddings.process`
- **Consumer**: `embeddings_workers` (shared across instances)
- **Message format**: `{"document_id": "..."}`
- **Completion**: publishes to `embeddings.completed.complete` or `embeddings.completed.error`

## Embedding Storage Format

Embeddings are serialized as BF16 with a minimal header:

```
[num_pages: u32]
  [seq_len: u32] [dims: u32] [bf16 data: seq_len × dims × 2 bytes]
  [seq_len: u32] [dims: u32] [bf16 data: ...]
  ...
```

Compressed with zstd level 9 and stored as `{document_id}_embeddings.zst`.

## Requirements

- **Rust** 1.75+ (installed automatically by `install.sh`)
- **PostgreSQL** database with `documents` and `pages` tables
- **S3-compatible storage** (MinIO, UpCloud S3, AWS S3)
- **CUDA 12.x** + GPU with 16 GB+ VRAM (for GPU inference)
- **NATS** server with JetStream (optional, for async processing)
- [crane-core](https://github.com/xellDart/Crane) as a path dependency

## License

Proprietary — NebuIA
