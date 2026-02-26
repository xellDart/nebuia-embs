#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────
# nebuia-embs — One-Line Installer
#
# Install and build nebuia-embs with a single command:
#
#   curl -fsSL https://raw.githubusercontent.com/xellDart/nebuia-embs/main/install.sh | bash
#
# Or with options:
#   curl -fsSL ... | bash -s -- --model-path /path/to/colqwen3 --start
#   curl -fsSL ... | bash -s -- --cpu
#   curl -fsSL ... | bash -s -- --dir /opt/nebuia-embs
#
# What it does:
#   1. Installs Rust (if not found)
#   2. Clones nebuia-embs (or updates if already present)
#   3. Auto-detects CUDA, GPU, Flash Attention
#   4. Builds with optimal features
#   5. Installs nebuia-ctl management CLI
#   6. Optionally starts the service
# ─────────────────────────────────────────────────────────────────────

REPO_URL="https://github.com/xellDart/nebuia-embs.git"
BRANCH="main"
INSTALL_DIR=""
MODEL_PATH=""
FORCE_CPU=false
START_AFTER=false
DAEMON_MODE=false
PORT=8000
ENV_FILE=""

for arg in "$@"; do
  case "$arg" in
    --model-path=*) MODEL_PATH="${arg#--model-path=}" ;;
    --model-path)   ;; # handled below
    --dir=*)        INSTALL_DIR="${arg#--dir=}" ;;
    --dir)          ;; # handled below
    --branch=*)     BRANCH="${arg#--branch=}" ;;
    --port=*)       PORT="${arg#--port=}" ;;
    --port)         ;; # handled below
    --env=*)        ENV_FILE="${arg#--env=}" ;;
    --env)          ;; # handled below
    --cpu)          FORCE_CPU=true ;;
    --start)        START_AFTER=true ;;
    --daemon)       START_AFTER=true; DAEMON_MODE=true ;;
    -h|--help)
      cat <<'HELP'
nebuia-embs Installer — document embedding service powered by ColQwen3

Usage:
  curl -fsSL https://raw.githubusercontent.com/xellDart/nebuia-embs/main/install.sh | bash
  curl -fsSL ... | bash -s -- [OPTIONS]

Options:
  --model-path <path>   Path to ColQwen3 model weights
  --dir <path>          Install directory (default: ./nebuia-embs)
  --branch <name>       Git branch to checkout (default: main)
  --env <path>          Path to .env configuration file
  --cpu                 Force CPU-only build (skip CUDA detection)
  --start               Start the service in foreground after build
  --daemon              Start the service as a background daemon after build
  --port <port>         Service port (default: 8000)
  -h, --help            Show this help

Examples:
  # Just build (auto-detect GPU)
  curl -fsSL ... | bash

  # Build + start with local model
  curl -fsSL ... | bash -s -- --model-path /path/to/colqwen3 --start

  # Build + start as daemon
  curl -fsSL ... | bash -s -- --daemon --port 9000

  # Install to custom directory
  curl -fsSL ... | bash -s -- --dir /opt/nebuia-embs
HELP
      exit 0
      ;;
    *)
      if [ "${PREV_ARG:-}" = "--model-path" ]; then MODEL_PATH="$arg"; fi
      if [ "${PREV_ARG:-}" = "--dir" ];        then INSTALL_DIR="$arg"; fi
      if [ "${PREV_ARG:-}" = "--port" ];       then PORT="$arg"; fi
      if [ "${PREV_ARG:-}" = "--env" ];        then ENV_FILE="$arg"; fi
      ;;
  esac
  PREV_ARG="$arg"
done

# ── Helpers ──────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

info()  { echo -e "  ${BLUE}${BOLD}[INFO]${RESET}  $*"; }
ok()    { echo -e "  ${GREEN}${BOLD}[ OK ]${RESET}  $*"; }
warn()  { echo -e "  ${YELLOW}${BOLD}[WARN]${RESET}  $*"; }
fail()  { echo -e "  ${RED}${BOLD}[FAIL]${RESET}  $*"; exit 1; }

elapsed() {
  local secs=$1
  printf "%dm %02ds" $((secs / 60)) $((secs % 60))
}

# ── Banner ───────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}${CYAN}"
cat << 'BANNER'
   ███╗   ██╗███████╗██████╗ ██╗   ██╗██╗ █████╗
   ████╗  ██║██╔════╝██╔══██╗██║   ██║██║██╔══██╗
   ██╔██╗ ██║█████╗  ██████╔╝██║   ██║██║███████║
   ██║╚██╗██║██╔══╝  ██╔══██╗██║   ██║██║██╔══██║
   ██║ ╚████║███████╗██████╔╝╚██████╔╝██║██║  ██║
   ╚═╝  ╚═══╝╚══════╝╚═════╝  ╚═════╝ ╚═╝╚═╝  ╚═╝
BANNER
echo -e "${RESET}"
echo -e "  ${DIM}Document Embedding Service — ColQwen3 + Rust${RESET}"
echo ""
echo "  ────────────────────────────────────────────────"
echo ""

# ═════════════════════════════════════════════════════════════════════
#  Step 1: Prerequisites
# ═════════════════════════════════════════════════════════════════════

info "Checking prerequisites..."
echo ""

# ── Git ──
if ! command -v git &>/dev/null; then
  fail "git is required. Install it: https://git-scm.com/"
fi
ok "git: $(git --version)"

# ── Rust ──
if command -v cargo &>/dev/null; then
  ok "Rust: $(rustc --version 2>/dev/null)"
else
  info "Rust not found. Installing via rustup..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
  if command -v cargo &>/dev/null; then
    ok "Rust installed: $(rustc --version 2>/dev/null)"
  else
    fail "Rust installation failed. Install manually: https://rustup.rs/"
  fi
fi

# ── CUDA ──
HAS_CUDA=false
HAS_FLASH_ATTN=false
GPU_COMPUTE_CAP=""
GPU_NAME=""
CUDA_VERSION=""
GPU_MEM=""

if [ "$FORCE_CPU" = false ]; then
  if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    ok "CUDA: $CUDA_VERSION"
    HAS_CUDA=true

    if command -v nvidia-smi &>/dev/null; then
      GPU_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')
      GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/^[[:space:]]*//')
      GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')

      if [ -n "$GPU_NAME" ]; then
        ok "GPU: $GPU_NAME ($GPU_MEM, sm_${GPU_COMPUTE_CAP//./_})"
      fi

      if [ -n "$GPU_COMPUTE_CAP" ]; then
        MAJOR=$(echo "$GPU_COMPUTE_CAP" | cut -d. -f1)
        if [ "$MAJOR" -ge 8 ] 2>/dev/null; then
          HAS_FLASH_ATTN=true
          ok "Flash Attention: supported (SM_80+)"
        fi
      fi
    fi
  else
    warn "CUDA not found. Building for CPU only."
    warn "For GPU: install CUDA Toolkit and ensure nvcc is in PATH."
  fi
else
  info "CPU-only build requested (--cpu)"
fi

echo ""

# ═════════════════════════════════════════════════════════════════════
#  Step 2: Clone / Update Repository
# ═════════════════════════════════════════════════════════════════════

# Detect if we're already inside the nebuia-embs repo
INSIDE_REPO=false
if [ -f "Cargo.toml" ] && grep -q 'name = "nebuia-embs"' Cargo.toml 2>/dev/null; then
  INSIDE_REPO=true
  PROJECT_DIR="$(pwd)"
fi

if [ "$INSIDE_REPO" = true ]; then
  info "Already inside nebuia-embs repository: $PROJECT_DIR"
  info "Pulling latest changes..."
  git pull --ff-only 2>/dev/null || warn "Could not pull (uncommitted changes?). Continuing with local version."
else
  [ -z "$INSTALL_DIR" ] && INSTALL_DIR="nebuia-embs"
  if [ -d "$INSTALL_DIR" ] && [ -f "$INSTALL_DIR/Cargo.toml" ]; then
    info "nebuia-embs directory exists at $INSTALL_DIR, updating..."
    cd "$INSTALL_DIR"
    git pull --ff-only 2>/dev/null || warn "Could not pull. Continuing with local version."
  else
    info "Cloning nebuia-embs..."
    git clone --branch "$BRANCH" --depth 1 "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
  fi
  PROJECT_DIR="$(pwd)"
fi

ok "Project directory: $PROJECT_DIR"
echo ""

# ═════════════════════════════════════════════════════════════════════
#  Step 3: Build
# ═════════════════════════════════════════════════════════════════════

FEATURES=()

if [ "$HAS_FLASH_ATTN" = true ]; then
  FEATURES+=("flash-attn")
  FA_CACHE_DIR="/tmp/nebuia_flash_attn_cache"
  mkdir -p "$FA_CACHE_DIR"
  export CANDLE_FLASH_ATTN_BUILD_DIR="$FA_CACHE_DIR"
elif [ "$HAS_CUDA" = true ]; then
  FEATURES+=("cuda")
fi

FEATURES_FLAG=""
if [ ${#FEATURES[@]} -gt 0 ]; then
  FEATURES_STR=$(IFS=,; echo "${FEATURES[*]}")
  FEATURES_FLAG="--features $FEATURES_STR"
  ok "Build features: $FEATURES_STR"
else
  info "Build features: CPU-only"
fi

BUILD_START=$(date +%s)

info "Building nebuia-embs (release)..."
if [ "$HAS_FLASH_ATTN" = true ]; then
  info "First build with Flash Attention takes ~10 min (CUTLASS compilation)."
  info "Subsequent builds are much faster (cached)."
fi

# shellcheck disable=SC2086
cargo build --release $FEATURES_FLAG 2>&1

BUILD_END=$(date +%s)
BUILD_ELAPSED=$((BUILD_END - BUILD_START))

echo ""
ok "Build completed in $(elapsed $BUILD_ELAPSED)"

BIN_PATH="$PROJECT_DIR/target/release/nebuia-embs"
if [ -f "$BIN_PATH" ]; then
  BIN_SIZE=$(du -sh "$BIN_PATH" | cut -f1)
  ok "Binary: $BIN_PATH ($BIN_SIZE)"
fi

# ── Install nebuia-ctl to PATH ──
if [ -f "$PROJECT_DIR/nebuia-ctl" ]; then
  chmod +x "$PROJECT_DIR/nebuia-ctl"
  LINK_DIR="$HOME/.local/bin"
  mkdir -p "$LINK_DIR"
  ln -sf "$PROJECT_DIR/nebuia-ctl" "$LINK_DIR/nebuia-ctl"

  if ! echo "$PATH" | tr ':' '\n' | grep -q "$LINK_DIR"; then
    SHELL_RC=""
    if [ -f "$HOME/.bashrc" ]; then SHELL_RC="$HOME/.bashrc";
    elif [ -f "$HOME/.zshrc" ]; then SHELL_RC="$HOME/.zshrc";
    fi
    if [ -n "$SHELL_RC" ] && ! grep -q '.local/bin' "$SHELL_RC" 2>/dev/null; then
      echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
      info "Added ~/.local/bin to PATH in $SHELL_RC (restart shell or run: source $SHELL_RC)"
    fi
    export PATH="$LINK_DIR:$PATH"
  fi
  ok "nebuia-ctl installed to $LINK_DIR/nebuia-ctl"
fi
echo ""

# ═════════════════════════════════════════════════════════════════════
#  Step 4: Start Service (optional)
# ═════════════════════════════════════════════════════════════════════

SERVICE_STARTED=false
LOG_FILE="$PROJECT_DIR/nebuia-embs.log"
PID_FILE="$PROJECT_DIR/nebuia-embs.pid"

ENV_FLAG=""
if [ -n "$ENV_FILE" ]; then
  ENV_FLAG="--env-file $ENV_FILE"
elif [ -f "$PROJECT_DIR/.env" ]; then
  ENV_FLAG="--env-file $PROJECT_DIR/.env"
fi

if [ "$START_AFTER" = true ]; then
  if [ ! -f "$BIN_PATH" ]; then
    fail "Binary not found at $BIN_PATH"
  fi

  if [ "$DAEMON_MODE" = true ]; then
    info "Starting nebuia-embs daemon on port $PORT..."

    # Kill existing daemon if running
    if [ -f "$PID_FILE" ]; then
      OLD_PID=$(cat "$PID_FILE" 2>/dev/null)
      if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        warn "Stopping existing daemon (PID $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
      fi
    fi

    # shellcheck disable=SC2086
    PORT="$PORT" nohup "$BIN_PATH" $ENV_FLAG > "$LOG_FILE" 2>&1 &

    DAEMON_PID=$!
    echo "$DAEMON_PID" > "$PID_FILE"

    sleep 5
    if kill -0 "$DAEMON_PID" 2>/dev/null; then
      ok "Daemon started (PID $DAEMON_PID)"
      ok "Logs:  tail -f $LOG_FILE"
      ok "Stop:  nebuia-ctl stop"
      SERVICE_STARTED=true
    else
      warn "Daemon may have failed to start. Check logs:"
      echo "  tail -20 $LOG_FILE"
    fi
  else
    info "Starting nebuia-embs on port $PORT (foreground)..."
    echo ""
    # shellcheck disable=SC2086
    exec env PORT="$PORT" "$BIN_PATH" $ENV_FLAG
  fi
fi

echo ""

# ═════════════════════════════════════════════════════════════════════
#  Summary
# ═════════════════════════════════════════════════════════════════════

echo -e "  ${BOLD}${CYAN}════════════════════════════════════════════════${RESET}"
echo -e "  ${BOLD}  nebuia-embs Installation Complete${RESET}"
echo -e "  ${BOLD}${CYAN}════════════════════════════════════════════════${RESET}"
echo ""
echo "  Location:    $PROJECT_DIR"
echo "  Binary:      $BIN_PATH"
echo "  Platform:    $(uname -s) $(uname -m)"
echo "  Rust:        $(rustc --version 2>/dev/null)"
if [ "$HAS_CUDA" = true ]; then
  echo "  CUDA:        $CUDA_VERSION"
  [ -n "$GPU_NAME" ] && echo "  GPU:         $GPU_NAME ($GPU_MEM)"
  echo "  Flash Attn:  $([ "$HAS_FLASH_ATTN" = true ] && echo "enabled" || echo "disabled")"
else
  echo "  CUDA:        not detected (CPU build)"
fi
echo "  Build time:  $(elapsed $BUILD_ELAPSED)"
if [ -n "$MODEL_PATH" ] && [ -d "$MODEL_PATH" ]; then
  echo "  Model:       $MODEL_PATH"
fi
if [ "$SERVICE_STARTED" = true ]; then
  echo "  Service:     http://localhost:$PORT (daemon, PID $(cat "$PID_FILE"))"
fi
echo ""

# ── nebuia-ctl usage ──
echo -e "  ${BOLD}Service management (nebuia-ctl):${RESET}"
echo ""
echo "    nebuia-ctl start        Start service as daemon"
echo "    nebuia-ctl stop         Stop running service"
echo "    nebuia-ctl restart      Restart service"
echo "    nebuia-ctl status       Show PID, memory, uptime"
echo "    nebuia-ctl log          Tail service logs"
echo "    nebuia-ctl health       Quick health check"
echo "    nebuia-ctl search <id> <query>  Test search"
echo ""

# ── How to test ──
echo -e "  ${BOLD}Test the API:${RESET}"
echo ""
cat << CURL_EXAMPLE
    # Health check
    curl http://localhost:$PORT/health

    # Process a document
    curl -X POST http://localhost:$PORT/process-pdf \\
      -H 'Content-Type: application/json' \\
      -d '{"document_id": "your-doc-id"}'

    # Search
    curl "http://localhost:$PORT/simple/search/your-doc-id?query=tabla&k=3"

    # Delete
    curl -X DELETE http://localhost:$PORT/document/your-doc-id
CURL_EXAMPLE
echo ""

# ── Configuration ──
if [ ! -f "$PROJECT_DIR/.env" ]; then
  echo -e "  ${BOLD}Configuration:${RESET}"
  echo ""
  echo "    Copy .env.example to .env and set your credentials:"
  echo "    cp $PROJECT_DIR/.env.example $PROJECT_DIR/.env"
  echo ""
fi

echo "  Re-run installer:  curl -fsSL https://raw.githubusercontent.com/xellDart/nebuia-embs/main/install.sh | bash"
echo "  Rebuild only:      cd $PROJECT_DIR && cargo build --release --features cuda"
echo ""
