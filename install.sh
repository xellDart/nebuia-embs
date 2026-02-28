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
#   curl -fsSL ... | bash -s -- --cpu
#   curl -fsSL ... | bash -s -- --dir /opt/nebuia-embs
#
# What it does:
#   1. Installs Rust (if not found)
#   2. Clones nebuia-embs (or updates if already present)
#   3. Clones Crane dependency (ColQwen3 embeddings engine)
#   4. Auto-detects CUDA, GPU, Flash Attention
#   5. Builds with optimal features (flash-attn on SM_80+ GPUs)
#   6. Installs nebuia-ctl management CLI
# ─────────────────────────────────────────────────────────────────────

REPO_URL="https://github.com/xellDart/nebuia-embs.git"
CRANE_REPO_URL="https://github.com/xellDart/Crane.git"
CRANE_BRANCH="feat/colqwen3-embeddings"
BRANCH="main"
INSTALL_DIR=""
FORCE_CPU=false

for arg in "$@"; do
  case "$arg" in
    --dir=*)        INSTALL_DIR="${arg#--dir=}" ;;
    --dir)          ;; # handled below
    --branch=*)     BRANCH="${arg#--branch=}" ;;
    --cpu)          FORCE_CPU=true ;;
    -h|--help)
      cat <<'HELP'
nebuia-embs Installer — document embedding service powered by ColQwen3

Usage:
  curl -fsSL https://raw.githubusercontent.com/xellDart/nebuia-embs/main/install.sh | bash
  curl -fsSL ... | bash -s -- [OPTIONS]

Options:
  --dir <path>          Install directory (default: ./nebuia-embs)
  --branch <name>       Git branch to checkout (default: main)
  --cpu                 Force CPU-only build (skip CUDA detection)
  -h, --help            Show this help

Examples:
  # Auto-detect GPU and build
  curl -fsSL ... | bash

  # CPU-only build
  curl -fsSL ... | bash -s -- --cpu

  # Install to custom directory
  curl -fsSL ... | bash -s -- --dir /opt/nebuia-embs

After installation:
  1. cp .env.example .env    # configure database, S3, model path
  2. nebuia-ctl start        # start as daemon
  3. nebuia-ctl status       # check health
HELP
      exit 0
      ;;
    *)
      if [ "${PREV_ARG:-}" = "--dir" ]; then INSTALL_DIR="$arg"; fi
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

# ── Build essentials (cc, pkg-config, libssl-dev) ──
if ! command -v cc &>/dev/null && ! command -v gcc &>/dev/null; then
  info "C compiler not found. Installing build dependencies..."
  if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq && sudo apt-get install -y -qq build-essential pkg-config libssl-dev
  elif command -v dnf &>/dev/null; then
    sudo dnf install -y gcc gcc-c++ make pkg-config openssl-devel
  elif command -v pacman &>/dev/null; then
    sudo pacman -Sy --noconfirm base-devel pkg-config openssl
  else
    fail "C compiler (cc/gcc) not found. Install build-essential or equivalent for your distro."
  fi
  ok "Build dependencies installed"
else
  # Ensure pkg-config and libssl-dev are present even if cc exists
  if ! command -v pkg-config &>/dev/null; then
    info "pkg-config not found. Installing..."
    if command -v apt-get &>/dev/null; then
      sudo apt-get update -qq && sudo apt-get install -y -qq pkg-config libssl-dev
    elif command -v dnf &>/dev/null; then
      sudo dnf install -y pkg-config openssl-devel
    elif command -v pacman &>/dev/null; then
      sudo pacman -Sy --noconfirm pkg-config openssl
    fi
  fi
  ok "C compiler: $(cc --version 2>/dev/null | head -1 || gcc --version 2>/dev/null | head -1)"
fi

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
  # Detect GPU via nvidia-smi first (works even without CUDA Toolkit)
  if command -v nvidia-smi &>/dev/null; then
    GPU_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/^[[:space:]]*//')
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')

    if [ -n "$GPU_NAME" ]; then
      ok "GPU: $GPU_NAME ($GPU_MEM, sm_${GPU_COMPUTE_CAP//./_})"
    fi
  fi

  # Try to find nvcc: check PATH, then common install locations
  NVCC_BIN=""
  if command -v nvcc &>/dev/null; then
    NVCC_BIN="$(command -v nvcc)"
  else
    for candidate in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc /opt/cuda/bin/nvcc; do
      if [ -x "$candidate" ] 2>/dev/null; then
        NVCC_BIN="$candidate"
        CUDA_BIN_DIR="$(dirname "$NVCC_BIN")"
        export PATH="$CUDA_BIN_DIR:$PATH"
        info "Found nvcc at $NVCC_BIN (added to PATH)"
        break
      fi
    done
  fi

  # If GPU detected but no CUDA Toolkit, install it
  if [ -z "$NVCC_BIN" ] && [ -n "$GPU_NAME" ]; then
    info "NVIDIA GPU detected but CUDA Toolkit (nvcc) not found. Installing..."
    if command -v apt-get &>/dev/null; then
      sudo apt-get update -qq && sudo apt-get install -y -qq nvidia-cuda-toolkit
    elif command -v dnf &>/dev/null; then
      sudo dnf install -y cuda-toolkit
    elif command -v pacman &>/dev/null; then
      sudo pacman -Sy --noconfirm cuda
    else
      fail "Could not auto-install CUDA Toolkit. Install manually: https://developer.nvidia.com/cuda-downloads"
    fi
    # Find nvcc after install
    if command -v nvcc &>/dev/null; then
      NVCC_BIN="$(command -v nvcc)"
    else
      for candidate in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
        if [ -x "$candidate" ] 2>/dev/null; then
          NVCC_BIN="$candidate"
          export PATH="$(dirname "$NVCC_BIN"):$PATH"
          break
        fi
      done
    fi
  fi

  if [ -n "$NVCC_BIN" ]; then
    CUDA_VERSION=$("$NVCC_BIN" --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    ok "CUDA Toolkit: $CUDA_VERSION ($(dirname "$NVCC_BIN"))"
    HAS_CUDA=true

    if [ -n "$GPU_COMPUTE_CAP" ]; then
      MAJOR=$(echo "$GPU_COMPUTE_CAP" | cut -d. -f1)
      if [ "$MAJOR" -ge 8 ] 2>/dev/null; then
        HAS_FLASH_ATTN=true
        ok "Flash Attention: supported (SM_80+)"
      fi
    fi
  elif [ -n "$GPU_NAME" ]; then
    warn "CUDA Toolkit installation failed. Building for CPU only."
    warn "Install manually: https://developer.nvidia.com/cuda-downloads"
  else
    warn "No NVIDIA GPU detected. Building for CPU only."
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
#  Step 2b: Clone / Update Crane dependency
# ═════════════════════════════════════════════════════════════════════

CRANE_DIR="$(dirname "$PROJECT_DIR")/Crane"

if [ -d "$CRANE_DIR" ] && [ -d "$CRANE_DIR/.git" ]; then
  info "Crane directory exists at $CRANE_DIR, updating..."
  cd "$CRANE_DIR"
  git fetch origin "$CRANE_BRANCH" 2>/dev/null || true
  git checkout "$CRANE_BRANCH" 2>/dev/null || warn "Could not checkout $CRANE_BRANCH"
  git pull --ff-only 2>/dev/null || warn "Could not pull Crane. Continuing with local version."
  cd "$PROJECT_DIR"
else
  info "Cloning Crane (ColQwen3 embeddings engine)..."
  git clone --branch "$CRANE_BRANCH" --depth 1 "$CRANE_REPO_URL" "$CRANE_DIR"
fi

ok "Crane directory: $CRANE_DIR"
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
#  Summary
# ═════════════════════════════════════════════════════════════════════

echo -e "  ${BOLD}${CYAN}════════════════════════════════════════════════${RESET}"
echo -e "  ${BOLD}  nebuia-embs Installation Complete${RESET}"
echo -e "  ${BOLD}${CYAN}════════════════════════════════════════════════${RESET}"
echo ""
echo "  Location:    $PROJECT_DIR"
echo "  Crane:       $CRANE_DIR"
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
echo ""

# ── Next steps ──
echo -e "  ${BOLD}Next steps:${RESET}"
echo ""
if [ ! -f "$PROJECT_DIR/.env" ]; then
  echo "    1. cp .env.example .env     # configure database, S3, model path"
  echo "    2. nebuia-ctl start         # start as daemon"
  echo "    3. nebuia-ctl status        # check health"
else
  echo "    1. nebuia-ctl start         # start as daemon"
  echo "    2. nebuia-ctl status        # check health"
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

echo "  Re-run installer:  curl -fsSL https://raw.githubusercontent.com/xellDart/nebuia-embs/main/install.sh | bash"
echo "  Rebuild only:      cd $PROJECT_DIR && cargo build --release --features flash-attn"
echo ""
