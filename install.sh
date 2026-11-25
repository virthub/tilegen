#!/usr/bin/env bash
#
# TileGen deterministic `uv` installer.
#
# Bootstraps `uv`, provisions a project-local `.venv/` (Python 3.10-3.12),
# syncs locked dependencies, and compiles the zero-compiler-dependency C++
# runtime (`libtilegen_runtime`) plus the optional PyTorch extension
# (`tilegen._C`) via scikit-build-core + CMake.
#
# Flags:
#   --backend <cuda|rocm|ascend>   target backend (default: cuda)
#   --runtime-only                 minimal serving deps (no dev/compiler extras)
#   --with-tilelang                add the TileLang compiler extra
#   --with-triton-tle              add the Triton-TLE extra
#   --with-evolve                  add the self-evolution (numpy) extra
#   --with-vllm-pspkv              fetch the vllm-pspkv fork (vLLM 0.34 + PSP-KV)
#                                  into third_party/vllm-pspkv (git clone, or a
#                                  local copy via PSPKV_SOURCE_DIR)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

BACKEND="cuda"
RUNTIME_ONLY=0
WITH_TILELANG=0
WITH_TRITON_TLE=0
WITH_EVOLVE=0
WITH_VLLM_PSPKV=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend) BACKEND="${2:-cuda}"; shift 2 ;;
    --runtime-only) RUNTIME_ONLY=1; shift ;;
    --with-tilelang) WITH_TILELANG=1; shift ;;
    --with-triton-tle) WITH_TRITON_TLE=1; shift ;;
    --with-evolve) WITH_EVOLVE=1; shift ;;
    --with-vllm-pspkv) WITH_VLLM_PSPKV=1; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

export UV_CACHE_DIR="${UV_CACHE_DIR:-${TMPDIR:-/tmp}/tilegen-uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${TMPDIR:-/tmp}/tilegen-uv-python}"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"

# --- 0. Optional third-party sources (removed by `./run.sh --clean`) ---------
if [[ "$WITH_VLLM_PSPKV" -eq 1 ]]; then
  THIRD_PARTY="$HERE/third_party"
  DEST="$THIRD_PARTY/vllm-pspkv"
  mkdir -p "$THIRD_PARTY"
  if [[ -n "${PSPKV_SOURCE_DIR:-}" && -d "$PSPKV_SOURCE_DIR" ]]; then
    echo "[install] copying vllm-pspkv from ${PSPKV_SOURCE_DIR} ..."
    rsync -a --delete \
      --exclude '.git' --exclude 'build' --exclude '.venv*' \
      --exclude '__pycache__' --exclude '*.pyc' \
      "$PSPKV_SOURCE_DIR"/ "$DEST"/
  elif [[ -d "$DEST/.git" ]]; then
    echo "[install] vllm-pspkv already present at ${DEST}"
  elif command -v git >/dev/null 2>&1; then
    echo "[install] cloning vllm-pspkv (https://github.com/virthub/vllm-pspkv) ..."
    if ! git clone --depth 1 https://github.com/virthub/vllm-pspkv.git "$DEST"; then
      echo "[install] WARNING: vllm-pspkv clone failed (offline?); set PSPKV_SOURCE_DIR=<local copy> and re-run" >&2
    fi
  else
    echo "[install] WARNING: git not found; set PSPKV_SOURCE_DIR=<local copy> to provision vllm-pspkv" >&2
  fi
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[install] bootstrapping uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "[install] using uv $(uv --version)"

PY_VERSION="${TILEGEN_PYTHON_VERSION:-3.12}"
if [[ ! -d .venv ]]; then
  echo "[install] creating .venv (python ${PY_VERSION}) ..."
  uv venv --python "$PY_VERSION" .venv
fi
export VIRTUAL_ENV="$HERE/.venv"
export PATH="$HERE/.venv/bin:$PATH"

echo "[install] installing build tooling ..."
uv pip install --python "$HERE/.venv/bin/python" scikit-build-core cmake ninja pyyaml

EXTRAS=""
if [[ "$RUNTIME_ONLY" -eq 0 ]]; then
  uv pip install --python "$HERE/.venv/bin/python" pytest numpy
fi
if [[ "$WITH_TILELANG" -eq 1 ]]; then EXTRAS="$EXTRAS,tilelang"; fi
if [[ "$WITH_TRITON_TLE" -eq 1 ]]; then EXTRAS="$EXTRAS,triton-tle"; fi
if [[ "$WITH_EVOLVE" -eq 1 ]]; then EXTRAS="$EXTRAS,evolve"; fi

echo "[install] building native runtime (backend=${BACKEND}) ..."
if [[ "$RUNTIME_ONLY" -eq 1 ]]; then
  CMAKE_ARGS="-DTILEGEN_BUILD_TESTS=OFF -DTILEGEN_BUILD_TORCH_EXT=OFF" \
    uv pip install --python "$HERE/.venv/bin/python" --no-build-isolation --no-deps -e .
else
  uv pip install --python "$HERE/.venv/bin/python" --no-build-isolation --no-deps -e .
fi

echo "[install] smoke check ..."
"$HERE/.venv/bin/python" -c "import tilegen; print('tilegen', tilegen.__version__, 'OK')"

echo "[install] done. Use ./run.sh to compile/test/verify/bench."
