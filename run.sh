#!/usr/bin/env bash

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${TMPDIR:-/tmp}/tilegen-uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${TMPDIR:-/tmp}/tilegen-uv-python}"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"

SUB=""
ARGS=()
CLEAN_ALL=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --compile)       SUB="compile"; shift ;;
    --autotune)      SUB="autotune"; shift ;;
    --evolve)        SUB="evolve"; shift ;;
    --test)          SUB="test"; shift ;;
    --selftest)      SUB="selftest"; shift ;;
    --verify)        SUB="verify"; shift ;;
    --bench)         SUB="bench"; shift ;;
    --benchmark)     SUB="benchmark"; BENCH_NAME="${2:?}"; shift 2 ;;
    --clean)         SUB="clean"; shift ;;
    --clean-all)     SUB="clean"; CLEAN_ALL=1; shift ;;
    --backend-ir)    ARGS+=("--backend-ir" "${2:?}"); shift 2 ;;
    --conf)          ARGS+=("--conf" "${2:?}"); shift 2 ;;
    --strategy)      ARGS+=("--strategy" "${2:?}"); shift 2 ;;
    --variant)       ARGS+=("--variant" "${2:?}"); shift 2 ;;
    --dump-ir)       ARGS+=("--dump-ir" "${2:?}"); shift 2 ;;
    --emit)          ARGS+=("--emit" "${2:?}"); shift 2 ;;
    --causal)        ARGS+=("--causal"); shift ;;
    --batch)         ARGS+=("--batch" "${2:?}"); shift 2 ;;
    --heads)         ARGS+=("--heads" "${2:?}"); shift 2 ;;
    --seq-len)       ARGS+=("--seq-len" "${2:?}"); shift 2 ;;
    --head-dim)      ARGS+=("--head-dim" "${2:?}"); shift 2 ;;
    --iters)         ARGS+=("--iters" "${2:?}"); shift 2 ;;
    --fmt)           ARGS+=("--fmt" "${2:?}"); shift 2 ;;
    --ctx)           ARGS+=("--ctx" "${2:?}"); shift 2 ;;
    --speed-gate)    ARGS+=("--speed-gate" "${2:?}"); shift 2 ;;
    --out)           ARGS+=("--out" "${2:?}"); shift 2 ;;
    --verbose)       ARGS+=("--verbose"); shift ;;
    -h|--help)       SUB="help"; shift ;;
    *) echo "[run] unknown flag: $1" >&2; exit 2 ;;
  esac
done

if [[ "$SUB" == "clean" ]]; then
  echo "[run] cleaning dynamically generated files ..."
  rm -rf "$HERE"/build "$HERE"/.tilegen_cache "$HERE"/.pytest_cache
  rm -rf "$HERE"/third_party
  find "$HERE" -type d -name __pycache__ -prune -exec rm -rf {} + || true
  find "$HERE" -type f -name '*.pyc' -delete || true
  if [[ "$CLEAN_ALL" -eq 1 ]]; then
    rm -rf "$HERE"/.venv
    echo "[run] removed .venv/ (the next ./run.sh call auto-reinstalls --runtime-only)"
  fi
  echo "[run] cleaned: build/ .tilegen_cache/ .pytest_cache/ third_party/ __pycache__/ *.pyc"
  exit 0
fi

if [[ -z "$SUB" ]]; then
  echo "usage: ./run.sh --compile|--autotune|--evolve|--test|--selftest|--verify|--bench|--benchmark <script>|--clean|--clean-all [opts]" >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[run] uv not found; installing ..." >&2
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ ! -d .venv ]]; then
  echo "[run] .venv missing; running ./install.sh --runtime-only ..." >&2
  ./install.sh --runtime-only
fi

PY="$HERE/.venv/bin/python"

if [[ "$SUB" == "benchmark" ]]; then
  script="$HERE/benchmarks/${BENCH_NAME}.py"
  if [[ "${BENCH_NAME}" == _* || ! -f "$script" ]]; then
    echo "[run] benchmark script not found: benchmarks/${BENCH_NAME}.py (names must not start with '_')" >&2
    exit 2
  fi
  BENCH_PY="${TILEGEN_BENCH_PYTHON:-$PY}"
  BENCH_PP="${TILEGEN_BENCH_PYTHONPATH:-$HERE/src}"
  exec env PYTHONPATH="$BENCH_PP" "$BENCH_PY" "$script" "${ARGS[@]}"
fi

if [[ "$SUB" == "help" ]]; then
  exec "$PY" -m tilegen.cli --help
fi

exec uv run --no-sync --python "$PY" python -m tilegen.cli "$SUB" "${ARGS[@]}"
