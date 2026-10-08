#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.14 python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 \
      && "$candidate" -c 'import sys, setuptools.build_meta; raise SystemExit(sys.version_info < (3, 10))' 2>/dev/null; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "VRAVEN requires Python 3.10 or newer." >&2
  exit 2
fi
PYTHON_BIN="$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"

"$PYTHON_BIN" scripts/generate_code_inventory.py
"$PYTHON_BIN" -m compileall -q src tests examples benchmarks scripts

if command -v ruff >/dev/null 2>&1; then
  ruff check .
else
  "$PYTHON_BIN" -m ruff check .
fi

PYTHONPATH=src pytest -q
bash scripts/build_package.sh

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
for example in quickstart visual_modes attention_demo causal_demo dataset_comparison visual_gallery; do
  echo "Validating example: $example"
  (
    cd "$TMP_DIR"
    PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" "$ROOT_DIR/examples/$example.py" >/dev/null
  )
done

(
  cd "$TMP_DIR"
  PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" "$ROOT_DIR/benchmarks/benchmark_capture.py" \
    --warmup 1 --repeats 3 --batch 8 --width 64
)

echo "VRAVEN release validation completed successfully"
