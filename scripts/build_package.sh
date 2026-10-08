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

rm -rf build dist src/vraven.egg-info
mkdir -p dist
"$PYTHON_BIN" -m pip wheel . --no-deps --no-build-isolation -w dist

WHEEL="$(ls -1t dist/vraven-*.whl | head -n 1)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
"$PYTHON_BIN" -m pip install --no-deps --target "$TMP_DIR" "$WHEEL"
(
  cd "$TMP_DIR"
  PYTHONPATH="$TMP_DIR" "$PYTHON_BIN" "$ROOT_DIR/scripts/wheel_smoke.py"
)

echo "Built and validated $WHEEL"
