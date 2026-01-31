#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found. Install uv or add it to PATH." >&2
  exit 1
fi

cd "${ROOT_DIR}"

uv run --no-project --python "${PYTHON_VERSION}" \
  --with "numpy>=2.2.0" \
  --with "pyarrow>=22.0.0" \
  --with "pyspark[connect]==4.1.0" \
  --with "rdkit>=2025.9.3" \
  -m exports.training.export_graph
