#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: create_training_sets.sh <export_graph.py|export_morgan_fp.py>

Examples:
  ./exports/create_training_sets.sh export_graph.py
  ./exports/create_training_sets.sh export_morgan_fp.py

Set SPARK_CONNECT_URL to override sc://localhost:15002.
EOF
}

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found. Install uv or add it to PATH." >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

target="${1}"
name="$(basename "${target}")"
name="${name%.py}"
module="exports.training.${name}"

cd "${ROOT_DIR}"

uv_args=(--no-project --python 3.10)
deps=(
  "numpy>=2.2.0"
  "pyarrow>=22.0.0"
  "pyspark[connect]==4.1.0"
  "rdkit>=2025.9.3"
)
for dep in "${deps[@]}"; do
  uv_args+=(--with "${dep}")
done

uv run "${uv_args[@]}" -m "${module}"
