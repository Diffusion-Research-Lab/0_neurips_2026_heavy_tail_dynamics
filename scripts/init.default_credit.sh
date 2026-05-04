#!/usr/bin/env bash
# Pre-stage Default-of-Credit-Card-Clients OpenML data on a login node.
# Compute nodes have no internet; run this before any Slurm preprocessing
# job that touches the default_credit dataset.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash scripts/init.default_credit.sh [--root DIR]

Options:
  --root DIR   sklearn data_home (default: $FLOWBENCH_DATA/raw/scikit_learn)
USAGE
}

ROOT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[init.default_credit] unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${ROOT}" ]]; then
  if [[ -z "${FLOWBENCH_DATA:-}" ]]; then
    echo "[init.default_credit] FLOWBENCH_DATA must be set, or pass --root <data_home>." >&2
    exit 1
  fi
  ROOT="${FLOWBENCH_DATA}/raw/scikit_learn"
fi

mkdir -p "${ROOT}"
echo "[init.default_credit] data_home: ${ROOT}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.venv/bin/activate"
fi

python - <<PY
from sklearn.datasets import fetch_openml
print("[init.default_credit] fetching OpenML 42477 into ${ROOT} ...", flush=True)
fetch_openml(data_id=42477, as_frame=True, parser="pandas", data_home="${ROOT}")
print("[init.default_credit] done", flush=True)
PY
