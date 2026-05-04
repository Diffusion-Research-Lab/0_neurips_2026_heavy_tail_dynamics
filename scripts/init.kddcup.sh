#!/usr/bin/env bash
# Pre-stage KDD Cup 99 raw data on a login node for the kddcup loader.
# Compute nodes have no internet; this script must run on a login/entry node
# before any Slurm preprocessing job that touches the kddcup dataset.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash scripts/init.kddcup.sh [--root DIR]

Options:
  --root DIR   sklearn data_home (default: $FLOWBENCH_DATA/raw/scikit_learn)
USAGE
}

ROOT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[init.kddcup] unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${ROOT}" ]]; then
  if [[ -z "${FLOWBENCH_DATA:-}" ]]; then
    echo "[init.kddcup] FLOWBENCH_DATA must be set, or pass --root <data_home>." >&2
    exit 1
  fi
  ROOT="${FLOWBENCH_DATA}/raw/scikit_learn"
fi

mkdir -p "${ROOT}"
echo "[init.kddcup] data_home: ${ROOT}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/.venv/bin/activate"
fi

python - <<PY
from sklearn.datasets import fetch_kddcup99
print("[init.kddcup] fetching kddcup99 (percent10) into ${ROOT} ...", flush=True)
fetch_kddcup99(as_frame=True, percent10=True, data_home="${ROOT}", download_if_missing=True)
print("[init.kddcup] done", flush=True)
PY
