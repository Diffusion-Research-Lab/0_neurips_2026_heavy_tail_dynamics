#!/usr/bin/env bash
# Pre-stage Clauset earthquake magnitudes on a login node.
# Compute nodes have no internet; run this before Slurm preprocessing jobs.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash scripts/init.earthquakes.sh [--root DIR]

Options:
  --root DIR   target directory (default: $FLOWBENCH_DATA/raw/powerlaws)
USAGE
}

URL="https://aaronclauset.github.io/powerlaws/data/quakes.txt"
FILE="quakes.txt"
ROOT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[init.earthquakes] unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${ROOT}" ]]; then
  if [[ -z "${FLOWBENCH_DATA:-}" ]]; then
    echo "[init.earthquakes] FLOWBENCH_DATA must be set, or pass --root <dir>." >&2
    exit 1
  fi
  ROOT="${FLOWBENCH_DATA}/raw/powerlaws"
fi

mkdir -p "${ROOT}"
DEST="${ROOT}/${FILE}"
if [[ -s "${DEST}" ]]; then
  echo "[init.earthquakes] already present: ${DEST}"
  exit 0
fi

echo "[init.earthquakes] downloading ${URL} -> ${DEST}"
if command -v curl >/dev/null 2>&1; then
  curl -fsSL "${URL}" -o "${DEST}.tmp"
elif command -v wget >/dev/null 2>&1; then
  wget -q "${URL}" -O "${DEST}.tmp"
else
  echo "[init.earthquakes] need curl or wget on the login node." >&2
  exit 1
fi
mv "${DEST}.tmp" "${DEST}"
echo "[init.earthquakes] done"
