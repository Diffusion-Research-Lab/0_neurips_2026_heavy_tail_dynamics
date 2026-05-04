#!/usr/bin/env bash
# Idempotently stage CIFAR-100 raw files for the cifar100_lt loader.
# Creates $FLOWBENCH_DATA/raw/cifar100_lt and symlinks the Jean Zay
# cifar-100-python directory into it. Run on a login node.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash scripts/init.cifar100_lt.sh [--root DIR] [--source DIR]

Options:
  --root DIR     CIFAR-100-LT root (default: $FLOWBENCH_DATA/raw/cifar100_lt)
  --source DIR   CIFAR-100 source directory (default: /lustre/fsmisc/dataset/cifar-100-python)
USAGE
}

ROOT=""
SRC="/lustre/fsmisc/dataset/cifar-100-python"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)   ROOT="$2"; shift 2 ;;
    --source) SRC="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[init.cifar100_lt] unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${ROOT}" ]]; then
  if [[ -z "${FLOWBENCH_DATA:-}" ]]; then
    echo "[init.cifar100_lt] FLOWBENCH_DATA must be set, or pass --root <cifar100_lt_root>." >&2
    exit 1
  fi
  ROOT="${FLOWBENCH_DATA}/raw/cifar100_lt"
fi

mkdir -p "${ROOT}"
echo "[init.cifar100_lt] root: ${ROOT}"

LINK="${ROOT}/cifar-100-python"
if [[ -L "${LINK}" || -e "${LINK}" ]]; then
  echo "[init.cifar100_lt] cifar-100-python already present at ${LINK}"
else
  if [[ ! -d "${SRC}" ]]; then
    echo "[init.cifar100_lt] source directory not found: ${SRC}" >&2
    echo "[init.cifar100_lt] place CIFAR-100 pickle files under ${ROOT}/cifar-100-python/" >&2
    exit 1
  fi
  ln -s "${SRC}" "${LINK}"
  echo "[init.cifar100_lt] symlinked ${SRC} -> ${LINK}"
fi

for required in train test meta; do
  if [[ ! -f "${LINK}/${required}" ]]; then
    echo "[init.cifar100_lt] expected file missing: ${LINK}/${required}" >&2
    exit 1
  fi
done
echo "[init.cifar100_lt] done"
