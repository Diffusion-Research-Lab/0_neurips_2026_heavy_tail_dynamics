#!/usr/bin/env bash
# Idempotently stage ImageNet-LT raw files for the imagenet_lt loader.
# Copies the bundled annotation files from datasets/imagenet_lt/annotations/
# into $FLOWBENCH_DATA/raw/imagenet_lt/annotations/ and symlinks the Jean Zay
# ImageNet image tree under the same root. Run on a login node.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash scripts/init.imagenet_lt.sh [--root DIR] [--source DIR]

Options:
  --root DIR     ImageNet-LT root (default: $FLOWBENCH_DATA/raw/imagenet_lt)
  --source DIR   ImageNet image tree (default: /lustre/fswork/dataset/imagenet)
USAGE
}

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ANN_SRC="${REPO_ROOT}/datasets/imagenet_lt/annotations"
ROOT=""
IMG_SRC="/lustre/fswork/dataset/imagenet"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)   ROOT="$2"; shift 2 ;;
    --source) IMG_SRC="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[init.imagenet_lt] unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${ROOT}" ]]; then
  if [[ -z "${FLOWBENCH_DATA:-}" ]]; then
    echo "[init.imagenet_lt] FLOWBENCH_DATA must be set, or pass --root <imagenet_lt_root>." >&2
    exit 1
  fi
  ROOT="${FLOWBENCH_DATA}/raw/imagenet_lt"
fi

mkdir -p "${ROOT}/annotations"
echo "[init.imagenet_lt] root: ${ROOT}"

for name in ImageNet_LT_train.txt ImageNet_LT_val.txt ImageNet_LT_test.txt; do
  src="${ANN_SRC}/${name}"
  dst="${ROOT}/annotations/${name}"
  if [[ ! -f "${src}" ]]; then
    echo "[init.imagenet_lt] bundled annotation missing: ${src}" >&2
    exit 1
  fi
  if [[ -f "${dst}" ]]; then
    echo "[init.imagenet_lt] annotation already present: ${dst}"
    continue
  fi
  cp "${src}" "${dst}"
  echo "[init.imagenet_lt] copied ${src} -> ${dst}"
done

LINK="${ROOT}/imagenet"
if [[ -L "${LINK}" || -e "${LINK}" ]]; then
  echo "[init.imagenet_lt] imagenet tree already present at ${LINK}"
else
  if [[ ! -d "${IMG_SRC}" ]]; then
    echo "[init.imagenet_lt] image source directory not found: ${IMG_SRC}" >&2
    echo "[init.imagenet_lt] symlink (or place) the ImageNet tree at ${LINK}" >&2
    exit 1
  fi
  ln -s "${IMG_SRC}" "${LINK}"
  echo "[init.imagenet_lt] symlinked ${IMG_SRC} -> ${LINK}"
fi
echo "[init.imagenet_lt] done"
