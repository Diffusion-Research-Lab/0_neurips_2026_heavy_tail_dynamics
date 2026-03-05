#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash benchmarks/01_setup.sh
#   bash benchmarks/01_setup.sh --env-name cauda --check
#   bash benchmarks/01_setup.sh --venv-dir /path/to/.venv --check
#   bash benchmarks/01_setup.sh --use-jz-module --check

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CAUDA_REQ_FILE="${PROJECT_ROOT}/cauda/requirements.txt"
LABKIT_REQ_FILE="${PROJECT_ROOT}/labkit/requirements.txt"
BENCH_REQ_FILE="${PROJECT_ROOT}/benchmarks/requirements.txt"
VENV_DIR="${PROJECT_ROOT}/.venv"
DO_CHECK=0
USE_JZ_MODULE=0
JZ_MODULE="pytorch-gpu/py3/2.8.0"
PYTHON_BIN="python"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      VENV_DIR="${PROJECT_ROOT}/.venv-$2"; shift 2 ;;
    --venv-dir)
      VENV_DIR="$2"; shift 2 ;;
    --use-jz-module)
      USE_JZ_MODULE=1; shift ;;
    --jz-module)
      JZ_MODULE="$2"; shift 2 ;;
    --check)
      DO_CHECK=1; shift ;;
    -h|--help)
      cat <<USAGE
Usage: bash benchmarks/01_setup.sh [--env-name NAME] [--venv-dir DIR] [--use-jz-module] [--jz-module NAME] [--check]

Options:
  --env-name NAME   Convenience alias for --venv-dir "./.venv-NAME"
  --venv-dir DIR    Virtual environment directory (default: ./.venv)
  --use-jz-module   Load Jean Zay PyTorch module and skip pip torch install
  --jz-module NAME  Module to load with --use-jz-module (default: pytorch-gpu/py3/2.8.0)
  --check           Run tests + example + blank launcher after setup
USAGE
      exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2 ;;
  esac
done

if [[ ! -f "${CAUDA_REQ_FILE}" ]]; then
  echo "[setup] Missing file: ${CAUDA_REQ_FILE}" >&2
  exit 1
fi

if [[ ! -f "${LABKIT_REQ_FILE}" ]]; then
  echo "[setup] Missing file: ${LABKIT_REQ_FILE}" >&2
  exit 1
fi

if [[ ! -f "${BENCH_REQ_FILE}" ]]; then
  echo "[setup] Missing file: ${BENCH_REQ_FILE}" >&2
  exit 1
fi

if [[ "${USE_JZ_MODULE}" == "1" ]]; then
  if ! type module >/dev/null 2>&1; then
    echo "[setup] --use-jz-module requested but 'module' command is unavailable." >&2
    exit 1
  fi
  module purge || true
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[setup] python command not found." >&2
  exit 1
fi

if [[ "${USE_JZ_MODULE}" == "1" ]]; then
  module load "${JZ_MODULE}"
fi

echo "[setup] Creating/updating virtual environment at ${VENV_DIR}"
if [[ "${USE_JZ_MODULE}" == "1" ]]; then
  "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
else
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python - <<'PY'
import sys
print(f"[setup] Using Python {sys.version.split()[0]}")
PY

echo "[setup] Upgrading build tooling"
python -m pip install --upgrade pip setuptools wheel

TMP_CAUDA_REQ="$(mktemp)"
TMP_LABKIT_REQ="$(mktemp)"
TMP_BENCH_REQ="$(mktemp)"
cleanup_tmp() {
  rm -f "${TMP_CAUDA_REQ}" "${TMP_LABKIT_REQ}" "${TMP_BENCH_REQ}"
}
trap cleanup_tmp EXIT

if [[ "${USE_JZ_MODULE}" == "1" ]]; then
  awk 'BEGIN{IGNORECASE=1} !($0 ~ /^[[:space:]]*torch([[:space:]]*[<>=!~].*)?$/) {print}' "${CAUDA_REQ_FILE}" > "${TMP_CAUDA_REQ}"
  awk 'BEGIN{IGNORECASE=1} !($0 ~ /^[[:space:]]*torch([[:space:]]*[<>=!~].*)?$/) {print}' "${LABKIT_REQ_FILE}" > "${TMP_LABKIT_REQ}"
  awk 'BEGIN{IGNORECASE=1} !($0 ~ /^[[:space:]]*torch([[:space:]]*[<>=!~].*)?$/) {print}' "${BENCH_REQ_FILE}" > "${TMP_BENCH_REQ}"
else
  cp "${CAUDA_REQ_FILE}" "${TMP_CAUDA_REQ}"
  cp "${LABKIT_REQ_FILE}" "${TMP_LABKIT_REQ}"
  cp "${BENCH_REQ_FILE}" "${TMP_BENCH_REQ}"
fi

echo "[setup] Installing cauda requirements"
pip install -r "${TMP_CAUDA_REQ}"

echo "[setup] Installing labkit requirements"
pip install -r "${TMP_LABKIT_REQ}"

echo "[setup] Installing benchmark requirements"
pip install -r "${TMP_BENCH_REQ}"

echo "[setup] Installing cauda package"
pip install -e "${PROJECT_ROOT}/cauda" --no-deps

echo "[setup] Verifying imports (cauda + labkit + torchvision)"
PYTHONPATH="${PROJECT_ROOT}/cauda:${PROJECT_ROOT}/labkit${PYTHONPATH:+:${PYTHONPATH}}" \
python - <<'PY'
import cauda
import torch
import torchvision
from labkit.config import load_config
print(f"imports_ok torch={torch.__version__} torchvision={torchvision.__version__}")
PY

if [[ "${DO_CHECK}" == "1" ]]; then
  echo "[setup] Running unit tests"
  PYTHONPATH="${PROJECT_ROOT}/cauda:${PROJECT_ROOT}/labkit${PYTHONPATH:+:${PYTHONPATH}}" \
  pytest -q "${PROJECT_ROOT}/cauda/cauda/tests"

  echo "[setup] Running cauda example"
  PYTHONPATH="${PROJECT_ROOT}/cauda:${PROJECT_ROOT}/labkit${PYTHONPATH:+:${PYTHONPATH}}" \
  python "${PROJECT_ROOT}/cauda/examples/spiral_example.py" --smoke

  echo "[setup] Running benchmark blank pipeline"
  bash "${SCRIPT_DIR}/03_launcher.sh" --blank
fi

cat <<NEXT

[setup] Done.
To use this environment in your current shell:
  source "${VENV_DIR}/bin/activate"
  export PYTHONPATH="${PROJECT_ROOT}/cauda:${PROJECT_ROOT}/labkit\${PYTHONPATH:+:\$PYTHONPATH}"

NEXT
