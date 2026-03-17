#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash benchmarks/01_setup.sh
#   bash benchmarks/01_setup.sh --env-name genkit --check
#   bash benchmarks/01_setup.sh --venv-dir /path/to/.venv --check
#   bash benchmarks/01_setup.sh --use-jz-module --check

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BENCH_REQ_FILE="${PROJECT_ROOT}/benchmarks/requirements.txt"
BENCH_DATA_DIR="${PROJECT_ROOT}/benchmarks/_data"
BENCH_WEIGHTS_DIR="${PROJECT_ROOT}/benchmarks/_weights"
VENV_DIR="${PROJECT_ROOT}/.venv"
DO_CHECK=0
USE_JZ_MODULE=0
JZ_MODULE="pytorch-gpu/py3/2.8.0"
PYTHON_BIN="python"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1

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

if [[ "${USE_JZ_MODULE}" == "1" ]]; then
  echo "[setup] Upgrading pip (module mode, pinned to avoid egg incompatibility)"
  python -m pip install --upgrade "pip<25.1" --no-input
else
  echo "[setup] Upgrading build tooling"
  python -m pip install --upgrade pip setuptools wheel --no-input
fi

TMP_BENCH_REQ="$(mktemp)"
cleanup_tmp() {
  rm -f "${TMP_BENCH_REQ}"
}
trap cleanup_tmp EXIT

cp "${BENCH_REQ_FILE}" "${TMP_BENCH_REQ}"

echo "[setup] Installing benchmark requirements"
pip install -r "${TMP_BENCH_REQ}" --no-input

echo "[setup] Installing project packages (genkit + labkit)"
pip install -e "${PROJECT_ROOT}[dev]" --no-input

echo "[setup] Verifying imports (genkit + labkit + torchvision)"
PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
python - <<'PY'
import genkit
import torch
import torchvision
from labkit.config import load_config
print(f"imports_ok torch={torch.__version__} torchvision={torchvision.__version__}")
PY

echo "[setup] Prefetching benchmark assets (_data and _weights)"
mkdir -p "${BENCH_DATA_DIR}" "${BENCH_WEIGHTS_DIR}"
PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
BENCH_DATA_DIR="${BENCH_DATA_DIR}" \
BENCH_WEIGHTS_DIR="${BENCH_WEIGHTS_DIR}" \
python - <<'PY'
import os
import torch
from torchvision import datasets, models

data_dir = os.environ["BENCH_DATA_DIR"]
weights_dir = os.environ["BENCH_WEIGHTS_DIR"]
torch.hub.set_dir(weights_dir)

datasets.CIFAR100(root=data_dir, train=True, download=True)
models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
print(f"assets_ok data={data_dir} weights={weights_dir}")
PY

if [[ "${DO_CHECK}" == "1" ]]; then
  echo "[setup] Running unit tests"
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  pytest -q "${PROJECT_ROOT}/tests"

  echo "[setup] Running genkit example"
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  python "${PROJECT_ROOT}/examples/spiral_example.py" --smoke

  echo "[setup] Running benchmark blank pipeline"
  bash "${SCRIPT_DIR}/03_launcher.sh" --blank
fi

cat <<NEXT

[setup] Done.
To use this environment in your current shell:
  source "${VENV_DIR}/bin/activate"
  export PYTHONPATH=${PROJECT_ROOT}/src\${PYTHONPATH:+:\$PYTHONPATH}

NEXT
