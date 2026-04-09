#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash benchmarks/01_setup.sh
#   bash benchmarks/01_setup.sh --env-name genkit --check
#   bash benchmarks/01_setup.sh --venv-dir /path/to/.venv --check
#   bash benchmarks/01_setup.sh --use-jz-module --check
#
# With --check, this script runs the unit tests. The benchmark smoke pipeline is
# provided separately via benchmarks/03_launcher.sh --blank.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHONPATH_VALUE="${PROJECT_ROOT}/src"
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
  --check           Run unit tests after setup
USAGE
      exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2 ;;
  esac
done

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
PYTHON_BIN="$(command -v python)"

"${PYTHON_BIN}" - <<'PY'
import sys
print(f"[setup] Using Python {sys.version.split()[0]}")
PY

if [[ "${USE_JZ_MODULE}" == "1" ]]; then
  echo "[setup] Upgrading pip (module mode, pinned to avoid egg incompatibility)"
  "${PYTHON_BIN}" -m pip install --upgrade "pip<25.1" --no-input
else
  echo "[setup] Upgrading build tooling"
  "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel --no-input
fi

echo "[setup] Installing project packages with benchmark extras"
"${PYTHON_BIN}" -m pip install -e "${PROJECT_ROOT}[dev,bench]" --no-input

echo "[setup] Verifying imports (core + benchmark deps)"
PYTHONPATH="${PYTHONPATH_VALUE}${PYTHONPATH:+:${PYTHONPATH}}" \
"${PYTHON_BIN}" - <<'PY'
import genkit
import pandas
import torch
from labkit.config import load_config
print(f"imports_ok torch={torch.__version__} pandas={pandas.__version__}")
PY

if [[ "${DO_CHECK}" == "1" ]]; then
  echo "[setup] Running unit tests"
  PYTHONPATH="${PYTHONPATH_VALUE}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m pytest -q "${PROJECT_ROOT}/tests"
fi

cat <<NEXT

[setup] Done.
To use this environment in your current shell:
  source "${VENV_DIR}/bin/activate"
  export PYTHONPATH=${PYTHONPATH_VALUE}\${PYTHONPATH:+:\$PYTHONPATH}

NEXT
