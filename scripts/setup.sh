#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
JZ_MODULE="pytorch-gpu/py3/2.8.0"
PYTHON_BIN="python"
DO_CHECK=0
USE_JZ_MODULE=0

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1

usage() {
  cat <<'USAGE'
Usage: bash scripts/setup.sh [--env-name NAME] [--venv-dir DIR] [--use-jz-module] [--jz-module NAME] [--check]

Options:
  --env-name NAME   Convenience alias for --venv-dir "./.venv-NAME"
  --venv-dir DIR    Virtual environment directory (default: ./.venv)
  --use-jz-module   Load Jean Zay PyTorch module and skip pip torch install
  --jz-module NAME  Module to load with --use-jz-module (default: pytorch-gpu/py3/2.8.0)
  --check           Run unit tests after setup
USAGE
}

die() {
  echo "[setup] $*" >&2
  exit 1
}

need_value() {
  [[ $# -ge 2 ]] || die "$1 requires a value"
}

run_python() {
  PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" "$@"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      need_value "$@"
      VENV_DIR="${PROJECT_ROOT}/.venv-$2"
      shift 2
      ;;
    --venv-dir)
      need_value "$@"
      VENV_DIR="$2"
      shift 2
      ;;
    --use-jz-module)
      USE_JZ_MODULE=1
      shift
      ;;
    --jz-module)
      need_value "$@"
      JZ_MODULE="$2"
      shift 2
      ;;
    --check)
      DO_CHECK=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

if [[ "${USE_JZ_MODULE}" == "1" ]]; then
  type module >/dev/null 2>&1 || die "--use-jz-module requested but 'module' is unavailable"
  module purge || true
  module load "${JZ_MODULE}"
fi

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "python command not found"

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
  echo "[setup] Upgrading pip"
  "${PYTHON_BIN}" -m pip install --upgrade "pip<25.1" --no-input
else
  echo "[setup] Upgrading build tooling"
  "${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel --no-input
fi

echo "[setup] Installing project packages with benchmark extras"
"${PYTHON_BIN}" -m pip install -e "${PROJECT_ROOT}[dev,bench]" --no-input

echo "[setup] Verifying imports"
run_python - <<'PY'
import datakit
import genkit
import pandas
import torch
from labkit.config import load_config

print(f"imports_ok torch={torch.__version__} pandas={pandas.__version__}")
PY

if [[ "${DO_CHECK}" == "1" ]]; then
  echo "[setup] Running unit tests"
  run_python -m pytest -q "${PROJECT_ROOT}/tests"
fi

cat <<NEXT

[setup] Done.
To use this environment in your current shell:
  source "${VENV_DIR}/bin/activate"
  export PYTHONPATH=${PROJECT_ROOT}/src\${PYTHONPATH:+:\$PYTHONPATH}

NEXT
