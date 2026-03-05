#!/usr/bin/env bash

#SBATCH --job-name=heavyflow
#SBATCH --output=heavyflow_%j.out
#SBATCH --error=heavyflow_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --qos=qos_gpu-dev
#SBATCH -A jcx@v100

set -euo pipefail

# Usage:
#   sbatch benchmarks/02_launcher.slurm
#
# This script prepares the Slurm environment and delegates benchmark execution to
# benchmarks/03_launcher.sh --run.

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
JZ_MODULE="${JZ_MODULE:-pytorch-gpu/py3/2.8.0}"
if [[ -d "${SUBMIT_DIR}/cauda" && -d "${SUBMIT_DIR}/labkit" && -d "${SUBMIT_DIR}/benchmarks" ]]; then
  PROJECT_ROOT="${SUBMIT_DIR}"
  SCRIPT_DIR="${SUBMIT_DIR}/benchmarks"
elif [[ -d "${SUBMIT_DIR}/../cauda" && -d "${SUBMIT_DIR}/../labkit" ]]; then
  PROJECT_ROOT="$(cd -- "${SUBMIT_DIR}/.." && pwd)"
  SCRIPT_DIR="${SUBMIT_DIR}"
else
  echo "[slurm] Could not infer project root from SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
  echo "[slurm] Submit from repo root or benchmarks directory." >&2
  exit 1
fi
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv-cauda}"

if [[ ! -d "${PROJECT_ROOT}/cauda" || ! -d "${PROJECT_ROOT}/labkit" ]]; then
  echo "[slurm] Expected directories not found:" >&2
  echo "  ${PROJECT_ROOT}/cauda" >&2
  echo "  ${PROJECT_ROOT}/labkit" >&2
  exit 1
fi

if ! command -v module >/dev/null 2>&1; then
  echo "[slurm] 'module' command is required on Jean Zay." >&2
  exit 1
fi

module purge || true
conda deactivate 2>/dev/null || true

module load "${JZ_MODULE}"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
    VENV_DIR="${PROJECT_ROOT}/.venv"
  else
    echo "[slurm] Missing virtual environment." >&2
    echo "[slurm] Expected: ${PROJECT_ROOT}/.venv-cauda (or ${PROJECT_ROOT}/.venv)" >&2
    echo "[slurm] Run setup first on login node:" >&2
    echo "  bash benchmarks/01_setup.sh --env-name cauda --use-jz-module" >&2
    exit 1
  fi
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/cauda:${PROJECT_ROOT}/labkit${PYTHONPATH:+:$PYTHONPATH}"

if ! command -v python >/dev/null 2>&1; then
  echo "[slurm] python command not found after venv activation." >&2
  exit 1
fi

python - <<'PY'
import torch
import cauda
from labkit.config import load_config
print(f"[slurm] Runtime preflight OK - python/torch={torch.__version__}")
PY

CPUS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${CPUS}"
export MKL_NUM_THREADS="${CPUS}"
export OPENBLAS_NUM_THREADS="${CPUS}"
export NUMEXPR_NUM_THREADS="${CPUS}"

echo "=============================================================================="
echo "Heavy Tail Flow Benchmark Runner (GPU / Slurm)"
echo "SUBMIT_DIR:    ${SUBMIT_DIR}"
echo "PROJECT_ROOT:  ${PROJECT_ROOT}"
echo "VENV_DIR:      ${VENV_DIR}"
echo "HOST:          $(hostname)"
echo "Python:        $(python --version 2>&1)"
echo "SLURM_JOB_ID:  ${SLURM_JOB_ID:-<none>}"
echo "CPUs:          ${CPUS}"
echo "CUDA devices:  ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "=============================================================================="

srun nvidia-smi || true
srun bash "${SCRIPT_DIR}/03_launcher.sh" --run --cpus "${CPUS}"
