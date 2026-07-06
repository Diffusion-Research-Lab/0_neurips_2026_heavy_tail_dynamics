#!/usr/bin/env bash

#SBATCH --job-name=htfm_dataset
#SBATCH --output=htfm_dataset_%j.out
#SBATCH --error=htfm_dataset_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --account=jcx@v100

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
JZ_MODULE="${JZ_MODULE:-pytorch-gpu/py3/2.8.0}"
DATASET_SCRIPT_REL="scripts/prefetch.datasets.py"
HRRR_SCRIPT_REL="scripts/ensure_hrrr.py"
PYTHON_BIN=""
CLI_OVERWRITE=0
FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --overwrite)
      CLI_OVERWRITE=1
      shift
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        FORWARD_ARGS+=("$1")
        shift
      done
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -d "${SUBMIT_DIR}/src/datakit" && -d "${SUBMIT_DIR}/src/genkit" && -d "${SUBMIT_DIR}/src/labkit" && -d "${SUBMIT_DIR}/benchmarks" ]]; then
  PROJECT_ROOT="${SUBMIT_DIR}"
elif [[ -d "${SUBMIT_DIR}/../src/datakit" && -d "${SUBMIT_DIR}/../src/genkit" && -d "${SUBMIT_DIR}/../src/labkit" && -d "${SUBMIT_DIR}/../benchmarks" ]]; then
  PROJECT_ROOT="$(cd -- "${SUBMIT_DIR}/.." && pwd)"
else
  echo "[dataset-slurm] Could not infer project root from SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
  exit 1
fi

VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
DATASET_SCRIPT="${PROJECT_ROOT}/${DATASET_SCRIPT_REL}"
HRRR_SCRIPT="${PROJECT_ROOT}/${HRRR_SCRIPT_REL}"

if ! command -v module >/dev/null 2>&1; then
  echo "[dataset-slurm] 'module' command is required on Jean Zay." >&2
  exit 1
fi

module purge || true
conda deactivate 2>/dev/null || true
module load "${JZ_MODULE}"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  if [[ -f "${PROJECT_ROOT}/.venv-genkit/bin/activate" ]]; then
    VENV_DIR="${PROJECT_ROOT}/.venv-genkit"
  else
    echo "[dataset-slurm] Missing virtual environment." >&2
    echo "[dataset-slurm] Expected: ${PROJECT_ROOT}/.venv (or ${PROJECT_ROOT}/.venv-genkit)" >&2
    echo "[dataset-slurm] Run setup first on login node:" >&2
    echo "  make setup" >&2
    exit 1
  fi
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
PYTHON_BIN="$(command -v python)"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "${DATASET_SCRIPT}" ]]; then
  echo "[dataset-slurm] Missing dataset prefetch script: ${DATASET_SCRIPT}" >&2
  exit 1
fi

echo "=============================================================================="
echo "Heavy Tail Flow Benchmark Dataset Prefetch (Slurm)"
echo "PROJECT_ROOT:  ${PROJECT_ROOT}"
echo "VENV_DIR:      ${VENV_DIR}"
echo "HOST:          $(hostname)"
echo "Python:        $("${PYTHON_BIN}" --version 2>&1)"
echo "SLURM_JOB_ID:  ${SLURM_JOB_ID:-<none>}"
echo "DATASET_SCRIPT: ${DATASET_SCRIPT}"
echo "FORWARD_ARGS:  ${FORWARD_ARGS[*]:-<none>}"
echo "=============================================================================="

NEEDS_HRRR=0
for ((i = 0; i < ${#FORWARD_ARGS[@]}; i++)); do
  if [[ "${FORWARD_ARGS[$i]}" == "--only-dataset=hrrr" ]]; then
    NEEDS_HRRR=1
  fi
  if [[ "${FORWARD_ARGS[$i]}" == "--only-dataset" && "${FORWARD_ARGS[$((i + 1))]:-}" == "hrrr" ]]; then
    NEEDS_HRRR=1
  fi
done

if [[ "${NEEDS_HRRR}" -eq 1 ]]; then
  if [[ ! -f "${HRRR_SCRIPT}" ]]; then
    echo "[dataset-slurm] Missing HRRR ensure script: ${HRRR_SCRIPT}" >&2
    exit 1
  fi
  HRRR_STATUS="${SLURM_TMPDIR:-/tmp}/hrrr_ensure_${SLURM_JOB_ID:-$$}.json"
  HRRR_CMD=("${PYTHON_BIN}" "${HRRR_SCRIPT}" --check-only --status-file "${HRRR_STATUS}")
  if [[ -n "${HRRR_MIN_SAMPLES:-}" ]]; then
    HRRR_CMD+=(--min-samples "${HRRR_MIN_SAMPLES}")
  fi
  if [[ -n "${HRRR_MIN_COVERAGE:-}" ]]; then
    HRRR_CMD+=(--min-coverage "${HRRR_MIN_COVERAGE}")
  fi
  echo "[dataset-slurm] checking raw HRRR tensor; network fetching must run on the login node"
  set +e
  "${HRRR_CMD[@]}"
  hrrr_status=$?
  set -e
  if [[ "${hrrr_status}" -eq 2 ]]; then
    echo "[dataset-slurm] Missing or undersized raw HRRR tensor." >&2
    echo "[dataset-slurm] Run on the login node first: make dataset DATASETS=hrrr" >&2
    exit 2
  elif [[ "${hrrr_status}" -ne 0 ]]; then
    exit "${hrrr_status}"
  fi
fi

CMD=("${PYTHON_BIN}" "${DATASET_SCRIPT}")
if [[ "${CLI_OVERWRITE}" -eq 1 ]]; then
  CMD+=(--overwrite)
fi
if [[ "${#FORWARD_ARGS[@]}" -gt 0 ]]; then
  CMD+=("${FORWARD_ARGS[@]}")
fi

srun "${CMD[@]}"
