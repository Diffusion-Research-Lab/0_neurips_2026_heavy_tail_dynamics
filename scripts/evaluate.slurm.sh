#!/usr/bin/env bash

#SBATCH --job-name=htfm_eval
#SBATCH --output=htfm_eval_%j.out
#SBATCH --error=htfm_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --account=jcx@v100

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
JZ_MODULE="${JZ_MODULE:-pytorch-gpu/py3/2.8.0}"
PYTHON_BIN=""
CLI_BATCH_DIR=""
CLI_SHARD_COUNT=""
CLI_SHARD_INDEX=""
CLI_DEVICE=""
CLI_N_EVAL_SAMPLES=""
CLI_N_EVAL_REPEATS=""
CLI_INSPECT_SAMPLES=""
CLI_PROBE_SIZE=""
CLI_OVERWRITE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch-dir)
      CLI_BATCH_DIR="$2"
      shift 2
      ;;
    --shard-count)
      CLI_SHARD_COUNT="$2"
      shift 2
      ;;
    --shard-index)
      CLI_SHARD_INDEX="$2"
      shift 2
      ;;
    --device)
      CLI_DEVICE="$2"
      shift 2
      ;;
    --n-eval-samples)
      CLI_N_EVAL_SAMPLES="$2"
      shift 2
      ;;
    --n-eval-repeats)
      CLI_N_EVAL_REPEATS="$2"
      shift 2
      ;;
    --inspect-samples)
      CLI_INSPECT_SAMPLES="$2"
      shift 2
      ;;
    --probe-size)
      CLI_PROBE_SIZE="$2"
      shift 2
      ;;
    --overwrite)
      CLI_OVERWRITE=1
      shift
      ;;
    *)
      echo "[eval-slurm] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -d "${SUBMIT_DIR}/src/genkit" && -d "${SUBMIT_DIR}/src/labkit" && -d "${SUBMIT_DIR}/benchmarks" ]]; then
  PROJECT_ROOT="${SUBMIT_DIR}"
elif [[ -d "${SUBMIT_DIR}/../src/genkit" && -d "${SUBMIT_DIR}/../src/labkit" && -d "${SUBMIT_DIR}/../benchmarks" ]]; then
  PROJECT_ROOT="$(cd -- "${SUBMIT_DIR}/.." && pwd)"
else
  echo "[eval-slurm] Could not infer project root from SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
  exit 1
fi

VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv-genkit}"
BATCH_DIR="${CLI_BATCH_DIR:-${BATCH_DIR:-}}"
case "${BATCH_DIR}" in
  "" ) echo "[eval-slurm] --batch-dir is required." >&2; exit 1 ;;
  /*) ;;
  *) BATCH_DIR="${PROJECT_ROOT}/${BATCH_DIR}" ;;
esac

if ! command -v module >/dev/null 2>&1; then
  echo "[eval-slurm] 'module' command is required on Jean Zay." >&2
  exit 1
fi

module purge || true
conda deactivate 2>/dev/null || true
module load "${JZ_MODULE}"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
    VENV_DIR="${PROJECT_ROOT}/.venv"
  else
    echo "[eval-slurm] Missing virtual environment." >&2
    exit 1
  fi
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
PYTHON_BIN="$(command -v python)"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

CPUS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${CPUS}"
export MKL_NUM_THREADS="${CPUS}"
export OPENBLAS_NUM_THREADS="${CPUS}"
export NUMEXPR_NUM_THREADS="${CPUS}"

SHARD_COUNT="${CLI_SHARD_COUNT:-${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}}"
SHARD_INDEX="${CLI_SHARD_INDEX:-${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}}"
DEVICE="${CLI_DEVICE:-${DEVICE:-cuda}}"
N_EVAL_SAMPLES="${CLI_N_EVAL_SAMPLES:-${N_EVAL_SAMPLES:-10000}}"
N_EVAL_REPEATS="${CLI_N_EVAL_REPEATS:-${N_EVAL_REPEATS:-10}}"
INSPECT_SAMPLES="${CLI_INSPECT_SAMPLES:-${INSPECT_SAMPLES:-2048}}"
PROBE_SIZE="${CLI_PROBE_SIZE:-${PROBE_SIZE:-256}}"

echo "=============================================================================="
echo "Heavy Tail Flow Benchmark Evaluator (GPU / Slurm)"
echo "PROJECT_ROOT:     ${PROJECT_ROOT}"
echo "BATCH_DIR:        ${BATCH_DIR}"
echo "SHARD:            $((SHARD_INDEX + 1))/${SHARD_COUNT}"
echo "DEVICE:           ${DEVICE}"
echo "N_EVAL_SAMPLES:   ${N_EVAL_SAMPLES}"
echo "N_EVAL_REPEATS:   ${N_EVAL_REPEATS}"
echo "INSPECT_SAMPLES:  ${INSPECT_SAMPLES}"
echo "PROBE_SIZE:       ${PROBE_SIZE}"
echo "=============================================================================="

srun nvidia-smi || true
CMD=(
  "${PYTHON_BIN}" -m benchmarks.evaluate
  --batch-dir "${BATCH_DIR}"
  --device "${DEVICE}"
  --shard-count "${SHARD_COUNT}"
  --shard-index "${SHARD_INDEX}"
  --n-eval-samples "${N_EVAL_SAMPLES}"
  --n-eval-repeats "${N_EVAL_REPEATS}"
  --inspect-samples "${INSPECT_SAMPLES}"
  --probe-size "${PROBE_SIZE}"
)

if [[ "${CLI_OVERWRITE}" -eq 1 ]]; then
  CMD+=(--overwrite)
fi

srun "${CMD[@]}"
