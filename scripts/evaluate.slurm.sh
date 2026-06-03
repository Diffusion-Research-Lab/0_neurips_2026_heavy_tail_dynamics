#!/usr/bin/env bash

#SBATCH --job-name=htfm_evaluate
#SBATCH --output=htfm_evaluate_%j.out
#SBATCH --error=htfm_evaluate_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --account=jcx@v100

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
JZ_MODULE="${JZ_MODULE:-pytorch-gpu/py3/2.8.0}"
BENCH_EVAL_REL="benchmarks/02_evaluate.py"
PYTHON_BIN=""
CLI_BATCH_DIR=""
CLI_SHARD_COUNT=""
CLI_SHARD_INDEX=""
CLI_DEVICE=""
CLI_N_EVAL_SAMPLES=""
CLI_N_EVAL_REPEATS=""
CLI_SAMPLE_BATCH_SIZE=""
CLI_MAX_MMD_SAMPLES=""
CLI_SELECTION_ONLY=0
CLI_SELECTION_SPLIT=""
CLI_SELECTION_REPEATS=""
CLI_SELECTION_BATCH_SIZE=""
CLI_OVERWRITE=0
CLI_FAIL_ON_ERROR=0

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
    --sample-batch-size)
      CLI_SAMPLE_BATCH_SIZE="$2"
      shift 2
      ;;
    --max-mmd-samples)
      CLI_MAX_MMD_SAMPLES="$2"
      shift 2
      ;;
    --selection-only)
      CLI_SELECTION_ONLY=1
      shift
      ;;
    --selection-split)
      CLI_SELECTION_SPLIT="$2"
      shift 2
      ;;
    --selection-repeats)
      CLI_SELECTION_REPEATS="$2"
      shift 2
      ;;
    --selection-batch-size)
      CLI_SELECTION_BATCH_SIZE="$2"
      shift 2
      ;;
    --overwrite)
      CLI_OVERWRITE=1
      shift
      ;;
    --fail-on-error)
      CLI_FAIL_ON_ERROR=1
      shift
      ;;
    *)
      echo "[eval-slurm] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -d "${SUBMIT_DIR}/src/datakit" && -d "${SUBMIT_DIR}/src/genkit" && -d "${SUBMIT_DIR}/src/labkit" && -d "${SUBMIT_DIR}/benchmarks" ]]; then
  PROJECT_ROOT="${SUBMIT_DIR}"
elif [[ -d "${SUBMIT_DIR}/../src/datakit" && -d "${SUBMIT_DIR}/../src/genkit" && -d "${SUBMIT_DIR}/../src/labkit" && -d "${SUBMIT_DIR}/../benchmarks" ]]; then
  PROJECT_ROOT="$(cd -- "${SUBMIT_DIR}/.." && pwd)"
else
  echo "[eval-slurm] Could not infer project root from SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
  exit 1
fi

VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
BENCH_EVAL="${PROJECT_ROOT}/${BENCH_EVAL_REL}"
BATCH_DIR="${CLI_BATCH_DIR:-${BATCH_DIR:-}}"
case "${BATCH_DIR}" in
  "" ) echo "[eval-slurm] --batch-dir is required." >&2; exit 1 ;;
  /*) ;;
  *) BATCH_DIR="${PROJECT_ROOT}/${BATCH_DIR}" ;;
esac

if [[ ! -f "${BENCH_EVAL}" ]]; then
  echo "[eval-slurm] Missing benchmark evaluator: ${BENCH_EVAL}" >&2
  exit 1
fi

if ! command -v module >/dev/null 2>&1; then
  echo "[eval-slurm] 'module' command is required on Jean Zay." >&2
  exit 1
fi

module purge || true
conda deactivate 2>/dev/null || true
module load "${JZ_MODULE}"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  if [[ -f "${PROJECT_ROOT}/.venv-genkit/bin/activate" ]]; then
    VENV_DIR="${PROJECT_ROOT}/.venv-genkit"
  else
    echo "[eval-slurm] Missing virtual environment." >&2
    echo "[eval-slurm] Expected: ${PROJECT_ROOT}/.venv (or ${PROJECT_ROOT}/.venv-genkit)" >&2
    exit 1
  fi
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
PYTHON_BIN="$(command -v python)"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export FLOWBENCH_REQUIRE_PREPROCESSED_REAL_DATA="${FLOWBENCH_REQUIRE_PREPROCESSED_REAL_DATA:-1}"

CPUS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${CPUS}"
export MKL_NUM_THREADS="${CPUS}"
export OPENBLAS_NUM_THREADS="${CPUS}"
export NUMEXPR_NUM_THREADS="${CPUS}"

SHARD_COUNT="${CLI_SHARD_COUNT:-${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}}"
SHARD_INDEX="${CLI_SHARD_INDEX:-${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}}"
DEVICE="${CLI_DEVICE:-${DEVICE:-cuda}}"
N_EVAL_SAMPLES="${CLI_N_EVAL_SAMPLES:-${N_EVAL_SAMPLES:-512}}"
N_EVAL_REPEATS="${CLI_N_EVAL_REPEATS:-${N_EVAL_REPEATS:-2}}"
SAMPLE_BATCH_SIZE="${CLI_SAMPLE_BATCH_SIZE:-${SAMPLE_BATCH_SIZE:-16}}"
MAX_MMD_SAMPLES="${CLI_MAX_MMD_SAMPLES:-${MAX_MMD_SAMPLES:-2048}}"
SELECTION_SPLIT="${CLI_SELECTION_SPLIT:-${SELECTION_SPLIT:-val}}"
SELECTION_REPEATS="${CLI_SELECTION_REPEATS:-${SELECTION_REPEATS:-8}}"
SELECTION_BATCH_SIZE="${CLI_SELECTION_BATCH_SIZE:-${SELECTION_BATCH_SIZE:-64}}"

echo "=============================================================================="
echo "Heavy Tail Flow Benchmark Evaluator (GPU / Slurm)"
echo "PROJECT_ROOT:     ${PROJECT_ROOT}"
echo "BATCH_DIR:        ${BATCH_DIR}"
echo "SHARD:            $((SHARD_INDEX + 1))/${SHARD_COUNT}"
echo "DEVICE:           ${DEVICE}"
echo "N_EVAL_SAMPLES:   ${N_EVAL_SAMPLES}"
echo "N_EVAL_REPEATS:   ${N_EVAL_REPEATS}"
echo "SAMPLE_BATCH:     ${SAMPLE_BATCH_SIZE}"
echo "SELECTION_ONLY:   ${CLI_SELECTION_ONLY}"
echo "SELECTION_SPLIT:  ${SELECTION_SPLIT}"
echo "SELECTION_REP:    ${SELECTION_REPEATS}"
echo "SELECTION_BATCH:  ${SELECTION_BATCH_SIZE}"
echo "MAX_MMD_SAMPLES:  ${MAX_MMD_SAMPLES}"
echo "REQUIRE_REAL_PRE: ${FLOWBENCH_REQUIRE_PREPROCESSED_REAL_DATA}"
echo "=============================================================================="

srun nvidia-smi || true
CMD=(
  "${PYTHON_BIN}" "${BENCH_EVAL}"
  --batch-dir "${BATCH_DIR}"
  --device "${DEVICE}"
  --shard-count "${SHARD_COUNT}"
  --shard-index "${SHARD_INDEX}"
  --n-eval-samples "${N_EVAL_SAMPLES}"
  --n-eval-repeats "${N_EVAL_REPEATS}"
  --sample-batch-size "${SAMPLE_BATCH_SIZE}"
  --max-mmd-samples "${MAX_MMD_SAMPLES}"
  --selection-split "${SELECTION_SPLIT}"
  --selection-repeats "${SELECTION_REPEATS}"
  --selection-batch-size "${SELECTION_BATCH_SIZE}"
)

if [[ "${CLI_SELECTION_ONLY}" -eq 1 ]]; then
  CMD+=(--selection-only)
fi

if [[ "${CLI_OVERWRITE}" -eq 1 ]]; then
  CMD+=(--overwrite)
fi

if [[ "${CLI_FAIL_ON_ERROR}" -eq 1 || "${FAIL_ON_ERROR:-0}" == "1" ]]; then
  CMD+=(--fail-on-error)
fi

srun "${CMD[@]}"
