#!/usr/bin/env bash

#SBATCH --job-name=htfm_run
#SBATCH --output=htfm_run_%j.out
#SBATCH --error=htfm_run_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --partition=gpu_p13
#SBATCH --qos=qos_gpu-t3
#SBATCH --account=jcx@v100

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
JZ_MODULE="${JZ_MODULE:-pytorch-gpu/py3/2.8.0}"
BENCH_MAIN_REL="benchmarks/01_main.py"
PYTHON_BIN=""
CLI_CONFIG_PATH=""
CLI_BATCH_DIR=""
CLI_SHARD_COUNT=""
CLI_SHARD_INDEX=""
CLI_SKIP_EXISTING=0
CLI_FAIL_ON_ERROR=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CLI_CONFIG_PATH="$2"
      shift 2
      ;;
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
    --skip-existing)
      CLI_SKIP_EXISTING=1
      shift
      ;;
    --fail-on-error)
      CLI_FAIL_ON_ERROR=1
      shift
      ;;
    *)
      echo "[slurm] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -d "${SUBMIT_DIR}/src/genkit" && -d "${SUBMIT_DIR}/src/labkit" && -d "${SUBMIT_DIR}/benchmarks" ]]; then
  PROJECT_ROOT="${SUBMIT_DIR}"
elif [[ -d "${SUBMIT_DIR}/../src/genkit" && -d "${SUBMIT_DIR}/../src/labkit" && -d "${SUBMIT_DIR}/../benchmarks" ]]; then
  PROJECT_ROOT="$(cd -- "${SUBMIT_DIR}/.." && pwd)"
else
  echo "[slurm] Could not infer project root from SLURM_SUBMIT_DIR=${SUBMIT_DIR}" >&2
  echo "[slurm] Submit from repo root or benchmarks directory." >&2
  exit 1
fi
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv-genkit}"
BENCH_MAIN="${PROJECT_ROOT}/${BENCH_MAIN_REL}"
CONFIG_PATH="${CLI_CONFIG_PATH:-${CONFIG_PATH:-${PROJECT_ROOT}/benchmarks/configs/pilot/image.yaml}}"
BATCH_DIR="${CLI_BATCH_DIR:-${BATCH_DIR:-}}"

case "${CONFIG_PATH}" in
  /*) ;;
  *) CONFIG_PATH="${PROJECT_ROOT}/${CONFIG_PATH}" ;;
esac

case "${BATCH_DIR}" in
  ""|/*) ;;
  *) BATCH_DIR="${PROJECT_ROOT}/${BATCH_DIR}" ;;
esac

if [[ ! -d "${PROJECT_ROOT}/src/genkit" || ! -d "${PROJECT_ROOT}/src/labkit" ]]; then
  echo "[slurm] Expected directories not found:" >&2
  echo "  ${PROJECT_ROOT}/src/genkit" >&2
  echo "  ${PROJECT_ROOT}/src/labkit" >&2
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
    echo "[slurm] Expected: ${PROJECT_ROOT}/.venv-genkit (or ${PROJECT_ROOT}/.venv)" >&2
    echo "[slurm] Run setup first on login node:" >&2
    echo "  make setup" >&2
    exit 1
  fi
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"
PYTHON_BIN="$(command -v python)"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export FLOWBENCH_REQUIRE_PREPROCESSED_REAL_DATA="${FLOWBENCH_REQUIRE_PREPROCESSED_REAL_DATA:-1}"

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[slurm] python command not found after venv activation." >&2
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[slurm] Missing config file: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -f "${BENCH_MAIN}" ]]; then
  echo "[slurm] Missing benchmark entrypoint: ${BENCH_MAIN}" >&2
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import torch
import genkit
from labkit.config import load_config
print(f"[slurm] Runtime preflight OK - python/torch={torch.__version__}")
PY

CPUS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${CPUS}"
export MKL_NUM_THREADS="${CPUS}"
export OPENBLAS_NUM_THREADS="${CPUS}"
export NUMEXPR_NUM_THREADS="${CPUS}"

SHARD_COUNT="${CLI_SHARD_COUNT:-${SHARD_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}}"
SHARD_INDEX="${CLI_SHARD_INDEX:-${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

if [[ -z "${BATCH_DIR}" && "${SHARD_COUNT}" -gt 1 ]]; then
  SAVE_ROOT="$("${PYTHON_BIN}" - <<PY
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path("${CONFIG_PATH}").read_text())
print(cfg.get("save", {}).get("root_dir", "runs"))
PY
)"
  JOB_TAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
  CONFIG_STEM="$(basename "${CONFIG_PATH}" .yaml)"
  if [[ "${SAVE_ROOT}" = /* ]]; then
    BATCH_DIR="${SAVE_ROOT}/${JOB_TAG}_${CONFIG_STEM}"
  elif [[ "${SAVE_ROOT}" == "benchmarks/data" || "${SAVE_ROOT}" == "benchmarks/data/"* ]]; then
    BATCH_DIR="${PROJECT_ROOT}/${SAVE_ROOT}/${JOB_TAG}_${CONFIG_STEM}"
  elif [[ -n "${RUN_ROOT:-}" ]]; then
    BATCH_DIR="${RUN_ROOT}/${SAVE_ROOT}/${JOB_TAG}_${CONFIG_STEM}"
  elif [[ -n "${WORK:-}" ]]; then
    BATCH_DIR="${WORK}/flowbench_runs/${SAVE_ROOT}/${JOB_TAG}_${CONFIG_STEM}"
  else
    BATCH_DIR="${PROJECT_ROOT}/${SAVE_ROOT}/${JOB_TAG}_${CONFIG_STEM}"
  fi
fi

if [[ -n "${BATCH_DIR}" ]]; then
  mkdir -p "${BATCH_DIR}"
fi

echo "=============================================================================="
echo "Heavy Tail Flow Benchmark Runner (GPU / Slurm)"
echo "SUBMIT_DIR:    ${SUBMIT_DIR}"
echo "PROJECT_ROOT:  ${PROJECT_ROOT}"
echo "VENV_DIR:      ${VENV_DIR}"
echo "HOST:          $(hostname)"
echo "Python:        $("${PYTHON_BIN}" --version 2>&1)"
echo "SLURM_JOB_ID:  ${SLURM_JOB_ID:-<none>}"
echo "CONFIG_PATH:   ${CONFIG_PATH}"
echo "SHARD:         $((SHARD_INDEX + 1))/${SHARD_COUNT}"
echo "BATCH_DIR:     ${BATCH_DIR:-<auto>}"
echo "CPUs:          ${CPUS}"
echo "CUDA devices:  ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "SKIP_EXISTING: ${SKIP_EXISTING}"
echo "REQUIRE_PREPROCESSED_REAL_DATA: ${FLOWBENCH_REQUIRE_PREPROCESSED_REAL_DATA}"
echo "=============================================================================="

srun nvidia-smi || true
CMD=(
  "${PYTHON_BIN}" "${BENCH_MAIN}"
  --config "${CONFIG_PATH}"
  --shard-count "${SHARD_COUNT}"
  --shard-index "${SHARD_INDEX}"
)

if [[ -n "${BATCH_DIR}" ]]; then
  CMD+=(--batch-dir "${BATCH_DIR}")
fi

if [[ "${CLI_SKIP_EXISTING}" -eq 1 || "${SKIP_EXISTING}" == "1" ]]; then
  CMD+=(--skip-existing)
fi

if [[ "${CLI_FAIL_ON_ERROR}" -eq 1 || "${FAIL_ON_ERROR:-0}" == "1" ]]; then
  CMD+=(--fail-on-error)
fi

srun "${CMD[@]}"
