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
##SBATCH -p <gpu_partition_if_needed>  # only if your account requires it

set -euo pipefail

# Usage:
#   Jean Zay (login): sbatch bench_0_run_all.slurm

####################################################################################################
# Settings
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
JZ_MODULE="${JZ_MODULE:-pytorch-gpu/py3/2.2.0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(
  d="$SUBMIT_DIR"
  for _ in 0 1 2 3 4; do
    if [[ -d "${d}/cauda" && -d "${d}/labkit" ]]; then
      printf '%s\n' "$d"
      exit 0
    fi
    d="$(cd -- "${d}/.." && pwd)"
  done
  exit 1
)" || { echo "[slurm] Could not find project root containing ./cauda and ./labkit starting from: $SUBMIT_DIR" >&2; exit 1; }

# Load Torch module
if command -v module >/dev/null 2>&1; then
  module purge || true
  conda deactivate 2>/dev/null || true
  if [[ -n "${JZ_MODULE}" ]]; then
    module load "${JZ_MODULE}"
  fi
fi

# Check cauda and labkit accessible
if [[ ! -d "${PROJECT_ROOT}/cauda" || ! -d "${PROJECT_ROOT}/labkit" ]]; then
  echo "[slurm] Expected directories not found:" >&2
  echo "  ${PROJECT_ROOT}/cauda" >&2
  echo "  ${PROJECT_ROOT}/labkit" >&2
  echo "[slurm] Move this script so that ../cauda and ../labkit exist." >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/cauda:${PROJECT_ROOT}/labkit${PYTHONPATH:+:$PYTHONPATH}"

CPUS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$CPUS"
export MKL_NUM_THREADS="$CPUS"
export OPENBLAS_NUM_THREADS="$CPUS"
export NUMEXPR_NUM_THREADS="$CPUS"

####################################################################################################
# Main

echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-<none>}"
echo "PWD:             $(pwd)"
echo "BASH_SOURCE[0]:   ${BASH_SOURCE[0]}"

START_TIME="$(date)"

echo "=============================================================================="
echo "Heavy Tail Flow Benchmark Runner (GPU / Slurm)"
echo "START:        ${START_TIME}"
echo "HOST:         $(hostname)"
echo "PWD:          $(pwd)"
echo "Python:       $(python --version 2>&1)"
echo "CPUs:         ${CPUS}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-<none>}"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-<none>}"
echo "PYTHONPATH:   ${PYTHONPATH}"
echo "=============================================================================="

srun nvidia-smi || true

echo "=============================================================================="
echo "[Experiment 1]"
srun python bench_1_AlphaStableFlowLinear_comparison.py --config bench_1_config.yaml
echo "[✓] Done Experiment 1"

echo "=============================================================================="
echo "[Experiment 2]"
srun python bench_2_alpha_values_benchmark.py --config bench_2_config.yaml
echo "[✓] Done Experiment 2"

END_TIME="$(date)"
echo "=============================================================================="
echo "All done."
echo "END: ${END_TIME}"
