#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   Laptop:           bash bench_0_run_all.sh

####################################################################################################
# Settings

CPUS=""
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpus) CPUS="$2"; shift 2 ;;
    -h|--help)
      echo "bash runner.sh [--srun] [--cpus N]"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

# Check cauda and labkit accessible
if [[ ! -d "${PROJECT_ROOT}/cauda" || ! -d "${PROJECT_ROOT}/labkit" ]]; then
  echo "[runner] Expected directories not found:" >&2
  echo "  ${PROJECT_ROOT}/cauda" >&2
  echo "  ${PROJECT_ROOT}/labkit" >&2
  echo "[runner] Move this script so that ../cauda and ../labkit exist." >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/cauda:${PROJECT_ROOT}/labkit${PYTHONPATH:+:$PYTHONPATH}"

####################################################################################################
# Main

START_TIME="$(date)"

echo "-------------------------------------------------------------------------------"
echo "Heavy Tail Flow Benchmark Runner (CPU)"
echo "START:        ${START_TIME}"
echo "HOST:         $(hostname)"
echo "PWD:          $(pwd)"
echo "Python:       $(python --version 2>&1)"
echo "CPUs:         ${CPUS}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-<none>}"
echo "PYTHONPATH:   ${PYTHONPATH}"

echo "-------------------------------------------------------------------------------"
echo "[Experiment 1]"
python bench_1_AlphaStableFlowLinear_comparison.py --config bench_1_config.yaml
echo "[✓] Done Experiment 1"

echo "-------------------------------------------------------------------------------"
echo "[Experiment 2]"
python bench_2_alpha_values_benchmark.py --config bench_2_config.yaml
echo "[✓] Done Experiment 2"

END_TIME="$(date)"
echo "-------------------------------------------------------------------------------"
echo "All done."
echo "END: ${END_TIME}"