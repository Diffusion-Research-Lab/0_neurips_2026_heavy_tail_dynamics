#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   Local quick pipeline check: bash benchmarks/03_launcher.sh --blank
#   Local full benchmark:       bash benchmarks/03_launcher.sh --run [--cpus N]

MODE=""
CPUS=""
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      MODE="run"; shift ;;
    --blank)
      MODE="blank"; shift ;;
    --cpus)
      CPUS="$2"; shift 2 ;;
    --venv-dir)
      VENV_DIR="$2"; shift 2 ;;
    -h|--help)
      cat <<USAGE
Usage: bash benchmarks/03_launcher.sh [--run|--blank] [--cpus N] [--venv-dir DIR]

Options:
  --run       Run full benchmark (experiments 1 and 2)
  --blank     Run minimal benchmark smoke test (uses *_blank.yml)
  --cpus N    Thread controls for OMP/MKL/OPENBLAS/NUMEXPR
  --venv-dir  Virtual env to activate before running (default: auto)
USAGE
      exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2 ;;
  esac
done

if [[ -z "${MODE}" ]]; then
  echo "[launcher] Missing mode. Use --run or --blank." >&2
  exit 2
fi

if [[ ! -d "${PROJECT_ROOT}/cauda" || ! -d "${PROJECT_ROOT}/labkit" ]]; then
  echo "[launcher] Expected directories not found:" >&2
  echo "  ${PROJECT_ROOT}/cauda" >&2
  echo "  ${PROJECT_ROOT}/labkit" >&2
  exit 1
fi

activate_venv_if_available() {
  local candidates=()
  if [[ -n "${VENV_DIR}" ]]; then
    candidates+=("${VENV_DIR}")
  else
    candidates+=("${PROJECT_ROOT}/.venv-cauda" "${PROJECT_ROOT}/.venv")
  fi

  local vdir
  for vdir in "${candidates[@]}"; do
    if [[ -f "${vdir}/bin/activate" ]]; then
      # shellcheck disable=SC1090
      source "${vdir}/bin/activate"
      echo "[launcher] Activated venv: ${vdir}"
      return 0
    fi
  done
}

activate_venv_if_available

if ! command -v python >/dev/null 2>&1; then
  echo "[launcher] python command not found." >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/cauda:${PROJECT_ROOT}/labkit${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${CPUS}" ]]; then
  export OMP_NUM_THREADS="${CPUS}"
  export MKL_NUM_THREADS="${CPUS}"
  export OPENBLAS_NUM_THREADS="${CPUS}"
  export NUMEXPR_NUM_THREADS="${CPUS}"
fi

CFG_DIR="${SCRIPT_DIR}/configs"
RUN_CFG_1="${CFG_DIR}/bench_1_config.yaml"
RUN_CFG_2="${CFG_DIR}/bench_2_config.yaml"
BLANK_CFG_1="${CFG_DIR}/bench_1_blank.yml"
BLANK_CFG_2="${CFG_DIR}/bench_2_blank.yml"

for f in "${RUN_CFG_1}" "${RUN_CFG_2}" "${BLANK_CFG_1}" "${BLANK_CFG_2}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[launcher] Missing config file: ${f}" >&2
    exit 1
  fi
done

START_TIME="$(date)"

echo "-------------------------------------------------------------------------------"
echo "Heavy Tail Flow Benchmark Launcher"
echo "MODE:         ${MODE}"
echo "START:        ${START_TIME}"
echo "HOST:         $(hostname)"
echo "PWD:          $(pwd)"
echo "Python:       $(python --version 2>&1)"
echo "CPUs:         ${CPUS:-<default>}"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-<none>}"
echo "PYTHONPATH:   ${PYTHONPATH}"

if [[ "${MODE}" == "blank" ]]; then
  echo "-------------------------------------------------------------------------------"
  echo "[Blank] Import checks"
  python - <<'PY'
import cauda
from labkit.config import load_config
print("blank_imports_ok")
PY

  echo "-------------------------------------------------------------------------------"
  echo "[Blank] Experiment 1"
  (
    cd "${SCRIPT_DIR}"
    python 04_AlphaStableFlowLinear_comparison.py --config "${BLANK_CFG_1}"
  )
  echo "[✓] Blank Experiment 1"

  echo "-------------------------------------------------------------------------------"
  echo "[Blank] Experiment 2"
  (
    cd "${SCRIPT_DIR}"
    python 05_alpha_values_benchmark.py --config "${BLANK_CFG_2}"
  )
  echo "[✓] Blank Experiment 2"
else
  echo "-------------------------------------------------------------------------------"
  echo "[Run] Experiment 1"
  (
    cd "${SCRIPT_DIR}"
    python 04_AlphaStableFlowLinear_comparison.py --config "${RUN_CFG_1}"
  )
  echo "[✓] Done Experiment 1"

  echo "-------------------------------------------------------------------------------"
  echo "[Run] Experiment 2"
  (
    cd "${SCRIPT_DIR}"
    python 05_alpha_values_benchmark.py --config "${RUN_CFG_2}"
  )
  echo "[✓] Done Experiment 2"
fi

END_TIME="$(date)"
echo "-------------------------------------------------------------------------------"
echo "All done."
echo "END: ${END_TIME}"
