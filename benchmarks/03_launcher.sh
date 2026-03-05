#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   Local quick pipeline check: bash benchmarks/03_launcher.sh --blank
#   Local full benchmark:       bash benchmarks/03_launcher.sh --run [--cpus N]
#   Cleanup benchmark artifacts: bash benchmarks/03_launcher.sh --clean

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
    --clean)
      MODE="clean"; shift ;;
    --cpus)
      CPUS="$2"; shift 2 ;;
    --venv-dir)
      VENV_DIR="$2"; shift 2 ;;
    -h|--help)
      cat <<USAGE
Usage: bash benchmarks/03_launcher.sh [--run|--blank|--clean] [--cpus N] [--venv-dir DIR]

Options:
  --run       Run full benchmark (experiments 1, 2 and 3)
  --blank     Run minimal benchmark smoke test (uses *_blank.yml)
  --clean     Remove benchmark-generated artifacts and run pyclean
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
  echo "[launcher] Missing mode. Use --run, --blank or --clean." >&2
  exit 2
fi

if [[ ! -d "${PROJECT_ROOT}/cauda" || ! -d "${PROJECT_ROOT}/labkit" ]]; then
  echo "[launcher] Expected directories not found:" >&2
  echo "  ${PROJECT_ROOT}/cauda" >&2
  echo "  ${PROJECT_ROOT}/labkit" >&2
  exit 1
fi

clean_outputs() {
  echo "[clean] Removing benchmark artifacts..."
  rm -rf "${SCRIPT_DIR}/_results" \
         "${SCRIPT_DIR}/_figures" \
         "${SCRIPT_DIR}/_tables" \
         "${SCRIPT_DIR}/_data/cifar-100-batches-py"

  rm -f "${SCRIPT_DIR}"/heavyflow_*.out "${SCRIPT_DIR}"/heavyflow_*.err

  rm -rf "${SCRIPT_DIR}/__pycache__"

  if command -v pyclean >/dev/null 2>&1; then
    pyclean "${PROJECT_ROOT}" || true
    echo "[clean] pyclean completed."
  else
    echo "[clean] pyclean not found; skipping."
  fi

  echo "[clean] Done."
}

if [[ "${MODE}" == "clean" ]]; then
  clean_outputs
  exit 0
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

CFG_RUN_DIR="${SCRIPT_DIR}/config"
CFG_BLANK_DIR="${SCRIPT_DIR}/config_blank"
RUN_CFG_1="${CFG_RUN_DIR}/bench_1_config.yaml"
RUN_CFG_2="${CFG_RUN_DIR}/bench_2_config.yaml"
BLANK_CFG_1="${CFG_BLANK_DIR}/bench_1_config_blank.yml"
BLANK_CFG_2="${CFG_BLANK_DIR}/bench_2_config_blank.yml"
RUN_CFG_3="${CFG_RUN_DIR}/bench_3_config.yaml"
BLANK_CFG_3="${CFG_BLANK_DIR}/bench_3_config_blank.yml"

if [[ "${MODE}" == "blank" ]]; then
  CFG_1="${BLANK_CFG_1}"
  CFG_2="${BLANK_CFG_2}"
  CFG_3="${BLANK_CFG_3}"
  MODE_TAG="Blank"
else
  CFG_1="${RUN_CFG_1}"
  CFG_2="${RUN_CFG_2}"
  CFG_3="${RUN_CFG_3}"
  MODE_TAG="Run"
fi

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
fi

echo "-------------------------------------------------------------------------------"
echo "[${MODE_TAG}] Experiment 1"
(
  cd "${SCRIPT_DIR}"
  python 04_AlphaStableFlowLinear_comparison_exp.py --config "${CFG_1}"
  python 04_AlphaStableFlowLinear_comparison_fig.py
)
echo "[✓] ${MODE_TAG} Experiment 1"

echo "-------------------------------------------------------------------------------"
echo "[${MODE_TAG}] Experiment 2"
(
  cd "${SCRIPT_DIR}"
  python 05_alpha_values_benchmark_exp.py --config "${CFG_2}"
  python 05_alpha_values_benchmark_fig.py
)
echo "[✓] ${MODE_TAG} Experiment 2"

echo "-------------------------------------------------------------------------------"
echo "[${MODE_TAG}] Experiment 3 (CIFAR100)"
(
  cd "${SCRIPT_DIR}"
  python 06_cifar100_longtail_benchmark_exp.py --config "${CFG_3}"
  python 06_cifar100_longtail_benchmark_fig.py
)
echo "[✓] ${MODE_TAG} Experiment 3"

END_TIME="$(date)"
echo "-------------------------------------------------------------------------------"
echo "All done."
echo "END: ${END_TIME}"
