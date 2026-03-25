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
TOTAL_T0=0

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
  --run       Run benchmark 04 and benchmark 05 with full configs
  --blank     Run benchmark 04 and benchmark 05 with *_blank.yml configs
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

if [[ ! -d "${PROJECT_ROOT}/src/genkit" || ! -d "${PROJECT_ROOT}/src/labkit" ]]; then
  echo "[launcher] Expected directories not found:" >&2
  echo "  ${PROJECT_ROOT}/src/genkit" >&2
  echo "  ${PROJECT_ROOT}/src/labkit" >&2
  exit 1
fi

clean_outputs() {
  echo "[clean] Removing benchmark artifacts..."
  rm -rf "${SCRIPT_DIR}/_results" \
         "${SCRIPT_DIR}/_figures" \
         "${SCRIPT_DIR}/_tables" \
         "${SCRIPT_DIR}/_data/cifar-100-batches-py"

  rm -f "${SCRIPT_DIR}"/heavyflow_*.out "${SCRIPT_DIR}"/heavyflow_*.err \
        "${SCRIPT_DIR}"/htfm_*.out "${SCRIPT_DIR}"/htfm_*.err

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
    candidates+=("${PROJECT_ROOT}/.venv-genkit" "${PROJECT_ROOT}/.venv")
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

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${CPUS}" ]]; then
  export OMP_NUM_THREADS="${CPUS}"
  export MKL_NUM_THREADS="${CPUS}"
  export OPENBLAS_NUM_THREADS="${CPUS}"
  export NUMEXPR_NUM_THREADS="${CPUS}"
fi

if [[ "${MODE}" == "blank" ]]; then
  CFG_04="${SCRIPT_DIR}/04_alpha_values_benchmark_cfg_blank.yml"
  CFG_05="${SCRIPT_DIR}/05_loss_H_comparison_benchmark_cfg_blank.yml"
  MODE_TAG="Blank"
else
  CFG_04="${SCRIPT_DIR}/04_alpha_values_benchmark_cfg.yml"
  CFG_05="${SCRIPT_DIR}/05_loss_H_comparison_benchmark_cfg.yml"
  MODE_TAG="Run"
fi

START_TIME="$(date)"
TOTAL_T0="$(date +%s)"

fmt_duration() {
  local dt="$1"
  local h=$((dt / 3600))
  local m=$(((dt % 3600) / 60))
  local s=$((dt % 60))
  printf "%02dh:%02dm:%02ds" "${h}" "${m}" "${s}"
}

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
  step_t0="$(date +%s)"
  python - <<'PY'
import genkit
from labkit.config import load_config
print("blank_imports_ok")
PY
  step_dt=$(( $(date +%s) - step_t0 ))
  echo "[time] Blank import checks: $(fmt_duration "${step_dt}") (${step_dt}s)"
fi

echo "-------------------------------------------------------------------------------"
echo "[${MODE_TAG}] Benchmark 04"
step_t0="$(date +%s)"
(
  cd "${SCRIPT_DIR}"
  python 04_alpha_values_benchmark_exp.py --config "${CFG_04}"
  python 04_alpha_values_benchmark_fig.py
)
echo "[✓] ${MODE_TAG} Benchmark 04"
step_dt=$(( $(date +%s) - step_t0 ))
echo "[time] ${MODE_TAG} Benchmark 04: $(fmt_duration "${step_dt}") (${step_dt}s)"

echo "-------------------------------------------------------------------------------"
echo "[${MODE_TAG}] Benchmark 05"
step_t0="$(date +%s)"
(
  cd "${SCRIPT_DIR}"
  python 05_loss_H_comparison_benchmark_exp.py --config "${CFG_05}"
  python 05_loss_H_comparison_benchmark_fig.py
)
echo "[✓] ${MODE_TAG} Benchmark 05"
step_dt=$(( $(date +%s) - step_t0 ))
echo "[time] ${MODE_TAG} Benchmark 05: $(fmt_duration "${step_dt}") (${step_dt}s)"

END_TIME="$(date)"
TOTAL_DT=$(( $(date +%s) - TOTAL_T0 ))
echo "-------------------------------------------------------------------------------"
echo "All done."
echo "END: ${END_TIME}"
echo "TOTAL: $(fmt_duration "${TOTAL_DT}") (${TOTAL_DT}s)"
