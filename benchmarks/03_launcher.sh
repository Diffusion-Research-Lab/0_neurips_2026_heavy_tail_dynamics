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
CONFIG_DIR="${SCRIPT_DIR}/configs"
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
  --run       Run all benchmark configs found in benchmarks/configs in lexical order
  --blank     Run the smoke config benchmarks/configs/00_blank.yaml only
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
  rm -rf "${PROJECT_ROOT}/_results"
  rm -rf "${PROJECT_ROOT}/runs"

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

if [[ ! -d "${CONFIG_DIR}" ]]; then
  echo "[launcher] Missing config directory: ${CONFIG_DIR}" >&2
  exit 1
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

if [[ "${MODE}" == "blank" ]]; then
  CONFIGS=("${CONFIG_DIR}/00_blank.yaml")
  MODE_TAG="Blank"
else
  mapfile -t CONFIGS < <(find "${CONFIG_DIR}" -maxdepth 1 -type f -name '*.yaml' | sort)
  MODE_TAG="Run"
fi

if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
  echo "[launcher] No config files found in ${CONFIG_DIR}" >&2
  exit 1
fi

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
echo "CONFIG_DIR:   ${CONFIG_DIR}"
printf "CONFIGS:      %s\n" "${CONFIGS[*]}"

echo "-------------------------------------------------------------------------------"
echo "[${MODE_TAG}] Import checks"
step_t0="$(date +%s)"
python - <<'PY'
import genkit
from labkit.config import load_config
print("blank_imports_ok")
PY
step_dt=$(( $(date +%s) - step_t0 ))
echo "[time] ${MODE_TAG} import checks: $(fmt_duration "${step_dt}") (${step_dt}s)"

for cfg in "${CONFIGS[@]}"; do
  cfg_name="$(basename "${cfg}")"
  echo "-------------------------------------------------------------------------------"
  echo "[${MODE_TAG}] ${cfg_name}"
  step_t0="$(date +%s)"
  (
    cd "${PROJECT_ROOT}"
    python benchmarks/main.py --config "${cfg}"
  )
  echo "[✓] ${MODE_TAG} ${cfg_name}"
  step_dt=$(( $(date +%s) - step_t0 ))
  echo "[time] ${MODE_TAG} ${cfg_name}: $(fmt_duration "${step_dt}") (${step_dt}s)"
done

END_TIME="$(date)"
TOTAL_DT=$(( $(date +%s) - TOTAL_T0 ))
echo "-------------------------------------------------------------------------------"
echo "All done."
echo "END: ${END_TIME}"
echo "TOTAL: $(fmt_duration "${TOTAL_DT}") (${TOTAL_DT}s)"
