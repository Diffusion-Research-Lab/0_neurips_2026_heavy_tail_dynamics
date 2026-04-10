#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   Local quick pipeline check: bash benchmarks/launchers/local.sh --blank
#   Local full benchmark:       bash benchmarks/launchers/local.sh --run

MODE=""
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd -- "${BENCHMARK_ROOT}/.." && pwd)"
CONFIG_DIR="${BENCHMARK_ROOT}/configs"
VENV_DIR=""
PYTHON_BIN=""
PYTHONPATH_VALUE="${PROJECT_ROOT}/src"
TOTAL_T0=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      MODE="run"; shift ;;
    --blank)
      MODE="blank"; shift ;;
    --venv-dir)
      VENV_DIR="$2"; shift 2 ;;
    -h|--help)
      cat <<USAGE
Usage: bash benchmarks/launchers/local.sh [--run|--blank] [--venv-dir DIR]

Options:
  --run       Run all benchmark configs found in benchmarks/configs in lexical order
  --blank     Run the smoke config benchmarks/configs/00_blank.yaml only
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

if [[ ! -d "${PROJECT_ROOT}/src/genkit" || ! -d "${PROJECT_ROOT}/src/labkit" ]]; then
  echo "[launcher] Expected directories not found:" >&2
  echo "  ${PROJECT_ROOT}/src/genkit" >&2
  echo "  ${PROJECT_ROOT}/src/labkit" >&2
  exit 1
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

PYTHON_BIN="$(command -v python)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[launcher] python command not found." >&2
  exit 1
fi

export PYTHONPATH="${PYTHONPATH_VALUE}${PYTHONPATH:+:$PYTHONPATH}"

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
echo "Python:       $("${PYTHON_BIN}" --version 2>&1)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-<none>}"
echo "PYTHONPATH:   ${PYTHONPATH}"
echo "CONFIG_DIR:   ${CONFIG_DIR}"
printf "CONFIGS:      %s\n" "${CONFIGS[*]}"

echo "-------------------------------------------------------------------------------"
echo "[${MODE_TAG}] Import checks"
step_t0="$(date +%s)"
"${PYTHON_BIN}" - <<'PY'
import genkit
from labkit.config import load_config
print("launcher_imports_ok")
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
    "${PYTHON_BIN}" -m benchmarks.runner.main --config "${cfg}"
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
