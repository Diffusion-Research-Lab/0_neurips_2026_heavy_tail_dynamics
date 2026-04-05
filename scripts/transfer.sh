#!/usr/bin/env bash
set -euo pipefail

#===============================================================================
# Script to manage transfers with Jean Zay.
# Usage:
#   ./transfer.sh send        # Send code to Jean Zay
#   ./transfer.sh fetch       # Fetch benchmarks/_figures and _tables from Jean Zay
#   ./transfer.sh supp        # Build code.zip supplementary package
#===============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_DIR="/tmp/flowbench_send"
SUPP_DIR="/tmp/anonymous_code_supp"
REMOTE_ROOT="jz:/lustre/fswork/projects/rech/jcx/uor49lv/src"
REMOTE_PROJECT="${REMOTE_ROOT}/flowbench"
REMOTE_FIGURES_DIR="${REMOTE_PROJECT}/benchmarks/_figures/"
REMOTE_TABLES_DIR="${REMOTE_PROJECT}/benchmarks/_tables/"
LOCAL_BENCH_DIR="${PROJECT_ROOT}/benchmarks"
ZIP_NAME="code.zip"
MODE="${1:-}"
SCRIPT_NAME="$(basename "$0")"

if [[ -t 1 ]]; then
    BLUE="\033[94m"
    CYAN="\033[96m"
    GREEN="\033[92m"
    YELLOW="\033[93m"
    RED="\033[91m"
    BOLD="\033[1m"
    RESET="\033[0m"
else
    BLUE=""; CYAN=""; GREEN=""; YELLOW=""; RED=""; BOLD=""; RESET=""
fi

if [[ ! -d "${PROJECT_ROOT}" ]]; then
    echo -e "${RED}Project root does not exist:${RESET} ${PROJECT_ROOT}" >&2
    exit 1
fi

if [[ ! -d "${PROJECT_ROOT}/src/genkit" || ! -d "${PROJECT_ROOT}/src/labkit" ]]; then
    echo -e "${RED}Expected src layout not found.${RESET}" >&2
    echo "Missing: ${PROJECT_ROOT}/src/genkit and/or ${PROJECT_ROOT}/src/labkit" >&2
    exit 1
fi

if [[ -z "${MODE}" ]]; then
    echo -e "${RED}Missing option.${RESET} Choose one of: send, fetch, supp." >&2
    echo "Usage: ${SCRIPT_NAME} [send|fetch|supp]" >&2
    exit 2
fi

ensure_safe_dir() {
    local dir="$1"
    if [[ -z "${dir}" || "${dir}" == "/" || "${dir}" == "${HOME}" ]]; then
        echo -e "${RED}Unsafe directory value:${RESET} ${dir}" >&2
        exit 1
    fi
}

repair_local_perms() {
    # Ensure benchmark outputs/config dirs are traversable for staging.
    # Make local tree readable/traversable for staging.
    chmod -R u+rwX "${PROJECT_ROOT}/benchmarks" 2>/dev/null || true
    chmod -R u+rwX "${PROJECT_ROOT}/examples/_figures" 2>/dev/null || true
}

prepare_common_staging() {
    local target_dir="$1"
    ensure_safe_dir "${target_dir}"
    repair_local_perms
    rm -rf "${target_dir}"

    # Use rsync staging with excludes so unreadable generated artifacts do not break send.
    rsync -a \
      --exclude '.git/' \
      --exclude '.github/' \
      --exclude '.gitignore' \
      --exclude '.pytest_cache/' \
      --exclude '.mypy_cache/' \
      --exclude '.ruff_cache/' \
      --exclude '.venv/' \
      --exclude 'venv/' \
      --exclude 'build/' \
      --exclude 'dist/' \
      --exclude '*.egg-info/' \
      --exclude '__pycache__/' \
      --exclude '*.py[cod]' \
      --exclude '*~' \
      --exclude 'code.zip' \
      --exclude 'sandbox/' \
      --exclude 'benchmarks/_figures*/' \
      --exclude 'benchmarks/_tables*/' \
      --exclude 'benchmarks/_results*/' \
      --exclude 'benchmarks/_reports*/' \
      --exclude 'benchmarks/_data/' \
      --exclude 'benchmarks/_weights/' \
      --exclude 'examples/_figures/' \
      "${PROJECT_ROOT}/" "${target_dir}/"

    cd "${target_dir}"

    if find . -type l -print -quit | grep -q .; then
        echo -e "${YELLOW}  - Removing symlinks from staging copy${RESET}"
        find . -type l -print -delete
    fi
    find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*~" \) -delete
    find . -type d -name "__pycache__" -prune -exec rm -rf {} +
}

send_code() {
    echo -e "${BLUE}${BOLD}Send to Jean Zay${RESET}"
    echo -e "${BLUE}Source:${RESET} ${PROJECT_ROOT}"
    echo -e "${BLUE}Staging:${RESET} ${WORK_DIR}"
    echo -e "${BLUE}Remote:${RESET} ${REMOTE_ROOT}/"
    echo ""

    echo -e "${CYAN}[1/3] Preparing code...${RESET}"
    prepare_common_staging "${WORK_DIR}"

    echo -e "${CYAN}[2/3] Sending code...${RESET}"
    rsync -avh --info=stats2,progress2 "${WORK_DIR}" "${REMOTE_ROOT}/"

    echo -e "${CYAN}[3/3] Cleanup...${RESET}"
    rm -rf "${WORK_DIR}"

    echo -e "${GREEN}${BOLD}Done.${RESET}"
}

fetch_figures() {
    echo -e "${BLUE}${BOLD}Fetch benchmark artifacts from Jean Zay${RESET}"
    echo -e "${BLUE}Remote figures:${RESET} ${REMOTE_FIGURES_DIR}"
    echo -e "${BLUE}Remote tables:${RESET} ${REMOTE_TABLES_DIR}"
    echo -e "${BLUE}Local target:${RESET} ${LOCAL_BENCH_DIR}"
    echo ""

    mkdir -p "${LOCAL_BENCH_DIR}/_figures" "${LOCAL_BENCH_DIR}/_tables"
    (
        cd "${LOCAL_BENCH_DIR}"
        rsync -var --progress "${REMOTE_FIGURES_DIR}" "./_figures/" || true
        rsync -var --progress "${REMOTE_TABLES_DIR}" "./_tables/" || true
    )

    echo -e "${GREEN}${BOLD}Artifacts synced.${RESET}"
}

build_supp_zip() {
    echo -e "${BLUE}${BOLD}Build supplementary package${RESET}"
    echo -e "${BLUE}Source:${RESET} ${PROJECT_ROOT}"
    echo -e "${BLUE}Staging:${RESET} ${SUPP_DIR}"
    echo -e "${BLUE}Output:${RESET} ${PROJECT_ROOT}/${ZIP_NAME}"
    echo ""

    echo -e "${CYAN}[1/3] Preparing package...${RESET}"
    prepare_common_staging "${SUPP_DIR}"
    cd "${SUPP_DIR}"

    rm -f scripts/transfer.sh scripts/fetch_vendor.sh
    if [[ -f "pyproject.toml" ]]; then
        sed -i 's/^name = "flowbench"/name = "anonymous-code-supplement"/' pyproject.toml
    fi

    cat > README.md <<'EORD'
# Anonymous Code Supplement

This archive contains code and benchmark scripts for anonymous peer review.
EORD

    # Fail fast if identity-bearing markers are still present in text files.
    local leaked
    leaked="$(
        grep -R -n -E \
        'Hamza|Cherkaoui|hcherkaoui|uor49lv|github.com/hcherkaoui|flowbench|Heavy-Tail Flow Matching' \
        --binary-files=without-match \
        --exclude-dir=.git \
        . || true
    )"
    if [[ -n "${leaked}" ]]; then
        echo -e "${RED}Anonymization check failed. Leaked markers found:${RESET}" >&2
        echo "${leaked}" >&2
        rm -rf "${SUPP_DIR}"
        exit 1
    fi

    echo -e "${CYAN}[2/3] Building ${ZIP_NAME}...${RESET}"
    rm -f "${PROJECT_ROOT}/${ZIP_NAME}"
    (
        cd "${SUPP_DIR}"
        zip -qr "${PROJECT_ROOT}/${ZIP_NAME}" .
    )

    echo -e "${CYAN}[3/3] Cleanup...${RESET}"
    rm -rf "${SUPP_DIR}"
    echo -e "${GREEN}${BOLD}Created:${RESET} ${PROJECT_ROOT}/${ZIP_NAME}"
}

case "${MODE}" in
    send)
        send_code
        ;;
    fetch)
        fetch_figures
        ;;
    supp)
        build_supp_zip
        ;;
    *)
        echo "Usage: ${SCRIPT_NAME} [send|fetch|supp]" >&2
        exit 2
        ;;
esac
