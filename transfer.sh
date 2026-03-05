#!/usr/bin/env bash
set -euo pipefail

#===============================================================================
# Script to manage transfers with Jean Zay.
# Usage:
#   ./transfer.sh send        # Send code to Jean Zay
#   ./transfer.sh fetch       # Fetch benchmarks/_figures* from Jean Zay
#   ./transfer.sh supp        # Build code.zip supplementary package
#===============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}"
WORK_DIR="/tmp/neurips_2026_heaytail_flow_matching"
SUPP_DIR="/tmp/neurips_2026_heaytail_flow_matching"
REMOTE_ROOT="jz:/lustre/fswork/projects/rech/jcx/uor49lv/src"
REMOTE_PROJECT="${REMOTE_ROOT}/neurips_2026_heaytail_flow_matching"
REMOTE_FIGURES="${REMOTE_PROJECT}/benchmarks/_figures_*"
LOCAL_BENCH_DIR="${SRC_DIR}/benchmarks"
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

if [[ ! -d "${SRC_DIR}" ]]; then
    echo -e "${RED}Source directory does not exist:${RESET} ${SRC_DIR}" >&2
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
    chmod -R u+rwX "${SRC_DIR}/benchmarks/config" 2>/dev/null || true
    chmod -R u+rwX "${SRC_DIR}/benchmarks/config_blank" 2>/dev/null || true
    chmod -R u+rwX "${SRC_DIR}/benchmarks/_data" 2>/dev/null || true
    chmod -R u+rwX "${SRC_DIR}/benchmarks"/_figures* 2>/dev/null || true
    chmod -R u+rwX "${SRC_DIR}/benchmarks"/_tables* 2>/dev/null || true
    chmod -R u+rwX "${SRC_DIR}/benchmarks"/_results* 2>/dev/null || true
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
      --exclude '__pycache__/' \
      --exclude '*.py[cod]' \
      --exclude '*~' \
      --exclude 'code.zip' \
      --exclude 'transfer.sh' \
      --exclude 'benchmarks/_figures*/' \
      --exclude 'benchmarks/_tables*/' \
      --exclude 'benchmarks/_results*/' \
      --exclude 'benchmarks/_reports*/' \
      --exclude 'cauda/examples/_figures/' \
      "${SRC_DIR}/" "${target_dir}/"

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
    echo -e "${BLUE}Source:${RESET} ${SRC_DIR}"
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
    echo -e "${BLUE}${BOLD}Fetch figures from Jean Zay${RESET}"
    echo -e "${BLUE}Remote figures:${RESET} ${REMOTE_FIGURES}"
    echo -e "${BLUE}Local target:${RESET} ${LOCAL_BENCH_DIR}"
    echo ""

    mkdir -p "${LOCAL_BENCH_DIR}"
    (
        cd "${LOCAL_BENCH_DIR}"
        rsync -var --progress "${REMOTE_FIGURES}" .
    )

    echo -e "${GREEN}${BOLD}Figures synced.${RESET}"
}

build_supp_zip() {
    echo -e "${BLUE}${BOLD}Build supplementary package${RESET}"
    echo -e "${BLUE}Source:${RESET} ${SRC_DIR}"
    echo -e "${BLUE}Staging:${RESET} ${SUPP_DIR}"
    echo -e "${BLUE}Output:${RESET} ${SRC_DIR}/${ZIP_NAME}"
    echo ""

    echo -e "${CYAN}[1/3] Preparing package...${RESET}"
    prepare_common_staging "${SUPP_DIR}"
    cd "${SUPP_DIR}"

    cat > README.md <<'EORD'
# Anonymous Code Supplement

This archive contains code and benchmark scripts for anonymous peer review.
EORD

    # Fail fast if identity-bearing markers are still present in text files.
    local leaked
    leaked="$(
        grep -R -n -E \
        'Hamza|Cherkaoui|hcherkaoui|uor49lv|github.com/hcherkaoui|neurips_2026_heaytail_flow_matching|Heavy-Tail Flow Matching' \
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
    rm -f "${SRC_DIR}/${ZIP_NAME}"
    (
        cd "${SUPP_DIR}"
        zip -qr "${SRC_DIR}/${ZIP_NAME}" .
    )

    echo -e "${CYAN}[3/3] Cleanup...${RESET}"
    rm -rf "${SUPP_DIR}"
    echo -e "${GREEN}${BOLD}Created:${RESET} ${SRC_DIR}/${ZIP_NAME}"
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
