#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODE="${1:-}"
REMOTE_HOST="jz"
REMOTE_BASE="/lustre/fswork/projects/rech/jcx/uor49lv/src"
REMOTE_ROOT="${REMOTE_HOST}:${REMOTE_BASE}"
REMOTE_PROJECT="${REMOTE_HOST}:${REMOTE_BASE}/flowbench"
REMOTE_PROJECT_PATH="${REMOTE_BASE}/flowbench"
LOCAL_ANALYSIS_DIR="${PROJECT_ROOT}/benchmarks/data"
WORK_DIR="/tmp/flowbench"
SUPP_DIR="/tmp/anonymous_code_supp"
ZIP_NAME="code.zip"
SSH_CONTROL_PATH="/tmp/flowbench-ssh-%r@%h:%p"
SSH_OPTS=(
  -o ControlMaster=auto
  -o ControlPersist=10m
  -o ControlPath="${SSH_CONTROL_PATH}"
)

usage() {
    cat >&2 <<EOF
Usage: $(basename "$0") [send|fetch|supp]
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

ssh_master_up() {
    ssh "${SSH_OPTS[@]}" -fN "${REMOTE_HOST}"
}

ssh_master_down() {
    ssh -o ControlPath="${SSH_CONTROL_PATH}" -O exit "${REMOTE_HOST}" >/dev/null 2>&1 || true
}

remote_ssh() {
    ssh "${SSH_OPTS[@]}" "${REMOTE_HOST}" "$@"
}

remote_rsync() {
    rsync -e "ssh -o ControlMaster=auto -o ControlPersist=10m -o ControlPath=${SSH_CONTROL_PATH}" "$@"
}

stage_project() {
    local target_dir="$1"
    [[ -n "${target_dir}" && "${target_dir}" != "/" && "${target_dir}" != "${HOME}" ]] || die "unsafe staging dir: ${target_dir}"

    chmod -R u+rwX "${PROJECT_ROOT}/benchmarks" 2>/dev/null || true
    chmod -R u+rwX "${PROJECT_ROOT}/examples/_figures" 2>/dev/null || true
    chmod -R u+rwX "${PROJECT_ROOT}"/*_results* 2>/dev/null || true
    rm -rf "${target_dir}"

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
      --exclude '*_results*/' \
      --exclude '_reports*/' \
      --exclude '_data/' \
      --exclude '_weights/' \
      --exclude 'benchmarks/*_results*/' \
      --exclude 'benchmarks/data/' \
      --exclude 'benchmarks/analysis/data/' \
      --exclude 'benchmarks/_reports*/' \
      --exclude 'benchmarks/_data/' \
      --exclude 'benchmarks/_weights/' \
      --exclude 'examples/_figures/' \
      "${PROJECT_ROOT}/" "${target_dir}/"

    (
        cd "${target_dir}"
        find . -type l -delete
        find . -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '*~' \) -delete
        find . -type d -name '__pycache__' -prune -exec rm -rf {} +
    )
}

send_code() {
    echo "Send to Jean Zay"
    echo "Source: ${PROJECT_ROOT}"
    echo "Remote: ${REMOTE_ROOT}"

    stage_project "${WORK_DIR}"
    rsync -avh --info=stats2,progress2 "${WORK_DIR}" "${REMOTE_ROOT}/"
    rm -rf "${WORK_DIR}"
}

fetch_results() {
    local tmp_root_dirs tmp_analysis_dirs
    echo "Fetch result directories from Jean Zay"
    echo "Remote root: ${REMOTE_PROJECT}"
    echo "Local target: ${LOCAL_ANALYSIS_DIR}"

    mkdir -p "${LOCAL_ANALYSIS_DIR}"
    tmp_root_dirs="$(mktemp)"
    tmp_analysis_dirs="$(mktemp)"
    trap 'rm -f "${tmp_root_dirs}" "${tmp_analysis_dirs}"; ssh_master_down' RETURN

    ssh_master_up
    remote_ssh "
        find '${REMOTE_PROJECT_PATH}' -mindepth 1 -maxdepth 1 -type d -name '*_results*' -printf 'ROOT:%f\n' 2>/dev/null
        find '${REMOTE_PROJECT_PATH}/benchmarks/data' -mindepth 1 -maxdepth 1 -type d -printf 'ANALYSIS:%f\n' 2>/dev/null
        find '${REMOTE_PROJECT_PATH}/benchmarks/analysis/data' -mindepth 1 -maxdepth 1 -type d -printf 'ANALYSIS_OLD:%f\n' 2>/dev/null
    " | while IFS= read -r line; do
        case "${line}" in
            ROOT:*) printf '%s\n' "${line#ROOT:}" >> "${tmp_root_dirs}" ;;
            ANALYSIS:*) printf 'NEW:%s\n' "${line#ANALYSIS:}" >> "${tmp_analysis_dirs}" ;;
            ANALYSIS_OLD:*) printf 'OLD:%s\n' "${line#ANALYSIS_OLD:}" >> "${tmp_analysis_dirs}" ;;
        esac
    done || true

    if [[ ! -s "${tmp_root_dirs}" && ! -s "${tmp_analysis_dirs}" ]]; then
        echo "No remote result directories found under:"
        echo "  ${REMOTE_PROJECT_PATH}/*_results*"
        echo "  ${REMOTE_PROJECT_PATH}/benchmarks/data/*"
        echo "  ${REMOTE_PROJECT_PATH}/benchmarks/analysis/data/*"
        return 0
    fi

    echo "Fetching discovered result directories..."
    if [[ -s "${tmp_root_dirs}" ]]; then
        while IFS= read -r name; do
            [[ -n "${name}" ]] || continue
            echo "Fetching ${REMOTE_PROJECT}/${name}/"
            remote_rsync -var --progress "${REMOTE_PROJECT}/${name}/" "${LOCAL_ANALYSIS_DIR}/"
        done < "${tmp_root_dirs}"
    fi
    if [[ -s "${tmp_analysis_dirs}" ]]; then
        while IFS= read -r entry; do
            [[ -n "${entry}" ]] || continue
            name="${entry#*:}"
            if [[ "${entry}" == NEW:* ]]; then
                echo "Fetching ${REMOTE_PROJECT}/benchmarks/data/${name}/"
                remote_rsync -var --progress "${REMOTE_PROJECT}/benchmarks/data/${name}/" "${LOCAL_ANALYSIS_DIR}/${name}/"
            else
                echo "Fetching ${REMOTE_PROJECT}/benchmarks/analysis/data/${name}/"
                remote_rsync -var --progress "${REMOTE_PROJECT}/benchmarks/analysis/data/${name}/" "${LOCAL_ANALYSIS_DIR}/${name}/"
            fi
        done < "${tmp_analysis_dirs}"
    fi
}

build_supp_zip() {
    echo "Build supplementary package"
    echo "Source: ${PROJECT_ROOT}"
    echo "Output: ${PROJECT_ROOT}/${ZIP_NAME}"

    stage_project "${SUPP_DIR}"
    (
        cd "${SUPP_DIR}"
        rm -f scripts/transfer.sh scripts/fetch_vendor.sh
        if [[ -f pyproject.toml ]]; then
            sed -i 's/^name = "flowbench"/name = "anonymous-code-supplement"/' pyproject.toml
        fi
        cat > README.md <<'EOF'
# Anonymous Code Supplement

This archive contains code and benchmark scripts for anonymous peer review.
EOF

        leaked="$(
            grep -R -n -E \
            'Hamza|Cherkaoui|hcherkaoui|uor49lv|github.com/hcherkaoui|flowbench|Heavy-Tail Flow Matching' \
            --binary-files=without-match \
            --exclude-dir=.git \
            . || true
        )"
        [[ -z "${leaked}" ]] || {
            echo "Anonymization check failed:" >&2
            echo "${leaked}" >&2
            exit 1
        }

        rm -f "${PROJECT_ROOT}/${ZIP_NAME}"
        zip -qr "${PROJECT_ROOT}/${ZIP_NAME}" .
    )
    rm -rf "${SUPP_DIR}"
}

[[ -d "${PROJECT_ROOT}/src/genkit" ]] || die "missing ${PROJECT_ROOT}/src/genkit"
[[ -d "${PROJECT_ROOT}/src/labkit" ]] || die "missing ${PROJECT_ROOT}/src/labkit"

case "${MODE}" in
    send) send_code ;;
    fetch) fetch_results ;;
    supp) build_supp_zip ;;
    *) usage; exit 2 ;;
esac
