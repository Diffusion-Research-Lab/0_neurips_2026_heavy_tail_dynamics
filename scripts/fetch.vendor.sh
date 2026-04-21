#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENDOR_DIR="${REPO_ROOT}/src/genkit/_vendor"
PYTHON_BIN="${PYTHON:-python}"

mkdir -p "${VENDOR_DIR}"

cleanup_git_metadata() {
    local path="$1"

    find "${path}" -name '.git' -type d -prune -exec rm -rf {} +
    find "${path}" -name '.gitmodules' -type f -delete
}

stage_repo() {
    local name="$1"
    local repo_url="$2"
    local stage_root="$3"
    local stage_dir="${stage_root}/${name}"
    git clone --depth 1 "${repo_url}" "${stage_dir}"
    cleanup_git_metadata "${stage_dir}"
}

clone_or_update() {
    local name="$1"
    local repo_url="$2"
    local target_dir="${VENDOR_DIR}/${name}"
    local stage_root
    stage_root="$(mktemp -d)"

    trap 'rm -rf "${stage_root}"' RETURN

    stage_repo "${name}" "${repo_url}" "${stage_root}"

    if [[ -e "${target_dir}" ]]; then
        rm -rf "${target_dir}"
    fi

    mv "${stage_root}/${name}" "${target_dir}"
    rm -rf "${stage_root}"
    trap - RETURN
}

clone_or_update "DLPM" "https://github.com/hcherkaoui/DLPM"
clone_or_update "flow_matching" "https://github.com/facebookresearch/flow_matching"
clone_or_update "physicsnemo" "https://github.com/NVIDIA/physicsnemo"
clone_or_update "score_sde_pytorch" "https://github.com/yang-song/score_sde_pytorch"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/patch.vendor.py" --vendor-root "${VENDOR_DIR}"

"${PYTHON_BIN}" -m pip install -e "${REPO_ROOT}"
"${PYTHON_BIN}" -m pip install torchdiffeq pot torchquad Cython
"${PYTHON_BIN}" -m pip install -e "${VENDOR_DIR}/flow_matching"
