#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENDOR_DIR="${REPO_ROOT}/src/genkit/_vendor"
PYTHON_BIN="${PYTHON:-python}"

mkdir -p "${VENDOR_DIR}"

clone_or_update() {
    local name="$1"
    local repo_url="$2"
    local target_dir="${VENDOR_DIR}/${name}"

    if [[ -d "${target_dir}/.git" ]]; then
        git -C "${target_dir}" pull --ff-only
        return 0
    fi

    if [[ -e "${target_dir}" ]]; then
        echo "Refusing to overwrite existing path: ${target_dir}" >&2
        return 1
    fi

    git clone "${repo_url}" "${target_dir}"
}

clone_or_update "DLPM" "https://github.com/hcherkaoui/DLPM"
clone_or_update "flow_matching" "https://github.com/facebookresearch/flow_matching"
clone_or_update "score_sde_pytorch" "https://github.com/yang-song/score_sde_pytorch"

"${PYTHON_BIN}" -m pip install -e "${REPO_ROOT}"
"${PYTHON_BIN}" -m pip install torchdiffeq pot torchquad Cython
"${PYTHON_BIN}" -m pip install -e "${VENDOR_DIR}/flow_matching"
