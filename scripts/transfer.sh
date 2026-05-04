#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODE="${1:-}"
REMOTE_HOST="jz"
REMOTE_BASE="/lustre/fswork/projects/rech/jcx/uor49lv/src"
ZIP_NAME="code.zip"

usage() { echo "Usage: $(basename "$0") [send|supp]" >&2; }
die()   { echo "Error: $*" >&2; exit 1; }

stage_project() {
    local target_dir="$1"
    [[ -n "${target_dir}" && "${target_dir}" != "/" && "${target_dir}" != "${HOME}" ]] \
        || die "unsafe staging dir: ${target_dir}"

    chmod -R u+rwX "${PROJECT_ROOT}/benchmarks" 2>/dev/null || true
    chmod -R u+rwX "${PROJECT_ROOT}/examples/_figures" 2>/dev/null || true
    rm -rf "${target_dir}"

    rsync -a \
      --exclude '.git/' \
      --exclude '.claude/' \
      --exclude '.github/' \
      --exclude '.gitignore' \
      --exclude '.pytest_cache/' \
      --exclude '.mypy_cache/' \
      --exclude '.ruff_cache/' \
      --exclude '.venv/' \
      --exclude 'venv/' \
      --exclude 'src/genkit/_vendor/' \
      --exclude 'build/' \
      --exclude 'dist/' \
      --exclude '*.codex' \
      --exclude '*.egg-info/' \
      --exclude '__pycache__/' \
      --exclude '*.py[cod]' \
      --exclude '*~' \
      --exclude 'code.zip' \
      --exclude 'sandbox/*.ipynb' \
      --exclude 'sandbox/*.sh' \
      --exclude 'sandbox/figures/' \
      --exclude '*_results*/' \
      --exclude 'benchmarks/artifacts/' \
      --exclude 'benchmarks/data_archive/' \
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
    local work_dir="/tmp/flowbench"
    echo "Send -> ${REMOTE_HOST}:${REMOTE_BASE}/flowbench"
    stage_project "${work_dir}"
    rsync -avh --info=stats2,progress2 "${work_dir}" "${REMOTE_HOST}:${REMOTE_BASE}/"
    rm -rf "${work_dir}"
}

build_supp_zip() {
    local supp_dir="/tmp/anonymous_code_supp"
    echo "Build supplementary package -> ${PROJECT_ROOT}/${ZIP_NAME}"
    stage_project "${supp_dir}"
    (
        cd "${supp_dir}"
        rm -f scripts/transfer.sh scripts/fetch.vendor.sh
        if [[ -f pyproject.toml ]]; then
            sed -i 's/^name = "flowbench"/name = "anonymous-code-supplement"/' pyproject.toml
        fi
        cat > README.md <<'EOF'
# Anonymous Code Supplement

This archive contains code and benchmark scripts for anonymous peer review.
EOF
        python - <<'PY'
import json
from pathlib import Path

for path in Path(".").rglob("*.ipynb"):
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    changed = False
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True
    if changed:
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
PY
        leaked="$(
            grep -R -n -E \
              'Hamza|Cherkaoui|hcherkaoui|uor49lv|github.com/hcherkaoui|Heavy-Tail Flow Matching' \
              --binary-files=without-match \
              --exclude-dir=.git \
              . || true
        )"
        [[ -z "${leaked}" ]] || { echo "Anonymization check failed:" >&2; echo "${leaked}" >&2; exit 1; }

        rm -f "${PROJECT_ROOT}/${ZIP_NAME}"
        zip -qr "${PROJECT_ROOT}/${ZIP_NAME}" .
    )
    rm -rf "${supp_dir}"
}

[[ -d "${PROJECT_ROOT}/src/genkit" ]] || die "missing ${PROJECT_ROOT}/src/genkit"
[[ -d "${PROJECT_ROOT}/src/labkit" ]] || die "missing ${PROJECT_ROOT}/src/labkit"

case "${MODE}" in
    send)  send_code ;;
    supp)  build_supp_zip ;;
    *)     usage; exit 2 ;;
esac
