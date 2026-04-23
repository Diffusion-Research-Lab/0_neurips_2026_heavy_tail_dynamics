#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TOOLS_ROOT="${PROJECT_ROOT}/.tools"
TOOLS_BIN="${TOOLS_ROOT}/bin"
CONDA_PREFIX="${TOOLS_ROOT}/conda-wgrib2"

mkdir -p "${TOOLS_BIN}"

if command -v wgrib2 >/dev/null 2>&1; then
  echo "[setup] Found wgrib2 at $(command -v wgrib2)"
  exit 0
fi

if [[ -x "${CONDA_PREFIX}/bin/wgrib2" ]]; then
  ln -sf "../conda-wgrib2/bin/wgrib2" "${TOOLS_BIN}/wgrib2"
  echo "[setup] Reusing cached project-local wgrib2"
  exit 0
fi

PACKAGE_MANAGER=""
for candidate in micromamba mamba conda; do
  if command -v "${candidate}" >/dev/null 2>&1; then
    PACKAGE_MANAGER="$(command -v "${candidate}")"
    break
  fi
done

if [[ -z "${PACKAGE_MANAGER}" ]]; then
  echo "[setup] wgrib2 is missing and no conda-compatible executable was found to install it." >&2
  echo "[setup] Install wgrib2 manually or make conda, mamba, or micromamba available, then rerun setup." >&2
  exit 1
fi

echo "[setup] Installing project-local wgrib2 via conda-forge"
if [[ -d "${CONDA_PREFIX}" ]]; then
  "${PACKAGE_MANAGER}" install -y -p "${CONDA_PREFIX}" -c conda-forge wgrib2
else
  "${PACKAGE_MANAGER}" create -y -p "${CONDA_PREFIX}" -c conda-forge wgrib2
fi

if [[ ! -x "${CONDA_PREFIX}/bin/wgrib2" ]]; then
  echo "[setup] Conda completed but ${CONDA_PREFIX}/bin/wgrib2 was not created." >&2
  exit 1
fi

ln -sf "../conda-wgrib2/bin/wgrib2" "${TOOLS_BIN}/wgrib2"
echo "[setup] Installed wgrib2 at ${TOOLS_BIN}/wgrib2"
