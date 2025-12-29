#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash sync_to_jz.sh
#
# Assumes:
# - ssh host alias "jz" is configured in ~/.ssh/config

LOCAL_DIR="../neurips_2026_heaytail_flow_matching/"
REMOTE_DIR="/lustre/fswork/projects/rech/jcx/uor49lv/src/neurips_2026_heaytail_flow_matching/"

rsync -avP --delete \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.py[cod]' \
  --exclude '.pytest_cache/' \
  --exclude '*figure*/' \
  --exclude '*table*/' \
  --exclude '*.pdf' \
  --exclude '*.pkl' \
  "$LOCAL_DIR" \
  "jz:$REMOTE_DIR"
