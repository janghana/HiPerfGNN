#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PKG_DIR="$(dirname "$SCRIPT_DIR")"

cd "$SCRIPT_DIR"

python "${PKG_DIR}/train.py" \
  --base "${PKG_DIR}/configs/vqgan_sep.yaml" \
  -t True \
  --no-test True \
  > train.log 2>&1 &
echo "[INFO] train.py launched in background (log -> ${SCRIPT_DIR}/train.log)"
