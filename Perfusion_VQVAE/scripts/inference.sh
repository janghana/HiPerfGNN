#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PKG_DIR="$(dirname "$SCRIPT_DIR")"

CKPT="${PKG_DIR}/weights/dsc_perfusion_vqvae.ckpt"
CFG="${PKG_DIR}/configs/vqgan_sep.yaml"
OUTDIR="${OUTDIR:-/mnt/hdd3/hjang/data/internal/GBM/projects/quantized_DSC/habitat_maps_quantized}"
MODE="${MODE:-cluster}"

python "${PKG_DIR}/inference.py" \
  --mode "$MODE" \
  --resume "$CKPT" \
  --config "$CFG" \
  --outdir "$OUTDIR" \
  ${NODE_OUTDIR:+--node_outdir "$NODE_OUTDIR"} \
  --center internal \
  --task quantized_DSC \
  --device 0 \
  --batch_size 512
