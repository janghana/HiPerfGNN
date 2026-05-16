#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${SCRIPT_DIR}/logs"; mkdir -p "$LOG_DIR"

ROOT="/mnt/hdd3/hjang/data/internal/GBM/projects/quantized_DSC/graph_structures/IDH_graph"
CG_PATH="${ROOT}/hact_cell/cell_graphs"
TG_PATH="${ROOT}/hact_tissue/tissue_graphs"
ASSIGN_PATH="${ROOT}/hact_cell/assign_mat"
CFG_FILE="${PKG_DIR}/config/hact.yml"
CKPT_DIR="${ROOT}/hact_cell/checkpoints_hact/single_split"
mkdir -p "$CKPT_DIR"

EPOCHS=300
BATCH=4
LR=1e-3
NUM_CLASSES=2
MASTER_PORT=29507
CUDA_VISIBLE_DEVICES=0,1

LOG_FILE="${LOG_DIR}/train.log"
echo "[INFO] start training (log -> ${LOG_FILE})"

torchrun \
  --nproc_per_node=2 \
  --master_port=${MASTER_PORT} \
  "${PKG_DIR}/train.py" \
    --cg_path "${CG_PATH}" \
    --tg_path "${TG_PATH}" \
    --assign_mat_path "${ASSIGN_PATH}" \
    --config_fpath "${CFG_FILE}" \
    --model_path "${CKPT_DIR}" \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH} \
    --learning_rate ${LR} \
    --num_classes ${NUM_CLASSES} \
    --in_ram \
    --no_amp \
    > "${LOG_FILE}" 2>&1

echo "[DONE] training finished (log => ${LOG_FILE})"
