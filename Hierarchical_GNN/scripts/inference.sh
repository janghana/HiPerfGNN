#!/usr/bin/env bash
set -euo pipefail

# HiPerfGNN inference launcher. Selects task + test split via env vars.
#
#   TASK = idh (default) | 1p19q | who_grade
#   TEST = internal (default) | external   (external only valid for TASK=idh)
#
# Example:
#   bash inference.sh                       # IDH internal
#   TEST=external bash inference.sh         # IDH external (UPenn)
#   TASK=1p19q bash inference.sh            # 1p/19q internal
#   TASK=who_grade bash inference.sh        # WHO Grade internal

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PKG_DIR="$(dirname "$SCRIPT_DIR")"

TASK="${TASK:-idh}"
TEST="${TEST:-internal}"

CFG="${PKG_DIR}/config/hact.yml"
BASE="/mnt/hdd3/hjang/data/internal/GBM/projects/quantized_DSC/graph_structures"

case "$TASK" in
  idh)
    CKPT_FILE="${PKG_DIR}/weights/idh_hact_classifier.pt"
    NUM_CLASSES=2
    INT_ROOT="${BASE}/IDH_graph/hact_cell"
    ;;
  1p19q)
    CKPT_FILE="${PKG_DIR}/weights/1p19q_hact_classifier.pt"
    NUM_CLASSES=2
    INT_ROOT="${BASE}/1p19q_graph/hact_cell"
    ;;
  who_grade)
    CKPT_FILE="${PKG_DIR}/weights/who_grade_hact_classifier.pt"
    NUM_CLASSES=3
    INT_ROOT="${BASE}/who_grade_graph/hact_cell"
    ;;
  *)
    echo "TASK must be one of: idh, 1p19q, who_grade (got: $TASK)" >&2
    exit 1
    ;;
esac

if [[ "$TEST" == "internal" ]]; then
  ROOT="$INT_ROOT"
  EXTRA_ARGS=()
elif [[ "$TEST" == "external" ]]; then
  if [[ "$TASK" != "idh" ]]; then
    echo "external test is only available for TASK=idh" >&2
    exit 1
  fi
  ROOT="/mnt/hdd3/hjang/data/UPENN/GBM/projects/quantized_DSC/graph_structures/IDH_graph/anat_4ch_mri_slic"
  EXTRA_ARGS=(
    --cg_path "${ROOT}/cell_graphs/test"
    --tg_path "/mnt/hdd3/hjang/data/UPENN/GBM/projects/quantized_DSC/graph_structures/tissue_graphs/test"
    --am_path "${ROOT}/assign_mat"
  )
else
  echo "TEST must be internal or external (got: $TEST)" >&2
  exit 1
fi

# inference.py expects --ckpt_dir; symlink the single chosen ckpt into a temp dir
CKPT_DIR="$(mktemp -d -t hiperfgnn_ckpt_XXXXXX)"
trap "rm -rf '$CKPT_DIR'" EXIT
ln -s "$CKPT_FILE" "$CKPT_DIR/$(basename "$CKPT_FILE")"

LOG_DIR="${SCRIPT_DIR}/logs";    mkdir -p "$LOG_DIR"
OUT_DIR="${SCRIPT_DIR}/outputs"; mkdir -p "$OUT_DIR"

TS=$(date '+%Y%m%d_%H%M%S')
LOG="${LOG_DIR}/infer_${TASK}_${TEST}_${TS}.log"
CSV="${OUT_DIR}/infer_${TASK}_${TEST}_${TS}.csv"

echo "TASK=${TASK} TEST=${TEST}"
echo "ckpt -> ${CKPT_FILE}"
echo "log  -> ${LOG}"
echo "csv  -> ${CSV}"

PYTHON="${PYTHON:-python}"

CUDA_VISIBLE_DEVICES=0 \
"$PYTHON" "${PKG_DIR}/inference.py" \
  --ckpt_dir   "$CKPT_DIR" \
  --root       "$ROOT" \
  --cfg        "$CFG" \
  --num_classes "$NUM_CLASSES" \
  --batch      4 \
  --out_csv    "$CSV" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG"
