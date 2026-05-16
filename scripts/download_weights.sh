#!/usr/bin/env bash
# Fetch pre-trained checkpoints into the matching weights/ subfolders.
#
# Requires: pip install gdown

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

FOLDER_URL="https://drive.google.com/drive/folders/1umdKHyWaPxT6LinbZGcJOZVoDtBKQYzS"

if ! command -v gdown >/dev/null 2>&1; then
  echo "[ERROR] gdown not found. Install with:  pip install gdown" >&2
  exit 1
fi

STAGING="$(mktemp -d -t hiperfgnn_weights_XXXXXX)"
trap "rm -rf '$STAGING'" EXIT

echo "[INFO] fetching weights from Google Drive folder..."
gdown --folder "$FOLDER_URL" -O "$STAGING"

mkdir -p "$REPO_DIR/Perfusion_VQVAE/weights"
mkdir -p "$REPO_DIR/Hierarchical_GNN/weights"

mv "$STAGING"/*/dsc_perfusion_vqvae.ckpt          "$REPO_DIR/Perfusion_VQVAE/weights/"   2>/dev/null || \
  mv "$STAGING"/dsc_perfusion_vqvae.ckpt          "$REPO_DIR/Perfusion_VQVAE/weights/"

for f in idh_hact_classifier.pt 1p19q_hact_classifier.pt who_grade_hact_classifier.pt; do
  mv "$STAGING"/*/"$f"  "$REPO_DIR/Hierarchical_GNN/weights/"  2>/dev/null || \
    mv "$STAGING"/"$f"  "$REPO_DIR/Hierarchical_GNN/weights/"
done

echo "[DONE] weights placed under:"
ls -lh "$REPO_DIR/Perfusion_VQVAE/weights/"
ls -lh "$REPO_DIR/Hierarchical_GNN/weights/"
