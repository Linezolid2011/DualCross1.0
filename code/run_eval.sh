#!/bin/bash
# Evaluate a single checkpoint
VENV="/home/dataset-assist-0/cuimenglong/workspace/code/state/state-main/.venv/bin"
export PATH="$VENV:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

CKPT="${1:-$BASE_DIR/checkpoints/dualcross/checkpoints/step-step=10000.ckpt}"
OUTDIR="${2:-$BASE_DIR/results/single}"

python "$SCRIPT_DIR/evaluate_tahoe.py" \
  --ckpt "$CKPT" \
  --output-dir "$OUTDIR"

# Patch metrics that cell_eval can't compute correctly
python "$SCRIPT_DIR/compute_lfcspear.py" --eval-dir "$OUTDIR" 2>&1
python "$SCRIPT_DIR/compute_effect_size_corr.py" --eval-dir "$OUTDIR" --patch 2>&1
