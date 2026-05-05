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
