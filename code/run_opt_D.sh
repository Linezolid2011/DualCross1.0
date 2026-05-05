#!/bin/bash
# Opt D: hidden-diff classifier + no cycle + larger model (384-dim)
# Hypothesis: Larger model capacity + hidden-diff dual + no cycle gives the
# model more room to learn both primary and dual tasks without interference.
VENV="/home/dataset-assist-0/cuimenglong/workspace/code/state/state-main/.venv/bin"
export PATH="$VENV:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=3

"$VENV/python" "$SCRIPT_DIR/train.py" \
  --toml "$BASE_DIR/configs/tahoe_ver2.toml" \
  --output-dir "$BASE_DIR/checkpoints" \
  --charts-dir "$BASE_DIR/charts" \
  --name "opt_D_hidden384_nocycle" \
  --embed-key X_hvg \
  --pert-col drugname_drugconc \
  --cell-type-key cell_name \
  --batch-col plate \
  --output-space all \
  --num-workers 4 \
  --max-steps 40000 \
  --val-freq 2000 \
  --batch-size 16 \
  --lr 1e-4 \
  --cell-set-len 64 \
  --hidden-dim 384 \
  --n-heads 8 \
  --cross-attn-layers 2 \
  --cross-attn-heads 8 \
  --mse-aux-weight 0.1 \
  --dual-loss-weight 0.1 \
  --cycle-loss-weight 0.0 \
  --dual-on-hidden \
  --seed 789 \
  --overwrite
