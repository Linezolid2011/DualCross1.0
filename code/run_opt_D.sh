#!/bin/bash
# Opt D: hidden-diff classifier + no cycle + larger model (384-dim)
set -euo pipefail

# Defaults
SEED=789
GPU=0

# Parse optional args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed=*) SEED="${1#*=}" ;;
        --gpu=*)  GPU="${1#*=}" ;;
        *) echo "Unknown option: $1"; echo "Usage: $0 [--seed=N] [--gpu=N]"; exit 1 ;;
    esac
    shift
done

VENV="/home/dataset-assist-0/cuimenglong/workspace/code/state/state-main/.venv/bin"
export PATH="$VENV:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="$GPU"

echo "Starting dualcross training — seed=$SEED gpu=$GPU"

"$VENV/python" "$SCRIPT_DIR/train.py" \
  --toml "$BASE_DIR/configs/tahoe_ver2.toml" \
  --output-dir "$BASE_DIR/checkpoints" \
  --charts-dir "$BASE_DIR/charts" \
  --name "dualcross" \
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
  --seed "$SEED" \
  --overwrite
