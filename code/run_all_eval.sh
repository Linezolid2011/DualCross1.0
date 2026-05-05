#!/bin/bash
# Evaluate all checkpoints and plot metric trends
set -euo pipefail

VENV="/home/dataset-assist-0/cuimenglong/workspace/code/state/state-main/.venv/bin"
export PATH="$VENV:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

CKPT_DIR="$BASE_DIR/checkpoints/dualcross/checkpoints"
PERT_MAP="$BASE_DIR/checkpoints/dualcross/pert_onehot_map.pt"
EVAL_BASE="$BASE_DIR/results"

if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: checkpoint directory not found at $CKPT_DIR"
    exit 1
fi

echo "=============================================="
echo "Evaluating all checkpoints in $CKPT_DIR"
echo "=============================================="

CKPTS=()
while IFS= read -r -d '' f; do
    CKPTS+=("$f")
done < <(find "$CKPT_DIR" -maxdepth 1 -name '*.ckpt' -print0)

if [ ${#CKPTS[@]} -eq 0 ]; then
    echo "No checkpoints found."
    exit 0
fi

echo "Found ${#CKPTS[@]} checkpoint(s)."
for ckpt in "${CKPTS[@]}"; do
    ckpt_name=$(basename "$ckpt" .ckpt)
    eval_dir="$EVAL_BASE/$ckpt_name"
    results_file="$eval_dir/eval_results.json"

    echo "----------------------------------------------"
    echo "Evaluating: $ckpt_name"
    echo "----------------------------------------------"

    if [ -f "$results_file" ]; then
        echo "  Results already exist — skipping."
        continue
    fi

    mkdir -p "$eval_dir"
    python "$SCRIPT_DIR/evaluate_tahoe.py" \
        --ckpt "$ckpt" \
        --pert-map "$PERT_MAP" \
        --output-dir "$eval_dir" \
        --cell-set-len 64 2>&1 | tail -20

    if [ -f "$results_file" ]; then
        echo "  Done → $results_file"
    else
        echo "  WARNING: no results file for $ckpt_name"
    fi
done

# Generate metric plots
python "$SCRIPT_DIR/plot_eval_results.py" "$EVAL_BASE"
