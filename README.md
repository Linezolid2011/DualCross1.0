# DualCross 1.0 — Hidden-Diff Dual Learning

Best model from trial#11: CrossPert with **hidden-state difference dual learning**, exported as a standalone package (no external STATE dependency).

## Architecture

- **Backbone**: GPT2 (8 layers, 384-dim) + true cross-attention (2 layers, 8 heads)
- **Dual learning**: `DualPerturbationClassifier` on hidden-state difference (treated − control)
- **No cycle consistency**: cycle_loss_weight=0.0 — cycle loss interfered with primary task
- **MSE auxiliary loss**: weight=0.1 on top of Sinkhorn (energy) loss

## Performance vs PD_ver2 (14 metrics)

**10/13 wins** at step 10000 (best checkpoint):

| Metric | DualCross1.0 | PD_ver2 | Win? |
|--------|-------------|---------|------|
| R2 | **0.953** | 0.639 | ✓ |
| DEOver | **0.458** | 0.429 | ✓ |
| DEPrec | **0.423** | 0.421 | ✓ |
| ES | 0.084 | 0.578 | ✗ |
| DirAgr | **0.875** | 0.776 | ✓ |
| AUPRC | **0.443** | 0.441 | ✓ |
| AUROC | **0.691** | 0.664 | ✓ |
| PDCorr | 0.712 | 0.809 | ✗ |
| PDS_L1 | **0.935** | 0.927 | ✓ |
| PDS_L2 | **0.947** | 0.944 | ✓ |
| PDS_cos | 0.959 | 0.960 | ✗ |
| MSE | **0.020** | 0.032 | ✓ |
| MAE | **0.042** | 0.077 | ✓ |

Losses: ES (effect size ranking), PDCorr (per-cell correlation), PDS_cos (reconstruction cosine similarity) — universal bottlenecks no CrossPert variant beats.

## Files

```
DualCross1.0/
├── .gitignore
├── README.md
├── charts/                  # Loss plots (generated during training)
├── checkpoints/
│   └── dualcross/
│       ├── config.json      # Training configuration
│       ├── pert_onehot_map.pt
│       ├── var_dims.pkl
│       ├── data_module.torch
│       ├── version_0/       # Lightning logs
│       └── checkpoints/
│           └── step-step=10000.ckpt  # Best checkpoint
├── code/
│   ├── __init__.py
│   ├── callbacks.py
│   ├── evaluate_tahoe.py    # Tahoe evaluation script (with R2)
│   ├── model.py             # CrossPertModel with DualPerturbationClassifier
│   ├── plot_eval_results.py
│   ├── run_eval.sh          # Single checkpoint evaluation
│   ├── run_all_eval.sh      # Batch evaluation + plots
│   ├── run_opt_D.sh         # Training script (same params as best model)
│   ├── state/               # Vendored STATE package
│   └── train.py             # Training script with dual learning args
├── configs/
│   └── tahoe_ver2.toml      # Dataset config
└── results/                 # Evaluation outputs
```

## Usage

**Evaluate** (requires Tahoe ver2 data at same paths):
```bash
bash code/run_eval.sh
```

**Train from scratch**:
```bash
bash code/run_opt_D.sh
```

## Model config

- hidden_dim=384, cross_attn_layers=2, cross_attn_heads=8
- dual_loss_weight=0.1, dual_on_hidden=True, cycle_loss_weight=0.0
- lr=1e-4, max_steps=40000, seed=789
