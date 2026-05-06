# DualCross 1.0 — Hidden-Diff Dual Learning

Best model from trial#11: CrossPert with **hidden-state difference dual learning**, exported as a standalone package (no external STATE dependency).

# **Caution: this readme is NOT up to date. Browse ALL results files and compare to get the actual metrics comparison!!!**

## Architecture

- **Backbone**: GPT2 (8 layers, 384-dim) + true cross-attention (2 layers, 8 heads)
- **Dual learning**: `DualPerturbationClassifier` on hidden-state difference (treated − control)
- **No cycle consistency**: cycle_loss_weight=0.0 — cycle loss interfered with primary task
- **MSE auxiliary loss**: weight=0.1 on top of Sinkhorn (energy) loss

## Performance vs PD_ver2 (14 metrics)

**12/14 wins** at step 32000 (best of 4 tied checkpoints at 18000/26000/32000/36000):

| Metric | DualCross1.0 | PD_ver2 | Win? |
|--------|-------------|---------|------|
| R2 | **0.940** | 0.639 | ✓ |
| DEOver | **0.536** | 0.429 | ✓ |
| DEPrec | **0.469** | 0.421 | ✓ |
| EScorr | **0.729** | 0.382 | ✓ |
| DirAgr | **0.904** | 0.776 | ✓ |
| LFCSpear | **0.802** | 0.528 | ✓ |
| AUPRC | **0.514** | 0.441 | ✓ |
| AUROC | **0.746** | 0.664 | ✓ |
| PDCorr | 0.743 | **0.809** | ✗ |
| PDS_L1 | **0.943** | 0.927 | ✓ |
| PDS_L2 | **0.951** | 0.944 | ✓ |
| PDS_cos | 0.947 | **0.960** | ✗ |
| MSE | **0.023** | 0.032 | ✓ |
| MAE | **0.042** | 0.077 | ✓ |

**EScorr** = PerturbDiff paper's effect_size_corr: Pearson correlation of per-condition DE gene counts (|δ| > std(δ_real)) across conditions. This replaces cell-eval's `de_spearman_sig` (Spearman correlation of FDR<0.05 gene counts per perturbation), which is a different metric not used in the PerturbDiff paper.

**2 universal bottlenecks** (no CrossPert variant beats PD_ver2 on these): PDCorr (per-cell correlation), PDS_cos (reconstruction cosine similarity). All other 12 metrics are wins.

### Checkpoint win counts (14 metrics, with EScorr)

| Step | Wins | Step | Wins |
|------|------|------|------|
| 2000 | 6/14 | 22000 | 11/14 |
| 4000 | 6/14 | 24000 | 10/14 |
| 6000 | 8/14 | **26000** | **12/14** |
| 8000 | 9/14 | 28000 | 11/14 |
| 10000 | 11/14 | 30000 | 10/14 |
| 12000 | 10/14 | **32000** | **12/14** ← best |
| 14000 | 11/14 | 34000 | 10/14 |
| 16000 | 11/14 | **36000** | **12/14** |
| **18000** | **12/14** | 38000 | 11/14 |
| 20000 | 11/14 | 40000 | 11/14 |

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
│           └── step-step=32000.ckpt  # Best checkpoint (12/14 wins)
├── code/
│   ├── __init__.py
│   ├── callbacks.py
│   ├── compute_effect_size_corr.py  # PerturbDiff paper's EScorr metric
│   ├── compute_lfcspear.py  # LFCSpear workaround (cell_eval dtype bug)
│   ├── evaluate_tahoe.py    # Tahoe evaluation script (with R2)
│   ├── model.py             # CrossPertModel with DualPerturbationClassifier
│   ├── plot_eval_results.py
│   ├── run_opt_D.sh         # Train with best model params (dualcross)
│   ├── run_train.sh         # Generic training script
│   ├── run_eval.sh          # Single checkpoint evaluation
│   ├── run_all_eval.sh      # Batch evaluate all checkpoints + plots
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
