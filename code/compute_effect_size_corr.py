"""
Compute PerturbDiff paper's effect_size_corr metric from pred.h5ad / real.h5ad.

Matches the computation in:
  cuimenglong/.../pathway model/scripts/evaluate.py

effect_size_corr = Pearson correlation of per-condition n_sig counts,
  where n_sig = count of genes with |delta| > std(delta_real).
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr


def compute_effect_size_corr(adata_pred, adata_real,
                              pert_col="perturbation",
                              cell_col="cell_type",
                              control_label="[('DMSO_TF', 0.0, 'uM')]"):
    """Compute PerturbDiff paper's effect size correlation metric.

    For each (cell_line, drug) condition:
      - Compute pseudobulk delta = drug_mean - control_mean
      - Count genes where |delta| > std(delta_real)
      - Pearson-correlate real vs pred counts across conditions.
    """
    X_pred = adata_pred.X
    X_real = adata_real.X
    if hasattr(X_pred, 'toarray'):
        X_pred = X_pred.toarray()
    if hasattr(X_real, 'toarray'):
        X_real = X_real.toarray()
    X_pred = np.asarray(X_pred, dtype=np.float64)
    X_real = np.asarray(X_real, dtype=np.float64)

    obs_pred = adata_pred.obs
    obs_real = adata_real.obs

    cell_lines = sorted(set(obs_real[cell_col].unique()))

    effect_real = []
    effect_pred = []

    for cl in cell_lines:
        # Control mean from real data
        ctrl_mask = ((obs_real[cell_col] == cl) &
                     (obs_real[pert_col] == control_label))
        if ctrl_mask.sum() == 0:
            continue
        ctrl_mean = X_real[ctrl_mask].mean(axis=0)

        drugs = sorted(set(obs_real[obs_real[cell_col] == cl][pert_col].unique()))
        for drug in drugs:
            if drug == control_label:
                continue

            real_mask = ((obs_real[cell_col] == cl) &
                         (obs_real[pert_col] == drug))
            pred_mask = ((obs_pred[cell_col] == cl) &
                         (obs_pred[pert_col] == drug))

            if real_mask.sum() == 0 or pred_mask.sum() == 0:
                continue

            real_mean = X_real[real_mask].mean(axis=0)
            pred_mean = X_pred[pred_mask].mean(axis=0)

            delta_real = real_mean - ctrl_mean
            delta_pred = pred_mean - ctrl_mean

            # Threshold: |delta| > std(delta_real) per condition
            # (matches PerturbDiff paper evaluate.py:338-344)
            sig_threshold = float(np.std(delta_real))
            if sig_threshold < 1e-12:
                continue

            n_sig_real = int(np.sum(np.abs(delta_real) > sig_threshold))
            n_sig_pred = int(np.sum(np.abs(delta_pred) > sig_threshold))

            effect_real.append(n_sig_real)
            effect_pred.append(n_sig_pred)

    if len(effect_real) < 3:
        return float("nan"), effect_real, effect_pred

    # Pearson correlation (matches PerturbDiff paper evaluate.py:648-653)
    if np.std(effect_real) < 1e-12 or np.std(effect_pred) < 1e-12:
        corr = float("nan")
    else:
        corr = float(pearsonr(effect_real, effect_pred)[0])

    return corr, effect_real, effect_pred


def patch_eval_results(output_dir, effect_size_corr):
    """Add effect_size_corr to eval_results.json."""
    results_path = os.path.join(output_dir, "eval_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    else:
        results = {}
    results["effect_size_corr"] = effect_size_corr
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Patched effect_size_corr={effect_size_corr:.4f} into {results_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute PerturbDiff paper's effect_size_corr metric")
    parser.add_argument("--eval-dir", required=True,
                        help="Directory containing pred.h5ad and real.h5ad")
    parser.add_argument("--patch", action="store_true",
                        help="Patch value into eval_results.json")
    args = parser.parse_args()

    import anndata as ad

    pred_path = os.path.join(args.eval_dir, "pred.h5ad")
    real_path = os.path.join(args.eval_dir, "real.h5ad")

    if not os.path.exists(pred_path):
        print(f"ERROR: {pred_path} not found")
        sys.exit(1)
    if not os.path.exists(real_path):
        print(f"ERROR: {real_path} not found")
        sys.exit(1)

    print(f"Loading {pred_path} ...")
    adata_pred = ad.read_h5ad(pred_path)
    print(f"Loading {real_path} ...")
    adata_real = ad.read_h5ad(real_path)

    corr, effect_real, effect_pred = compute_effect_size_corr(
        adata_pred, adata_real)

    print(f"\neffect_size_corr: {corr:.4f}")
    print(f"n_conditions: {len(effect_real)}")
    if len(effect_real) > 0:
        print(f"effect_size_real: mean={np.mean(effect_real):.1f}, "
              f"std={np.std(effect_real):.1f}, "
              f"range=[{min(effect_real)}, {max(effect_real)}]")
        print(f"effect_size_pred: mean={np.mean(effect_pred):.1f}, "
              f"std={np.std(effect_pred):.1f}, "
              f"range=[{min(effect_pred)}, {max(effect_pred)}]")

    if args.patch:
        patch_eval_results(args.eval_dir, corr)


if __name__ == "__main__":
    main()
