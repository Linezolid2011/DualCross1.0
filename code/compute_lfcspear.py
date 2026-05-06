#!/usr/bin/env python3
"""Compute LFCSpear (de_spearman_lfc_sig) from pred_de.csv / real_de.csv.

Workaround for cell_eval dtype bug: "cannot build list with different dtypes".
Reads the CSV files saved by cell_eval and computes the Spearman correlation
of log fold changes for significant genes independently.
"""

import argparse
import json
import os

import pandas as pd
from scipy.stats import spearmanr


def compute_lfcspear(pred_csv, real_csv, fdr_threshold=0.05):
    pred = pd.read_csv(pred_csv)
    real = pd.read_csv(real_csv)

    # Find significant genes in real data
    sig_mask = real["fdr"] < fdr_threshold
    sig_genes = real[sig_mask][["target", "feature"]].drop_duplicates()

    if sig_genes.empty:
        print("WARNING: no significant genes found")
        return {}, 0.0

    # Rename before merge to avoid ambiguous suffixes
    pred_fc = pred[["target", "feature", "fold_change"]].rename(columns={"fold_change": "fc_pred"})
    real_fc = real[["target", "feature", "fold_change"]].rename(columns={"fold_change": "fc_real"})

    merged = sig_genes.merge(pred_fc, on=["target", "feature"])
    merged = merged.merge(real_fc, on=["target", "feature"])

    per_pert = {}
    for pert, group in merged.groupby("target"):
        if len(group) < 3:
            continue
        corr, _ = spearmanr(group["fc_real"], group["fc_pred"])
        if not pd.isna(corr):
            per_pert[pert] = float(corr)

    mean_corr = float(pd.Series(list(per_pert.values())).mean()) if per_pert else 0.0
    return per_pert, mean_corr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True, help="Directory with pred_de.csv and real_de.csv")
    parser.add_argument("--fdr", type=float, default=0.05)
    args = parser.parse_args()

    pred_csv = os.path.join(args.eval_dir, "pred_de.csv")
    real_csv = os.path.join(args.eval_dir, "real_de.csv")

    if not os.path.exists(pred_csv) or not os.path.exists(real_csv):
        print(f"ERROR: missing pred_de.csv or real_de.csv in {args.eval_dir}")
        return

    per_pert, mean_corr = compute_lfcspear(pred_csv, real_csv, args.fdr)

    print(f"LFCSpear (fdr<{args.fdr}): {mean_corr:.4f}  (n_perts={len(per_pert)})")

    # Patch into existing eval_results.json
    results_path = os.path.join(args.eval_dir, "eval_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        results["de_spearman_lfc_sig"] = mean_corr
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Patched de_spearman_lfc_sig={mean_corr:.4f} into {results_path}")


if __name__ == "__main__":
    main()
