"""
Evaluate CrossPert on Tahoe ver2 test set — standalone.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from collections import defaultdict

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from tqdm import tqdm

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from model import CrossPertModel

logger = logging.getLogger(__name__)


def load_model(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = checkpoint["hyper_parameters"]
    model = CrossPertModel(**hparams)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def get_onehot(value, onehot_map):
    if value in onehot_map:
        return onehot_map[value]
    return torch.zeros(len(next(iter(onehot_map.values()))))


def predict_condition(model, ctrl_cells, pert_onehot, cell_set_len, device):
    n_cells = ctrl_cells.shape[0]
    if n_cells == 0:
        return None
    all_preds = []
    for start in range(0, n_cells, cell_set_len):
        end = min(start + cell_set_len, n_cells)
        chunk = ctrl_cells[start:end]
        chunk_size = chunk.shape[0]
        if chunk_size < cell_set_len:
            pad = torch.zeros(cell_set_len - chunk_size, chunk.shape[1])
            chunk_padded = torch.cat([chunk, pad], dim=0)
        else:
            chunk_padded = chunk
        pert_expanded = pert_onehot.unsqueeze(0).expand(cell_set_len, -1)
        batch = {
            "ctrl_cell_emb": chunk_padded.to(device),
            "pert_emb": pert_expanded.to(device),
        }
        with torch.no_grad():
            pred, _ = model.forward(batch, padded=True)
        pred = pred[:chunk_size].cpu()
        all_preds.append(pred)
    return torch.cat(all_preds, dim=0)


def run_evaluation(model, test_h5ad_path, train_h5ad_path, output_dir,
                   pert_onehot_map_path, cell_set_len=64, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Loading test data from {test_h5ad_path}...")
    test_adata = ad.read_h5ad(test_h5ad_path)
    print(f"Loading train data from {train_h5ad_path}...")
    train_adata = ad.read_h5ad(train_h5ad_path)
    print(f"Loading pert onehot map from {pert_onehot_map_path}...")
    pert_onehot_map = torch.load(pert_onehot_map_path, map_location="cpu", weights_only=False)

    test_hvg = torch.tensor(
        test_adata.obsm["X_hvg"].toarray() if sparse.issparse(test_adata.obsm["X_hvg"])
        else np.array(test_adata.obsm["X_hvg"]), dtype=torch.float32,
    )
    train_hvg = torch.tensor(
        train_adata.obsm["X_hvg"].toarray() if sparse.issparse(train_adata.obsm["X_hvg"])
        else np.array(train_adata.obsm["X_hvg"]), dtype=torch.float32,
    )

    print("Building control cell index...")
    ctrl_index = defaultdict(list)
    for i, (cn, drug) in enumerate(
        zip(train_adata.obs["cell_name"], train_adata.obs["drugname_drugconc"])
    ):
        if "DMSO_TF" in str(drug):
            ctrl_index[cn].append(i)
    test_ctrl_mask = test_adata.obs["split"] == "test_control"
    for i in np.where(test_ctrl_mask)[0]:
        cn = test_adata.obs["cell_name"].iloc[i]
        ctrl_index[cn].append(-(i + 1))

    print(f"Control cells available for {len(ctrl_index)} cell lines")

    test_pert_mask = test_adata.obs["split"] != "test_control"
    test_pert_adata = test_adata[test_pert_mask].copy()
    conditions = test_pert_adata.obs.groupby(["cell_name", "drugname_drugconc"]).indices
    print(f"Evaluating {len(conditions)} test conditions...")

    all_pred_hvg, all_real_hvg, all_pert_names, all_celltype_names = [], [], [], []

    for (cell_name, drug), indices in tqdm(conditions.items(), desc="Predicting"):
        ctrl_idxs = ctrl_index.get(cell_name, [])
        if not ctrl_idxs:
            continue
        ctrl_cells_list = []
        for idx in ctrl_idxs:
            if idx >= 0:
                ctrl_cells_list.append(train_hvg[idx])
            else:
                ctrl_cells_list.append(test_hvg[-(idx + 1)])
        ctrl_cells = torch.stack(ctrl_cells_list)
        drug_str = str(drug)
        pert_oh = get_onehot(drug_str, pert_onehot_map)
        n_test = len(indices)
        if len(ctrl_cells) >= n_test:
            sampled_idx = torch.randperm(len(ctrl_cells))[:n_test]
            ctrl_sample = ctrl_cells[sampled_idx]
        else:
            repeats = (n_test // len(ctrl_cells)) + 1
            ctrl_sample = ctrl_cells.repeat(repeats, 1)[:n_test]

        preds = predict_condition(model, ctrl_sample, pert_oh, cell_set_len, device)
        if preds is None:
            continue
        real_subset = torch.tensor(
            test_pert_adata.obsm["X_hvg"][indices].toarray()
            if sparse.issparse(test_pert_adata.obsm["X_hvg"])
            else np.array(test_pert_adata.obsm["X_hvg"][indices]), dtype=torch.float32,
        )
        all_pred_hvg.append(preds.numpy())
        all_real_hvg.append(real_subset.numpy())
        all_pert_names.extend([drug_str] * n_test)
        all_celltype_names.extend([cell_name] * n_test)

    pred_X = np.clip(np.vstack(all_pred_hvg), 0, 14)
    real_X = np.clip(np.vstack(all_real_hvg), 0, 14)
    print(f"Total predictions: {pred_X.shape[0]} cells")

    ctrl_hvg_list, ctrl_pert_names, ctrl_celltype_names = [], [], []
    control_label = "[('DMSO_TF', 0.0, 'uM')]"
    for ct in set(all_celltype_names):
        idxs = ctrl_index.get(ct, [])
        ctrl_cells_ct = []
        for idx in idxs[:200]:
            if idx >= 0:
                ctrl_cells_ct.append(train_hvg[idx].numpy())
            else:
                ctrl_cells_ct.append(test_hvg[-(idx + 1)].numpy())
        if ctrl_cells_ct:
            ctrl_arr = np.clip(np.stack(ctrl_cells_ct), 0, 14)
            ctrl_hvg_list.append(ctrl_arr)
            ctrl_pert_names.extend([control_label] * len(ctrl_arr))
            ctrl_celltype_names.extend([ct] * len(ctrl_arr))

    if ctrl_hvg_list:
        ctrl_X = np.vstack(ctrl_hvg_list)
        full_pred_X = np.vstack([pred_X, ctrl_X])
        full_real_X = np.vstack([real_X, ctrl_X])
        full_pert_names = all_pert_names + ctrl_pert_names
        full_celltype_names = all_celltype_names + ctrl_celltype_names
    else:
        full_pred_X, full_real_X = pred_X, real_X
        full_pert_names = all_pert_names
        full_celltype_names = all_celltype_names

    obs_df = pd.DataFrame({"perturbation": full_pert_names, "cell_type": full_celltype_names})
    obs_df.index = [f"cell_{i}" for i in range(len(obs_df))]
    adata_pred = ad.AnnData(X=full_pred_X, obs=obs_df.copy())
    adata_real = ad.AnnData(X=full_real_X, obs=obs_df.copy())
    os.makedirs(output_dir, exist_ok=True)
    adata_pred.write_h5ad(os.path.join(output_dir, "pred.h5ad"))
    adata_real.write_h5ad(os.path.join(output_dir, "real.h5ad"))
    print(f"Saved pred/real to {output_dir}")

    return evaluate_with_cell_eval(adata_pred, adata_real, output_dir)


def evaluate_with_cell_eval(adata_pred, adata_real, output_dir):
    try:
        from cell_eval import MetricsEvaluator
        evaluator = MetricsEvaluator(
            adata_pred=adata_pred, adata_real=adata_real,
            pert_col="perturbation", control_pert="[('DMSO_TF', 0.0, 'uM')]",
            outdir=output_dir, skip_de=False,
        )
        _, agg_results = evaluator.compute(profile="full")
        # agg_results = describe() of per-pert DataFrame
        # Col 0 = stat names ("count", "mean", "std", ...)
        # Cols 1+ = metric names (R2, DEOver, ...)
        results = {}
        stat_col = agg_results.columns[0]
        for row in agg_results.iter_rows(named=True):
            if row.get(stat_col) == "mean":
                for metric_name, val in row.items():
                    if metric_name != stat_col:
                        results[str(metric_name)] = float(val)
                break
        if not results:
            for col in agg_results.columns[1:]:
                try:
                    vals = {str(row[stat_col]): float(row[col]) for row in agg_results.iter_rows(named=True)}
                    results[col] = vals.get("mean", vals.get("50%", None))
                except Exception:
                    pass
        print(agg_results)

        # Add R2 (not in cell_eval) using the basic metrics function
        basic = compute_basic_metrics(adata_pred, adata_real)
        if "R2" in basic:
            results["R2"] = basic["R2"]

    except Exception as e:
        print(f"cell-eval failed: {e}")
        import traceback; traceback.print_exc()
        results = compute_basic_metrics(adata_pred, adata_real)

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS — CrossPert Standalone")
    print("=" * 70)
    for metric, value in sorted(results.items()):
        if isinstance(value, (float, np.floating)):
            print(f"  {metric:20s}: {value:.4f}")
        else:
            print(f"  {metric:20s}: {value}")
    print("=" * 70)

    results_path = os.path.join(output_dir, "eval_results.json")
    serializable = {}
    for k, v in results.items():
        if isinstance(v, (float, np.floating)):
            serializable[k] = float(v)
        elif isinstance(v, (int, np.integer)):
            serializable[k] = int(v)
        else:
            serializable[k] = str(v)
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {results_path}")
    return results


def compute_basic_metrics(adata_pred, adata_real):
    from scipy.stats import pearsonr
    pred_mean = np.mean(adata_pred.X, axis=0)
    real_mean = np.mean(adata_real.X, axis=0)
    r2 = pearsonr(pred_mean, real_mean)[0] ** 2
    mse = np.mean((pred_mean - real_mean) ** 2)
    mae = np.mean(np.abs(pred_mean - real_mean))
    corrs = []
    for p in adata_pred.obs["perturbation"].unique():
        mask = adata_pred.obs["perturbation"] == p
        pm = adata_pred.X[mask].mean(axis=0)
        rm = adata_real.X[mask].mean(axis=0)
        if np.std(pm) > 0 and np.std(rm) > 0:
            corrs.append(pearsonr(pm, rm)[0])
    return {"R2": r2, "MSE": mse, "MAE": mae, "PDCorr": np.mean(corrs) if corrs else 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--test-h5ad", default="/home/dataset-assist-0/cuimenglong/workspace/data/tahoe/ver2/test_new.h5ad")
    parser.add_argument("--train-h5ad", default="/home/dataset-assist-0/cuimenglong/workspace/data/tahoe/ver2/train.h5ad")
    parser.add_argument("--pert-map", default=None)
    parser.add_argument("--output-dir", default="/tmp/crosspert_eval")
    parser.add_argument("--cell-set-len", type=int, default=64)
    args = parser.parse_args()

    if args.pert_map is None:
        ckpt_dir = os.path.dirname(os.path.dirname(args.ckpt))
        args.pert_map = os.path.join(ckpt_dir, "pert_onehot_map.pt")

    model = load_model(args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    run_evaluation(
        model=model, test_h5ad_path=args.test_h5ad, train_h5ad_path=args.train_h5ad,
        output_dir=args.output_dir, pert_onehot_map_path=args.pert_map,
        cell_set_len=args.cell_set_len, device=device,
    )


if __name__ == "__main__":
    main()
