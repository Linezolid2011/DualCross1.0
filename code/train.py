"""
CrossPert Standalone Training Script — no external STATE dependency.
"""

import argparse
import json
import logging
import os
import pickle
import shutil
import sys
from os.path import exists, join

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import lightning.pytorch as pl
import torch
from cell_load.data_modules import PerturbationDataModule
from cell_load.utils.modules import get_datamodule
from lightning.pytorch.callbacks import ModelCheckpoint

from model import CrossPertModel
from callbacks import LossPlotCallback

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train CrossPert (standalone)")
    parser.add_argument("--toml", required=True, help="Path to TOML config")
    parser.add_argument("--output-dir", default=join(BASE_DIR, "checkpoints"), help="Output directory for checkpoints")
    parser.add_argument("--charts-dir", default=join(BASE_DIR, "charts"), help="Output directory for loss plots")
    parser.add_argument("--name", default="crosspert", help="Run name")
    parser.add_argument("--embed-key", default="X_hvg", help="Embedding key in h5ad")
    parser.add_argument("--pert-col", default="drugname_drugconc", help="Perturbation column")
    parser.add_argument("--cell-type-key", default="cell_name", help="Cell type column")
    parser.add_argument("--batch-col", default="plate", help="Batch column")
    parser.add_argument("--output-space", default="all", help="Output space: gene, all, embedding")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=40000)
    parser.add_argument("--val-freq", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--cell-set-len", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--n-blocks", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--cross-attn-layers", type=int, default=2)
    parser.add_argument("--cross-attn-heads", type=int, default=8)
    parser.add_argument("--mse-aux-weight", type=float, default=0.1)
    # Dual learning args
    parser.add_argument("--dual-loss-weight", type=float, default=0.05, help="Weight for dual perturbation classification loss")
    parser.add_argument("--cycle-loss-weight", type=float, default=0.1, help="Weight for cycle consistency reconstruction loss")
    parser.add_argument("--dual-on-hidden", action="store_true", help="Use hidden state difference for dual classifier instead of output space")
    parser.add_argument("--dual-warmup-steps", type=int, default=0, help="Linearly warmup dual/cycle loss weights over this many steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cosine-lr", action="store_true", help="Use cosine LR schedule from lr to 1e-6 over max_steps")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--devices", type=int, default=1)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(args.seed)

    run_output_dir = join(args.output_dir, args.name)
    if os.path.exists(run_output_dir) and args.overwrite:
        print(f"Output dir {run_output_dir} already exists, overwriting")
        shutil.rmtree(run_output_dir)
    os.makedirs(run_output_dir, exist_ok=True)

    config = vars(args)
    with open(join(run_output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    control_pert = "[('DMSO_TF', 0.0, 'uM')]" if args.pert_col == "drugname_drugconc" else "DMSO_TF"

    data_kwargs = {
        "toml_config_path": args.toml,
        "embed_key": args.embed_key,
        "output_space": args.output_space,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "batch_col": args.batch_col,
        "pert_col": args.pert_col,
        "cell_type_key": args.cell_type_key,
        "control_pert": control_pert,
        "basal_mapping_strategy": "random",
        "n_basal_samples": 1,
        "should_yield_control_cells": True,
        "map_controls": True,
        "barcode": True,
    }

    data_module: PerturbationDataModule = get_datamodule(
        "PerturbationDataModule",
        data_kwargs,
        batch_size=args.batch_size,
        cell_sentence_len=args.cell_set_len,
    )

    with open(join(run_output_dir, "data_module.torch"), "wb") as f:
        data_module.save_state(f)

    data_module.setup(stage="fit")
    dl = data_module.train_dataloader()
    actual_bs = getattr(dl.batch_sampler, "batch_size", dl.batch_size) if dl.batch_sampler else dl.batch_size
    print(f"num_workers: {dl.num_workers}, batch_size: {actual_bs}")

    var_dims = data_module.get_var_dims()
    _LONG_KEYS = {"gene_names": "genes", "pert_names": "perts"}
    log_var_dims = {
        k: (f"<{len(v)} {_LONG_KEYS.get(k, 'items')}>" if k in _LONG_KEYS else v)
        for k, v in var_dims.items()
    }
    print(f"var_dims: {log_var_dims}")

    if args.output_space == "gene":
        gene_dim = var_dims.get("hvg_dim", 2000)
    else:
        gene_dim = var_dims.get("gene_dim", 2000)
    latent_dim = var_dims["output_dim"]
    hidden_dims = [1024, 1024, 512]

    decoder_cfg = None
    if args.output_space in {"gene", "all"}:
        decoder_cfg = dict(
            latent_dim=latent_dim,
            gene_dim=gene_dim,
            hidden_dims=hidden_dims,
            dropout=0.1,
            residual_decoder=False,
        )

    with open(join(run_output_dir, "cell_type_onehot_map.pkl"), "wb") as f:
        pickle.dump(data_module.cell_type_onehot_map, f)
    torch.save(data_module.pert_onehot_map, join(run_output_dir, "pert_onehot_map.pt"))
    with open(join(run_output_dir, "batch_onehot_map.pkl"), "wb") as f:
        pickle.dump(data_module.batch_onehot_map, f)
    with open(join(run_output_dir, "var_dims.pkl"), "wb") as f:
        pickle.dump(var_dims, f)

    model = CrossPertModel(
        input_dim=var_dims["input_dim"],
        hidden_dim=args.hidden_dim,
        output_dim=var_dims["output_dim"],
        pert_dim=var_dims["pert_dim"],
        batch_dim=var_dims["batch_dim"],
        dropout=0.0,
        lr=args.lr,
        loss_fn="mse",
        control_pert=control_pert,
        embed_key=args.embed_key,
        output_space=args.output_space,
        gene_names=var_dims.get("gene_names"),
        batch_size=args.batch_size,
        gene_dim=gene_dim,
        hvg_dim=var_dims.get("hvg_dim", 2000),
        decoder_cfg=decoder_cfg,
        cell_set_len=args.cell_set_len,
        n_encoder_layers=4,
        n_decoder_layers=4,
        predict_residual=True,
        distributional_loss="energy",
        blur=0.05,
        activation="gelu",
        transformer_backbone_key="GPT2",
        transformer_backbone_kwargs={
            "n_positions": args.cell_set_len,
            "hidden_size": args.hidden_dim,
            "n_embd": args.hidden_dim,
            "n_layer": 8,
            "n_head": args.n_heads,
            "resid_pdrop": 0.0,
            "embd_pdrop": 0.0,
            "attn_pdrop": 0.0,
            "use_cache": False,
        },
        decoder_loss_weight=1.0,
        cross_attn_layers=args.cross_attn_layers,
        cross_attn_heads=args.cross_attn_heads,
        mse_aux_weight=args.mse_aux_weight,
        # Dual learning params
        dual_loss_weight=args.dual_loss_weight,
        cycle_loss_weight=args.cycle_loss_weight,
        dual_on_hidden=args.dual_on_hidden,
        dual_warmup_steps=args.dual_warmup_steps,
        cosine_lr=args.cosine_lr,
        max_steps=args.max_steps,
    )

    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
    print(f"Model created. Estimated params size: {param_size:.2f} GB")
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    from state.tx.utils import get_loggers
    loggers = get_loggers(
        output_dir=args.output_dir,
        name=args.name,
        wandb_project="crosspert",
        wandb_entity="",
        local_wandb_dir=join(run_output_dir, "wandb"),
        use_wandb=False,
        cfg=config,
    )

    checkpoint_dir = join(args.output_dir, args.name, "checkpoints")
    ckpt_periodic = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="step-{step}",
        save_last=True,
        every_n_train_steps=args.val_freq,
        save_top_k=-1,
    )
    ckpt_best_train = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-train",
        monitor="train_loss",
        mode="min",
        save_top_k=1,
        every_n_train_steps=args.val_freq,
    )

    os.makedirs(args.charts_dir, exist_ok=True)
    loss_plot = LossPlotCallback(save_dir=args.charts_dir, plot_freq=200)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=args.devices,
        max_steps=args.max_steps,
        logger=loggers,
        callbacks=[ckpt_periodic, ckpt_best_train, loss_plot],
        gradient_clip_val=10,
        use_distributed_sampler=False,
        limit_val_batches=0,
    )

    checkpoint_path = join(checkpoint_dir, "last.ckpt")
    if not exists(checkpoint_path):
        checkpoint_path = None
    else:
        print(f"Resuming from {checkpoint_path}")

    import numpy as _np
    _np_core = getattr(_np, "_core", _np.core)
    torch.serialization.add_safe_globals([
        _np_core.multiarray.scalar, _np.dtype, _np.int64, _np.float64,
        _np.dtypes.Int64DType, _np.dtypes.Float64DType, _np.dtypes.Float32DType,
        _np.dtypes.Int32DType, _np.dtypes.UInt8DType, _np.dtypes.BoolDType,
    ])

    print("Starting training...")
    trainer.fit(model, datamodule=data_module, ckpt_path=checkpoint_path)
    print("Training completed.")

    final_ckpt = join(checkpoint_dir, "final.ckpt")
    if not exists(final_ckpt):
        trainer.save_checkpoint(final_ckpt)
    print(f"Final checkpoint saved to {final_ckpt}")


if __name__ == "__main__":
    main()
