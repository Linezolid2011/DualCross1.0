import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
import typing as tp

from .utils import get_loss_fn

logger = logging.getLogger(__name__)


class LatentToGeneDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        gene_dim: int,
        hidden_dims: List[int] = [512, 1024],
        dropout: float = 0.1,
        residual_decoder=False,
    ):
        super().__init__()
        self.residual_decoder = residual_decoder
        if residual_decoder:
            self.blocks = nn.ModuleList()
            input_dim = latent_dim
            for hidden_dim in hidden_dims:
                block = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)
                )
                self.blocks.append(block)
                input_dim = hidden_dim
            self.final_layer = nn.Sequential(nn.Linear(input_dim, gene_dim), nn.ReLU())
        else:
            layers = []
            input_dim = latent_dim
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim
            layers.append(nn.Linear(input_dim, gene_dim))
            layers.append(nn.ReLU())
            self.decoder = nn.Sequential(*layers)

    def gene_dim(self):
        if self.residual_decoder:
            return self.final_layer[0].out_features
        else:
            for module in reversed(self.decoder):
                if isinstance(module, nn.Linear):
                    return module.out_features
            return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual_decoder:
            block_outputs = []
            current = x
            for i, block in enumerate(self.blocks):
                output = block(current)
                if i >= 1 and i % 2 == 1:
                    residual_idx = i - 1
                    output = output + block_outputs[residual_idx]
                block_outputs.append(output)
                current = output
            return self.final_layer(current)
        else:
            return self.decoder(x)


class PerturbationModel(ABC, LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        batch_dim: int = None,
        dropout: float = 0.1,
        lr: float = 3e-4,
        loss_fn: nn.Module = nn.MSELoss(),
        control_pert: str = "non-targeting",
        embed_key: Optional[str] = None,
        output_space: str = "gene",
        gene_names: Optional[List[str]] = None,
        batch_size: int = 64,
        gene_dim: int = 5000,
        hvg_dim: int = 2001,
        decoder_cfg: dict | None = None,
        **kwargs,
    ):
        super().__init__()
        self.decoder_cfg = decoder_cfg
        self.save_hyperparameters()
        self.gene_decoder_bool = kwargs.get("gene_decoder_bool", True)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pert_dim = pert_dim
        self.batch_dim = batch_dim
        self.gene_dim = gene_dim
        self.hvg_dim = hvg_dim

        if kwargs.get("batch_encoder", False):
            self.batch_dim = batch_dim
        else:
            self.batch_dim = None

        self.residual_decoder = kwargs.get("residual_decoder", False)
        self.embed_key = embed_key
        self.output_space = output_space
        if self.output_space not in {"embedding", "gene", "all"}:
            raise ValueError(
                f"Unsupported output_space '{self.output_space}'. Expected one of 'embedding', 'gene', or 'all'."
            )
        self.batch_size = batch_size
        self.control_pert = control_pert
        self.gene_names = gene_names
        self.dropout = dropout
        self.lr = lr
        self.loss_fn = get_loss_fn(loss_fn)

        if self.output_space == "embedding":
            self.gene_decoder_bool = False
            self.decoder_cfg = None
            try:
                if hasattr(self, "hparams"):
                    self.hparams["gene_decoder_bool"] = False
                    self.hparams["decoder_cfg"] = None
            except Exception:
                pass

        self._build_decoder()

    def transfer_batch_to_device(self, batch, device, dataloader_idx: int):
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    @abstractmethod
    def _build_networks(self):
        pass

    def _build_decoder(self):
        if self.gene_decoder_bool == False:
            self.gene_decoder = None
            return
        if self.decoder_cfg is None:
            self.gene_decoder = None
            return
        self.gene_decoder = LatentToGeneDecoder(**self.decoder_cfg)

    def on_load_checkpoint(self, checkpoint: dict[str, tp.Any]) -> None:
        decoder_already_configured = (
            hasattr(self, "_decoder_externally_configured") and self._decoder_externally_configured
        )
        if self.gene_decoder_bool == False:
            self.gene_decoder = None
            return

        finetune_decoder_active = False
        hparams = getattr(self, "hparams", None)
        if hparams is not None:
            if hasattr(hparams, "get"):
                finetune_decoder_active = bool(hparams.get("finetune_vci_decoder", False))
            else:
                finetune_decoder_active = bool(getattr(hparams, "finetune_vci_decoder", False))
        if not finetune_decoder_active:
            finetune_decoder_active = bool(getattr(self, "finetune_vci_decoder", False))
        if finetune_decoder_active:
            if "decoder_cfg" in checkpoint.get("hyper_parameters", {}):
                self.decoder_cfg = checkpoint["hyper_parameters"]["decoder_cfg"]
            logger.info("Finetune VCI decoder active; keeping existing decoder during checkpoint load")
            return

        if not decoder_already_configured and "decoder_cfg" in checkpoint["hyper_parameters"]:
            self.decoder_cfg = checkpoint["hyper_parameters"]["decoder_cfg"]
            self.gene_decoder = LatentToGeneDecoder(**self.decoder_cfg)
            logger.info(f"Loaded decoder from checkpoint decoder_cfg: {self.decoder_cfg}")
        elif not decoder_already_configured:
            self.decoder_cfg = None
            self._build_decoder()
            logger.info(f"DEBUG: output_space: {self.output_space}")
            if self.gene_decoder is None:
                gene_dim = self.hvg_dim if self.output_space == "gene" else self.gene_dim
                logger.info(f"DEBUG: gene_dim: {gene_dim}")
                if (self.embed_key and self.embed_key != "X_hvg" and self.output_space == "gene") or (
                    self.embed_key and self.output_space == "all"
                ):
                    logger.info("DEBUG: Creating gene_decoder, checking conditions...")
                    if gene_dim > 10000:
                        hidden_dims = [1024, 512, 256]
                    else:
                        if "DMSO_TF" in self.control_pert:
                            if self.residual_decoder:
                                hidden_dims = [2058, 2058, 2058, 2058, 2058]
                            else:
                                hidden_dims = [4096, 2048, 2048]
                        elif "PBS" in self.control_pert:
                            hidden_dims = [2048, 1024, 1024]
                        else:
                            hidden_dims = [1024, 1024, 512]
                    self.gene_decoder = LatentToGeneDecoder(
                        latent_dim=self.output_dim,
                        gene_dim=gene_dim,
                        hidden_dims=hidden_dims,
                        dropout=self.dropout,
                        residual_decoder=self.residual_decoder,
                    )
                    logger.info(f"Initialized gene decoder for embedding {self.embed_key} to gene space")
        else:
            logger.info("Decoder was already configured externally, skipping checkpoint decoder configuration")

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        pred = self(batch)
        main_loss = self.loss_fn(pred, batch["pert_cell_emb"])
        self.log("train_loss", main_loss)
        decoder_loss = None
        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            with torch.no_grad():
                latent_preds = pred.detach()
            pert_cell_counts_preds = self.gene_decoder(latent_preds)
            gene_targets = batch["pert_cell_counts"]
            decoder_loss = self.loss_fn(pert_cell_counts_preds, gene_targets)
            self.log("decoder_loss", decoder_loss)
            total_loss = main_loss + decoder_loss
        else:
            total_loss = main_loss
        return total_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        pred = self(batch)
        loss = self.loss_fn(pred, batch["pert_cell_emb"])
        self.log("val_loss", loss)
        return {"loss": loss, "predictions": pred}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        latent_output = self(batch)
        target = batch[self.embed_key]
        loss = self.loss_fn(latent_output, target)
        output_dict = {
            "preds": latent_output,
            "pert_cell_emb": batch.get("pert_cell_emb", None),
            "pert_cell_counts": batch.get("pert_cell_counts", None),
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "batch": batch.get("batch", None),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
        }
        if self.gene_decoder is not None:
            pert_cell_counts_preds = self.gene_decoder(latent_output)
            output_dict["pert_cell_counts_preds"] = pert_cell_counts_preds
            decoder_loss = self.loss_fn(pert_cell_counts_preds, batch["pert_cell_counts"])
            self.log("test_decoder_loss", decoder_loss, prog_bar=True)
        self.log("test_loss", loss, prog_bar=True)

    def predict_step(self, batch, batch_idx, **kwargs):
        latent_output = self.forward(batch)
        output_dict = {
            "preds": latent_output,
            "pert_cell_emb": batch.get("pert_cell_emb", None),
            "pert_cell_counts": batch.get("pert_cell_counts", None),
            "pert_name": batch.get("pert_name", None),
            "celltype_name": batch.get("cell_type", None),
            "batch": batch.get("batch", None),
            "ctrl_cell_emb": batch.get("ctrl_cell_emb", None),
        }
        if self.gene_decoder is not None:
            pert_cell_counts_preds = self.gene_decoder(latent_output)
            output_dict["pert_cell_counts_preds"] = pert_cell_counts_preds
        return output_dict

    def decode_to_gene_space(self, latent_embeds: torch.Tensor, basal_expr: None) -> torch.Tensor:
        if self.gene_decoder is not None:
            pert_cell_counts_preds = self.gene_decoder(latent_embeds)
            if basal_expr is not None:
                pert_cell_counts_preds += basal_expr
            return pert_cell_counts_preds
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
