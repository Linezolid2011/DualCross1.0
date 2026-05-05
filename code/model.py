"""
CrossPert Standalone: STATE + true cross-attention, no external STATE dependency.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss

from state.tx.models.base import PerturbationModel
from state.tx.models.utils import build_mlp, get_activation_class, get_transformer_backbone

logger = logging.getLogger(__name__)


class CrossAttention(nn.Module):
    """True cross-attention: perturbed queries attend to control keys/values."""

    def __init__(self, hidden_dim: int, n_heads: int = 8, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "norm_pert": nn.LayerNorm(hidden_dim),
                "norm_ctrl": nn.LayerNorm(hidden_dim),
                "attn": nn.MultiheadAttention(
                    embed_dim=hidden_dim, num_heads=n_heads,
                    dropout=dropout, batch_first=True,
                ),
                "norm_out": nn.LayerNorm(hidden_dim),
                "ffn": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                ),
            }))

    def forward(self, perturbed: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        x = perturbed
        for layer in self.layers:
            x_norm = layer["norm_pert"](x)
            ctrl_norm = layer["norm_ctrl"](control)
            attn_out, _ = layer["attn"](x_norm, ctrl_norm, ctrl_norm)
            x = x + attn_out
            x = x + layer["ffn"](layer["norm_out"](x))
        return x


class DualPerturbationClassifier(nn.Module):
    """Dual learning perturbation classifier.

    Given (control_repr, predicted_treated_repr), predict which perturbation
    was applied. This is the dual task in the dual learning framework.

    When dual_on_hidden=True, takes just the hidden-state difference
    (treated - control) as single input — a harder, more useful task.
    """

    def __init__(self, control_dim: int, pred_dim: int, n_perturbations: int,
                 dual_on_hidden: bool = False):
        super().__init__()
        self.dual_on_hidden = dual_on_hidden
        if dual_on_hidden:
            input_dim = control_dim
        else:
            input_dim = control_dim + pred_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, n_perturbations),
        )

    def forward(self, control_rep: torch.Tensor, pred_rep: torch.Tensor = None) -> torch.Tensor:
        if self.dual_on_hidden:
            x = control_rep
        else:
            x = torch.cat([control_rep, pred_rep], dim=-1)
        return self.net(x)


class ControlReconstructor(nn.Module):
    """Cycle-consistency control reconstructor.

    Given (predicted_treated, perturbation_embedding), reconstruct the
    original control state. This enables the cycle loss:
    control → treated → reconstructed_control ≈ control.
    """

    def __init__(self, pred_dim: int, pert_enc_dim: int, output_dim: int):
        super().__init__()
        input_dim = pred_dim + pert_enc_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, pred: torch.Tensor, pert_embed: torch.Tensor) -> torch.Tensor:
        B, S, _ = pred.shape
        x = torch.cat([pred, pert_embed], dim=-1)
        x = x.reshape(-1, x.shape[-1])
        return self.net(x).reshape(B, S, -1)


class CrossPertModel(PerturbationModel):
    """STATE architecture + true cross-attention refinement + dual learning.

    Adds dual learning components on top of trial#10's CrossPert:
    - DualPerturbationClassifier: classify perturbation from (control, treated) pairs
    - ControlReconstructor: cycle-consistency reconstruction of control state
    These create a closed loop: control + pert → treated → control reconstruction
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        batch_dim: int = None,
        dropout: float = 0.0,
        lr: float = 1e-4,
        loss_fn: str = "mse",
        control_pert: str = "non-targeting",
        embed_key: Optional[str] = None,
        output_space: str = "all",
        gene_names: Optional[List[str]] = None,
        batch_size: int = 16,
        gene_dim: int = 5000,
        hvg_dim: int = 2000,
        decoder_cfg: dict = None,
        # STATE params
        cell_set_len: int = 64,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        predict_residual: bool = True,
        distributional_loss: str = "energy",
        blur: float = 0.05,
        activation: str = "gelu",
        transformer_backbone_key: str = "GPT2",
        transformer_backbone_kwargs: dict = None,
        decoder_loss_weight: float = 1.0,
        # cross-attention params
        cross_attn_layers: int = 2,
        cross_attn_heads: int = 8,
        mse_aux_weight: float = 0.1,
        self_cond_prob: float = 0.0,
        # dual learning params
        dual_loss_weight: float = 0.05,
        cycle_loss_weight: float = 0.1,
        dual_on_hidden: bool = False,
        dual_warmup_steps: int = 0,
        # LR scheduler
        cosine_lr: bool = False,
        max_steps: int = 40000,
        **kwargs,
    ):
        self._params = dict(
            cell_set_len=cell_set_len,
            n_encoder_layers=n_encoder_layers,
            n_decoder_layers=n_decoder_layers,
            predict_residual=predict_residual,
            distributional_loss=distributional_loss,
            blur=blur,
            activation=activation,
            transformer_backbone_key=transformer_backbone_key,
            transformer_backbone_kwargs=transformer_backbone_kwargs or {
                "n_positions": cell_set_len,
                "hidden_size": hidden_dim,
                "n_embd": hidden_dim,
                "n_layer": 8,
                "n_head": 8,
                "resid_pdrop": 0.0,
                "embd_pdrop": 0.0,
                "attn_pdrop": 0.0,
                "use_cache": False,
            },
            decoder_loss_weight=decoder_loss_weight,
            cross_attn_layers=cross_attn_layers,
            cross_attn_heads=cross_attn_heads,
            mse_aux_weight=mse_aux_weight,
            self_cond_prob=self_cond_prob,
            dual_loss_weight=dual_loss_weight,
            cycle_loss_weight=cycle_loss_weight,
            dual_on_hidden=dual_on_hidden,
            dual_warmup_steps=dual_warmup_steps,
            cosine_lr=cosine_lr,
            max_steps=max_steps,
        )

        super().__init__(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
            pert_dim=pert_dim, batch_dim=batch_dim, dropout=dropout, lr=lr,
            loss_fn=loss_fn, control_pert=control_pert, embed_key=embed_key,
            output_space=output_space, gene_names=gene_names, batch_size=batch_size,
            gene_dim=gene_dim, hvg_dim=hvg_dim, decoder_cfg=decoder_cfg, **kwargs,
        )
        self._build_networks()

    def _build_networks(self):
        p = self._params
        self.cell_set_len = p["cell_set_len"]
        self.predict_residual = p["predict_residual"]
        self.decoder_loss_weight = p["decoder_loss_weight"]
        self.mse_aux_weight = p["mse_aux_weight"]
        self.self_cond_prob = p["self_cond_prob"]
        self.dual_loss_weight = p["dual_loss_weight"]
        self.cycle_loss_weight = p["cycle_loss_weight"]
        self.dual_on_hidden = p["dual_on_hidden"]
        self.dual_warmup_steps = p["dual_warmup_steps"]

        activation_class = get_activation_class(p["activation"])

        self.pert_encoder = build_mlp(
            in_dim=self.pert_dim, out_dim=self.hidden_dim, hidden_dim=self.hidden_dim,
            n_layers=p["n_encoder_layers"], dropout=self.dropout, activation=activation_class,
        )
        self.basal_encoder = build_mlp(
            in_dim=self.input_dim, out_dim=self.hidden_dim, hidden_dim=self.hidden_dim,
            n_layers=p["n_encoder_layers"], dropout=self.dropout, activation=activation_class,
        )
        bb_kwargs = dict(p["transformer_backbone_kwargs"])
        bb_kwargs["n_positions"] = p["cell_set_len"]
        self.transformer_backbone, self.transformer_model_dim = get_transformer_backbone(
            p["transformer_backbone_key"], bb_kwargs,
        )
        self.project_out = build_mlp(
            in_dim=self.hidden_dim, out_dim=self.output_dim, hidden_dim=self.hidden_dim,
            n_layers=p["n_decoder_layers"], dropout=self.dropout, activation=activation_class,
        )
        if self.output_space == "all":
            self.final_down_then_up = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim // 8),
                nn.GELU(),
                nn.Linear(self.output_dim // 8, self.output_dim),
            )
        self.relu = nn.ReLU()
        self.energy_loss = SamplesLoss(loss=p["distributional_loss"], blur=p["blur"])

        self.cross_attn = CrossAttention(
            hidden_dim=self.hidden_dim,
            n_heads=p["cross_attn_heads"],
            n_layers=p["cross_attn_layers"],
            dropout=self.dropout,
        )

        # === Dual Learning Components ===
        self.dual_classifier = DualPerturbationClassifier(
            control_dim=self.hidden_dim,
            pred_dim=self.output_dim,
            n_perturbations=self.pert_dim,
            dual_on_hidden=self.dual_on_hidden,
        )
        self.control_reconstructor = ControlReconstructor(
            pred_dim=self.output_dim,
            pert_enc_dim=self.hidden_dim,
            output_dim=self.output_dim,
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear,)) and not any(
            module is p for p in self.transformer_backbone.parameters()
        ):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, batch: dict, padded=True):
        """Forward pass returning prediction and intermediate representations.

        Returns:
            pred: [B*S, output_dim] — predicted perturbed state
            extras: dict of intermediate tensors for dual learning loss computation
        """
        if padded:
            pert = batch["pert_emb"].reshape(-1, self.cell_set_len, self.pert_dim)
            basal = batch["ctrl_cell_emb"].reshape(-1, self.cell_set_len, self.input_dim)
        else:
            pert = batch["pert_emb"].reshape(1, -1, self.pert_dim)
            basal = batch["ctrl_cell_emb"].reshape(1, -1, self.input_dim)

        pert_embedding = self.pert_encoder(pert)  # [B, S, H]
        control_cells = self.basal_encoder(basal)  # [B, S, H]
        combined_input = pert_embedding + control_cells

        outputs = self.transformer_backbone(inputs_embeds=combined_input)
        transformer_output = outputs.last_hidden_state  # [B, S, H]

        refined_output = self.cross_attn(transformer_output, control_cells)

        if self.predict_residual and self.output_space == "all":
            out_pred = self.project_out(refined_output) + basal
            out_pred = self.final_down_then_up(out_pred)
        elif self.predict_residual:
            out_pred = self.project_out(refined_output + control_cells)
        else:
            out_pred = self.project_out(refined_output)

        out_pred = self.relu(out_pred)

        extras = {
            "control_repr": control_cells,      # [B, S, H]
            "pert_embedding": pert_embedding,     # [B, S, H]
            "refined_output": refined_output,     # [B, S, H]
            "basal_raw": basal,                   # [B, S, input_dim]
        }

        return out_pred.reshape(-1, self.output_dim), extras

    def training_step(self, batch, batch_idx, padded=True):
        pred, extras = self.forward(batch, padded=padded)
        target = batch["pert_cell_emb"]

        if padded:
            pred_3d = pred.reshape(-1, self.cell_set_len, self.output_dim)
            target_3d = target.reshape(-1, self.cell_set_len, self.output_dim)
        else:
            pred_3d = pred.reshape(1, -1, self.output_dim)
            target_3d = target.reshape(1, -1, self.output_dim)

        # === Primary loss (energy + MSE) ===
        energy = self.energy_loss(pred_3d, target_3d).mean()
        self.log("train_energy", energy, prog_bar=True)

        mse = F.mse_loss(pred_3d, target_3d)
        self.log("train_mse", mse, prog_bar=True)

        main_loss = energy + self.mse_aux_weight * mse
        self.log("train_loss_main", main_loss)

        total_loss = main_loss

        # === Curriculum ramp for dual learning weights ===
        if self.dual_warmup_steps > 0 and self.trainer is not None:
            ramp = min(1.0, self.trainer.global_step / self.dual_warmup_steps)
        else:
            ramp = 1.0
        current_dual_weight = self.dual_loss_weight * ramp
        current_cycle_weight = self.cycle_loss_weight * ramp
        if ramp < 1.0:
            self.log("dual_ramp", ramp, prog_bar=True)

        # === Dual Learning: Perturbation Classification ===
        if self.dual_on_hidden:
            # Classify from hidden-state difference: treated - control
            # This forces the model to encode perturbation-specific changes
            # in the hidden representation itself.
            hidden_diff = extras["refined_output"] - extras["control_repr"]  # [B, S, H]
            hidden_diff_pooled = hidden_diff.mean(dim=1)                     # [B, H]
            dual_logits = self.dual_classifier(hidden_diff_pooled)
        else:
            # Pool over cell dimension -> per-batch-item representations
            control_pooled = extras["control_repr"].mean(dim=1)   # [B, H]
            pred_pooled = pred_3d.mean(dim=1)                     # [B, output_dim]
            dual_logits = self.dual_classifier(control_pooled, pred_pooled)

        # Perturbation target: one-hot from batch, same for all cells in set
        pert_onehot = batch["pert_emb"]
        if padded:
            pert_onehot = pert_onehot.reshape(-1, self.cell_set_len, self.pert_dim)
        else:
            pert_onehot = pert_onehot.reshape(1, -1, self.pert_dim)
        pert_target = pert_onehot[:, 0, :].argmax(dim=-1)     # [B]

        dual_loss = F.cross_entropy(dual_logits, pert_target)
        self.log("train_dual_loss", dual_loss, prog_bar=True)

        total_loss = total_loss + current_dual_weight * dual_loss

        # === Dual Learning: Cycle Consistency ===
        # Reconstruct control from predicted treated + perturbation
        recon_control = self.control_reconstructor(
            pred_3d, extras["pert_embedding"]
        )  # [B, S, output_dim]

        # Target: original control cells in the same space
        cycle_target = extras["basal_raw"]  # [B, S, input_dim]
        if self.output_space == "all" and self.predict_residual:
            # out_pred was produced by residual + final_down_then_up
            # but the cycle target should match the reconstruction space
            cycle_target = cycle_target  # already in [B, S, input_dim]
        cycle_loss = F.mse_loss(recon_control, cycle_target)
        self.log("train_cycle_loss", cycle_loss, prog_bar=True)

        total_loss = total_loss + current_cycle_weight * cycle_loss

        # === Decoder loss (unchanged) ===
        if self.gene_decoder is not None and "pert_cell_counts" in batch:
            latent_preds = pred.detach()
            gene_preds = self.gene_decoder(latent_preds)
            gene_targets = batch["pert_cell_counts"]
            if padded:
                gene_preds = gene_preds.reshape(-1, self.cell_set_len, self.gene_decoder.gene_dim())
                gene_targets = gene_targets.reshape(-1, self.cell_set_len, self.gene_decoder.gene_dim())
            decoder_loss = self.energy_loss(gene_preds, gene_targets).mean()
            self.log("decoder_loss", decoder_loss)
            total_loss = total_loss + self.decoder_loss_weight * decoder_loss

        self.log("train_loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        pred, _ = self.forward(batch)
        target = batch["pert_cell_emb"]
        pred_3d = pred.reshape(-1, self.cell_set_len, self.output_dim)
        target_3d = target.reshape(-1, self.cell_set_len, self.output_dim)
        loss = self.energy_loss(pred_3d, target_3d).mean()
        self.log("val_loss", loss, prog_bar=True)
        return {"loss": loss}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        if self._params.get("cosine_lr"):
            from torch.optim.lr_scheduler import CosineAnnealingLR
            max_steps = self._params.get("max_steps", 40000)
            sched = CosineAnnealingLR(opt, T_max=max_steps, eta_min=1e-6)
            return {
                "optimizer": opt,
                "lr_scheduler": {"scheduler": sched, "interval": "step"},
            }
        return opt

    def predict_step(self, batch, batch_idx, **kwargs):
        latent_output, _ = self.forward(batch, padded=False)
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
            output_dict["pert_cell_counts_preds"] = self.gene_decoder(latent_output)
        return output_dict
