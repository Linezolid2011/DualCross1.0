import csv
import logging
import os
from contextlib import contextmanager
from os.path import join

from lightning.pytorch.loggers import CSVLogger as BaseCSVLogger


class RobustCSVLogger(BaseCSVLogger):
    """CSV logger that handles dynamic metrics by allowing new columns mid-training."""

    def log_metrics(self, metrics, step):
        try:
            super().log_metrics(metrics, step)
        except ValueError as e:
            if "dict contains fields not in fieldnames" in str(e):
                self._recreate_csv_with_new_fields(metrics)
                super().log_metrics(metrics, step)
            else:
                raise e

    def _recreate_csv_with_new_fields(self, new_metrics):
        if not hasattr(self.experiment, "metrics_file_path"):
            return
        existing_data = []
        csv_file = self.experiment.metrics_file_path
        if os.path.exists(csv_file):
            with open(csv_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                existing_data = list(reader)
        all_fieldnames = set()
        for row in existing_data:
            all_fieldnames.update(row.keys())
        all_fieldnames.update(new_metrics.keys())
        sorted_fieldnames = sorted(all_fieldnames)
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted_fieldnames)
            writer.writeheader()
            for row in existing_data:
                writer.writerow(row)
        self.experiment.metrics_keys = sorted_fieldnames


def get_loggers(
    output_dir: str,
    name: str,
    wandb_project: str = "",
    wandb_entity: str = "",
    local_wandb_dir: str = "",
    use_wandb: bool = False,
    use_csv: bool = True,
    cfg: dict = None,
):
    loggers = []
    if use_csv:
        csv_logger = RobustCSVLogger(save_dir=output_dir, name=name, version=0)
        loggers.append(csv_logger)
    if use_wandb:
        try:
            import wandb
            from lightning.pytorch.loggers import WandbLogger
            wandb_logger = WandbLogger(
                name=name, project=wandb_project, entity=wandb_entity,
                dir=local_wandb_dir, tags=cfg.get("wandb", {}).get("tags", []) if cfg else [],
            )
            if cfg is not None:
                wandb_logger.experiment.config.update(cfg)
            loggers.append(wandb_logger)
        except ImportError:
            print("Warning: wandb is not installed. Skipping wandb logging.")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb logger: {e}")
    if not loggers:
        print("Warning: No loggers configured. Adding robust CSV logger as fallback.")
        loggers.append(RobustCSVLogger(save_dir=output_dir, name=name, version=0))
    return loggers
