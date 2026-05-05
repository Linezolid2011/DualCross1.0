"""Training callbacks: loss tracking and live plotting."""

import os

import lightning.pytorch as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class LossPlotCallback(pl.Callback):
    """Records loss per step and periodically saves a loss curve plot."""

    def __init__(self, save_dir: str, plot_freq: int = 200):
        super().__init__()
        self.save_dir = save_dir
        self.plot_freq = plot_freq
        os.makedirs(save_dir, exist_ok=True)
        self.steps: list[int] = []
        self.losses: list[float] = []

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule,
        outputs, batch, batch_idx: int,
    ):
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            step = trainer.global_step
            self.steps.append(step)
            self.losses.append(float(loss))
            self._save_csv()
            if step % self.plot_freq == 0 and step > 0:
                self._plot()

    def _save_csv(self):
        path = os.path.join(self.save_dir, "losses.csv")
        with open(path, "w") as f:
            f.write("step,loss\n")
            for s, l in zip(self.steps, self.losses):
                f.write(f"{s},{l:.6f}\n")

    def _plot(self):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.steps, self.losses, linewidth=0.5)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, "loss_curve.png"), dpi=150)
        plt.close(fig)
