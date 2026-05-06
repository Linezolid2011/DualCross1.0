"""Plot evaluation results across all checkpoints, one chart per metric."""

import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_step(name: str) -> tuple:
    if name == "best-train":
        return (float("inf"), "best-train")
    if name == "last":
        return (float("inf") - 1, "last")
    if name == "final":
        return (float("inf") - 2, "final")
    m = re.match(r"step-step=(\d+)", name)
    if m:
        return (int(m.group(1)), name)
    return (0, name)


def main():
    eval_root = sys.argv[1] if len(sys.argv) > 1 else None
    if eval_root is None or not os.path.isdir(eval_root):
        print(f"Usage: {sys.argv[0]} <eval_results_dir>")
        sys.exit(1)

    results = {}
    skip = {"best-train", "final", "last"}
    checkpoints = []
    for entry in sorted(os.listdir(eval_root)):
        if entry in skip:
            continue
        rfile = os.path.join(eval_root, entry, "eval_results.json")
        if not os.path.isfile(rfile):
            continue
        with open(rfile) as f:
            data = json.load(f)
        sk, label = parse_step(entry)
        checkpoints.append((sk, label, data))
    checkpoints.sort(key=lambda x: x[0])

    if len(checkpoints) < 2:
        print(f"Need at least 2 checkpoints, found {len(checkpoints)}")
        return

    all_metrics = set()
    for _, _, data in checkpoints:
        all_metrics.update(k for k in data if isinstance(data[k], (int, float)))

    labels = [c[1] for c in checkpoints]
    x = range(len(labels))
    out_dir = os.path.join(eval_root, "_plots")
    os.makedirs(out_dir, exist_ok=True)

    for metric in sorted(all_metrics):
        values = [c[2].get(metric) for c in checkpoints]
        if all(v is None for v in values):
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, values, marker="o", linestyle="-", linewidth=1.5, markersize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} across training steps")
        ax.grid(True, alpha=0.3)
        for i, v in enumerate(values):
            if v is not None:
                ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{metric}.png"), dpi=150)
        plt.close(fig)
    print(f"Plots saved to {out_dir}/ ({len(all_metrics)} metrics)")


if __name__ == "__main__":
    main()
