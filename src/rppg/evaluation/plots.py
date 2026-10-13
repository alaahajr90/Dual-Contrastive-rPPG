"""Figures are generated from actual predictions; reference metrics are never plotted as runs."""
from pathlib import Path
import numpy as np


def plot_hr_figures(rows, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    truth, prediction = (np.asarray([r[k] for r in rows], dtype=float) for k in ("gt_hr", "pred_hr"))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(truth, prediction, alpha=0.75)
    bounds = [min(truth.min(), prediction.min()), max(truth.max(), prediction.max())]
    ax.plot(bounds, bounds, linestyle="--")
    ax.set(xlabel="Reference HR (bpm)", ylabel="Predicted HR (bpm)", title="Session-level correlation")
    fig.tight_layout()
    fig.savefig(destination / "correlation.pdf")
    plt.close(fig)
    mean, diff = (truth + prediction) / 2, prediction - truth
    bias, sd = float(diff.mean()), float(diff.std(ddof=1))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(mean, diff, alpha=0.75)
    for y, label in ((bias, "Bias"), (bias + 1.96 * sd, "+1.96 SD"),
                     (bias - 1.96 * sd, "-1.96 SD")):
        ax.axhline(y, linestyle="--", label=label)
    ax.legend()
    ax.set(xlabel="Mean HR (bpm)", ylabel="Prediction - reference (bpm)", title="Bland-Altman")
    fig.tight_layout()
    fig.savefig(destination / "bland_altman.pdf")
    plt.close(fig)
