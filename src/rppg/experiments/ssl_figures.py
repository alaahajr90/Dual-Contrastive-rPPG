"""Generate real Stage-1 t-SNE and cosine-similarity figures (manuscript Figs. 5-7).

The plots are computed from a locally trained model and processed ROI clips,
never from synthetic reconstructions of manuscript values.
"""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..stage1.model import ContrastiveNetwork
from ..preprocessing.dataset import ClipDataset, select_rows
from .ssl_analysis import contrastive_similarity, embed_tsne


def generate_ssl_figures(rows, split, encoder_path, *, destination, sample_limit=256, seed=42):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ContrastiveNetwork().eval()
    state = torch.load(encoder_path, map_location="cpu", weights_only=True)
    model.encoder.load_state_dict(state)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    output = {}
    for label in ("train", "test"):
        partition = select_rows(rows, split[label])
        if len(partition) > sample_limit:
            choice = rng.choice(len(partition), sample_limit, replace=False)
            partition = [partition[i] for i in choice]
        data = ClipDataset(partition, stage=1, augment=True)
        loader = DataLoader(data, batch_size=4, shuffle=False)
        first, second = [], []
        with torch.no_grad():
            for view_a, view_b in loader:
                first.append(model(view_a).numpy())
                second.append(model(view_b).numpy())
        z1, z2 = np.concatenate(first), np.concatenate(second)
        sim = contrastive_similarity(z1, z2)
        points, clusters = embed_tsne(z1, seed=seed, kmeans_clusters=6)
        fig, ax = plt.subplots(figsize=(7, 5))
        if label == "train":
            ax.scatter(points[:, 0], points[:, 1], c=clusters, s=9, cmap="tab10")
        else:
            ax.scatter(points[:, 0], points[:, 1], s=9)
        ax.set(title=f"{label} self-supervised embeddings (t-SNE)", xlabel="t-SNE 1", ylabel="t-SNE 2")
        fig.tight_layout()
        fig.savefig(destination / f"tsne_{label}.pdf")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(sim["positive"], bins=30, density=True, alpha=0.6, label="positive")
        ax.hist(sim["negative"], bins=30, density=True, alpha=0.6, label="negative")
        ax.legend()
        ax.set(xlabel="Cosine similarity", ylabel="Density", title=f"{label} view similarities")
        fig.tight_layout()
        fig.savefig(destination / f"cosine_{label}.pdf")
        plt.close(fig)
        output[label] = {"positive_mean": float(sim["positive"].mean()),
                         "negative_mean": float(sim["negative"].mean()),
                         "n_clips": len(z1), "plot_type": "current_model_not_archived_figures"}
    (destination / "stats.json").write_text(json.dumps(output, indent=2) + "\n")
    return output
