"""
Visualization helpers: UMAP, confusion matrix plotting, etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    normalize: bool = True,
    figsize=(8, 6),
    out_path: Optional[str | Path] = None,
):
    """
    Simple confusion matrix plot.
    """
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1e-12)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, aspect="auto")

    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)

    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")

    return fig, ax


def embed_2d(
    X: np.ndarray,
    method: str = "umap",
    random_state: int = 42,
    n_components: int = 2,
):
    """
    Project embeddings to 2D for visualization.
    """
    if method == "umap":
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
        )
    elif method == "tsne":
        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            init="pca",
        )
    elif method == "pca":
        reducer = PCA(
            n_components=n_components,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return reducer.fit_transform(X)
