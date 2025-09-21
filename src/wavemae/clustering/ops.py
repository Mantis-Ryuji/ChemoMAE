from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

__all__ = [
    "l2_normalize_rows",
    "cosine_similarity",
    "cosine_dissimilarity",
    "find_elbow_curvature",
    "plot_elbow",
]


def l2_normalize_rows(X: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Row-wise L2 normalization."""
    return F.normalize(X, dim=1, eps=eps)


def cosine_similarity(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Cosine similarity for row-normalized A,B (no check)."""
    return A @ B.T


def cosine_dissimilarity(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """1 - cosine similarity for row-normalized A,B (no check)."""
    return 1.0 - (A @ B.T)


def find_elbow_curvature(k_list: List[int], inertia_list: List[float]) -> Tuple[int, int, np.ndarray]:
    """Elbow by curvature on normalized curve. Returns (optimal_k, elbow_idx, kappa)."""
    x = np.asarray(k_list, dtype=float)
    y = np.asarray(inertia_list, dtype=float)
    if len(x) < 3:
        raise ValueError("k_list must have length >= 3")
    # enforce monotone non-increasing
    y = np.minimum.accumulate(y)
    # normalize axes
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)
    dy = np.gradient(y_n, x_n)
    d2y = np.gradient(dy, x_n)
    kappa = np.abs(d2y) / np.power(1.0 + dy * dy, 1.5)
    kappa[0] = -np.inf
    kappa[-1] = -np.inf
    idx = int(np.argmax(kappa))
    return int(k_list[idx]), idx, kappa


def plot_elbow(k_list, inertias, optimal_k, elbow_idx):
    """Quick elbow plot helper."""
    k_list = np.asarray(k_list)
    inertias = np.asarray(inertias, dtype=float)
    plt.figure(figsize=(6, 4))
    plt.plot(k_list, inertias, "o-", label="Mean Cosine Inertia")
    plt.scatter(k_list[elbow_idx], inertias[elbow_idx], s=120,
                label=f"Elbow: k={optimal_k}, inertia={inertias[elbow_idx]:.4f}")
    plt.axvline(optimal_k, linestyle="--", linewidth=1.5, alpha=0.7)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Mean Cosine Inertia")
    plt.legend(loc="best")
    plt.tight_layout()
