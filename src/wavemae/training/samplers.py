from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

# ---- reference & cosine utilities ----

def compute_reference_vector(X: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    X = np.asarray(X, np.float32)
    u = X.mean(axis=0)
    u /= (np.linalg.norm(u) + 1e-12)
    return u.astype(np.float32)


def cosine_to_reference(X: torch.Tensor | np.ndarray, ref: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(ref, torch.Tensor):
        ref = ref.detach().cpu().numpy()
    X = np.asarray(X, np.float32)
    ref = np.asarray(ref, np.float32)
    Xu = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    ref = ref / (np.linalg.norm(ref) + 1e-12)
    sims = np.clip(Xu @ ref, -1.0, 1.0).astype(np.float32)
    return sims


def make_weighted_sampler_by_cosine(
    train_ds: Dataset,
    ref_vec: np.ndarray | torch.Tensor,
    *,
    cos_mid: float = 0.50,
    cos_beta: float = 8.0,
    clip: Tuple[float, float] = (0.3, 3.0),
    replacement: bool = True,
) -> WeightedRandomSampler:
    """w = sigmoid(beta * (cos_mid - cos)) の連続重み。"""
    if hasattr(train_ds, "data") and isinstance(train_ds.data, torch.Tensor):
        X = train_ds.data
    else:
        X = torch.stack([train_ds[i] for i in range(len(train_ds))], dim=0)

    sims = cosine_to_reference(X, ref_vec).astype(np.float64)

    beta = float(cos_beta); mid = float(cos_mid)
    w = 1.0 / (1.0 + np.exp(-beta * (mid - sims)))   # monotonic sigmoid
    w = w.astype(np.float32)

    w /= (w.mean() + 1e-12)
    w = np.clip(w, clip[0], clip[1])

    return WeightedRandomSampler(weights=torch.from_numpy(w), num_samples=sims.shape[0], replacement=replacement)
