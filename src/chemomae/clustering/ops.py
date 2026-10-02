from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

__all__ = [
    "find_elbow_curvature",
    "plot_elbow_ckm",
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


def find_elbow_curvature(
    k_list: List[int],
    inertia_list: List[float],
    smooth: bool = True,
    window_length: int = 5,
    polyorder: int = 2,
) -> Tuple[int, int, float]:
    """
    Detect elbow point by curvature on a normalized curve.
    Compute kappa directly from Savitzky–Golay derivatives (deriv=1,2),
    returning kappa at its maximum.
    """
    x = np.asarray(k_list, dtype=float)
    y = np.asarray(inertia_list, dtype=float)
    n = len(x)
    if n < 3:
        raise ValueError("k_list must have length >= 3")

    # 1) Enforce a nonincreasing sequence.
    y = np.minimum.accumulate(y)

    # 2) Normalize.
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)

    # 3) Adjust S-G parameters for small samples.
    #    - The odd window length is bounded according to n.
    #    - Ensure polyorder < window_length.
    if smooth and n >= 5:
        wl = min(window_length, (n // 2) * 2 + 1)  # Largest odd length near n.
        wl = max(5, wl | 1)                        # At least five and odd.
        po = min(polyorder, wl - 1)
        po = max(2, po)                            # At least quadratic for curvature.
        # 4) Sampling interval on normalized x.
        dx = float(np.median(np.diff(x_n)))
        if not np.isfinite(dx) or dx <= 0:
            dx = 1.0

        # 5) Compute y', y'' directly with S-G and interpolated endpoints.
        #    Retain a zeroth-order pass for stability even without smoothed output.
        _ = savgol_filter(y_n, window_length=wl, polyorder=po, deriv=0, mode="interp")
        dy  = savgol_filter(y_n, window_length=wl, polyorder=po, deriv=1, delta=dx, mode="interp")
        d2y = savgol_filter(y_n, window_length=wl, polyorder=po, deriv=2, delta=dx, mode="interp")
    else:
        # Fallback when S-G is not used.
        dy  = np.gradient(y_n, x_n)
        d2y = np.gradient(dy,  x_n)

    # 6) Curvature kappa = |y''| / (1 + (y')^2)^(3/2).
    kappa = np.abs(d2y) / np.power(1.0 + dy * dy, 1.5)

    # 7) Ignore endpoints.
    kappa[0] = kappa[-1] = -np.inf

    idx = int(np.argmax(kappa))
    return int(k_list[idx]), idx, float(kappa[idx])


def plot_elbow_ckm(k_list, inertias, optimal_k, elbow_idx):
    r"""
    Plot elbow curve and highlight the chosen elbow point.

    Overview
    ----
    - Plot `k_list` and the corresponding `inertias` as a line graph.
    - Highlight `optimal_k` from `find_elbow_curvature` with a vertical line and marker.

    Parameters
    ----------
    k_list : array-like of int
        Evaluated cluster counts (for example, 1..k_max).
    inertias : array-like of float
        Inertia for each k, such as `mean(1 - cos)`.
    optimal_k : int
        Optimal cluster count estimated by curvature or another method.
    elbow_idx : int
        Index satisfying `k_list[elbow_idx] == optimal_k`.

    Notes
    -----
    - The Y-axis label is "Mean Cosine Inertia".
    - A labeled scatter marker identifies the elbow point.
    - `plt.show()` is not called; the caller handles display or saving.
    """
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


def plot_elbow_vmf(k_list, scores, optimal_k, elbow_idx, criterion: str = "bic"):
    r"""
    Plot elbow curve for vMF Mixture and highlight the chosen elbow point.

    Overview
    ----
    - Plot `k_list` and corresponding `scores` (BIC or mean NLL) as a line graph.
    - Highlight `optimal_k` from `find_elbow_curvature` with a vertical line and marker.

    Parameters
    ----------
    k_list : array-like of int
        Evaluated cluster counts (for example, 1..k_max).
    scores : array-like of float
        Score for each k: BIC for `criterion="bic"`, or mean NLL for
        `criterion="nll"`. Lower is better for both.
    optimal_k : int
        Optimal cluster count estimated by curvature or another method.
    elbow_idx : int
        Index satisfying `k_list[elbow_idx] == optimal_k`.
    criterion : {"bic", "nll"}, default="bic"
        Criterion used for display labels, including the Y axis.

    Notes
    -----
    - Lower values are better for both BIC and mean NLL.
    - `plt.show()` is not called; the caller handles display or saving.
    """
    k_list = np.asarray(k_list)
    scores = np.asarray(scores, dtype=float)

    crit = (criterion or "bic").lower()
    if crit == "bic":
        ylabel = "BIC (lower is better)"
        line_label = "BIC"
    elif crit in ("nll", "negloglik", "neg_log_likelihood"):
        ylabel = "Mean NLL (lower is better)"
        line_label = "Mean NLL"
    else:
        ylabel = "Score"
        line_label = "Score"

    plt.figure(figsize=(6, 4))
    plt.plot(k_list, scores, "o-", label=line_label)
    plt.scatter(k_list[elbow_idx], scores[elbow_idx], s=120,
                label=f"Elbow: k={optimal_k}, score={scores[elbow_idx]:.4f}")
    plt.axvline(optimal_k, linestyle="--", linewidth=1.5, alpha=0.7)
    plt.xlabel("Number of Components (k)")
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.tight_layout()
