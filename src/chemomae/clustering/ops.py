from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

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


def find_elbow_curvature(k_list: List[int], inertia_list: List[float]) -> Tuple[int, int, np.ndarray]:
    r"""
    Detect elbow point by curvature on a normalized curve.

    概要
    ----
    - k-means の `k_list` と対応する `inertia_list` を入力とし、  
      正規化曲線の曲率を計算して「折れ曲がり点 (elbow)」を推定する。
    - 曲率 κ を最大化する点をエルボーとする。

    Parameters
    ----------
    k_list : list of int
        評価したクラスタ数 K のリスト。
    inertia_list : list of float
        各 K における inertia 値（例: `mean(1 - cos)`）。

    Returns
    -------
    optimal_k : int
        曲率が最大となるクラスタ数（推奨値）。
    elbow_idx : int
        `k_list[elbow_idx] == optimal_k` を満たすインデックス。
    kappa : np.ndarray, shape (len(k_list),)
        各点の曲率値（両端は -inf に設定）。

    Notes
    -----
    - 前処理として `y = np.minimum.accumulate(y)` により非増加性を強制。
    - 正規化は x, y を [0, 1] にスケーリング。
    - 曲率 κ は以下で定義される：

        κ = |y''| / (1 + (y')^2)^(3/2)

    - 先頭と末尾の点は κ を -inf にして無視する。
    """
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


def plot_elbow_ckm(k_list, inertias, optimal_k, elbow_idx):
    r"""
    Plot elbow curve and highlight the chosen elbow point.

    概要
    ----
    - `k_list` と対応する `inertias` を折れ線グラフで描画。
    - `find_elbow_curvature` で得た最適クラスタ数 `optimal_k` を縦線とマーカーで強調。

    Parameters
    ----------
    k_list : array-like of int
        評価したクラスタ数のリスト (例: 1..k_max)。
    inertias : array-like of float
        各 k に対する inertia 値（`mean(1 - cos)` など）。
    optimal_k : int
        曲率法などで推定された最適クラスタ数。
    elbow_idx : int
        `k_list[elbow_idx] == optimal_k` を満たすインデックス。

    Notes
    -----
    - Y 軸ラベルは "Mean Cosine Inertia" として描画される。
    - エルボー点にはラベル付き散布図マーカーが追加される。
    - `plt.show()` は呼び出さないため、呼び出し側で表示や保存を行う。

    Examples
    --------
    >>> ks, inertias, K, idx, kappa = elbow_ckmeans(CosineKMeans, X)
    >>> plot_elbow(ks, inertias, K, idx)
    >>> plt.show()
    """
    k_list = np.asarray(k_list)
    inertias = np.asarray(inertias, dtype=float)
    plt.figure(figsize=(6, 4))
    plt.plot(k_list, inertias, "o-", label="Mean Cosine Inertia")
    plt.scatter(k_list[elbow_idx], inertias[elbow_idx], s=120,
                label=f"Elbow: k={optimal_k}, inertia={inertias[elbow_idx]:.5f}")
    plt.axvline(optimal_k, linestyle="--", linewidth=1.5, alpha=0.7)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Mean Cosine Inertia")
    plt.legend(loc="best")
    plt.tight_layout()


def plot_elbow_vmf(k_list, scores, optimal_k, elbow_idx, criterion: str = "bic"):
    r"""
    Plot elbow curve for vMF Mixture and highlight the chosen elbow point.

    概要
    ----
    - `k_list` と対応する `scores`（BIC もしくは平均NLL）を折れ線グラフで描画。
    - `find_elbow_curvature` で得た最適クラスタ数 `optimal_k` を縦線とマーカーで強調。

    Parameters
    ----------
    k_list : array-like of int
        評価したクラスタ数のリスト (例: 1..k_max)。
    scores : array-like of float
        各 k に対する評価値。`criterion="bic"` なら BIC（小さいほど良い）、
        `criterion="nll"` なら平均 NLL（小さいほど良い）。
    optimal_k : int
        曲率法などで推定された最適クラスタ数。
    elbow_idx : int
        `k_list[elbow_idx] == optimal_k` を満たすインデックス。
    criterion : {"bic", "nll"}, default="bic"
        縦軸ラベルなどの表示に使う指標名。

    Notes
    -----
    - BIC は「小さいほど良い」、平均NLL も「小さいほど良い」指標です。
    - `plt.show()` は呼び出さないため、呼び出し側で表示や保存を行ってください。

    Examples
    --------
    >>> ks, scores, K, idx, kappa = elbow_vmf(VMFMixture, X, k_max=30, criterion="bic")
    >>> plot_elbow_vmf(ks, scores, K, idx, criterion="bic")
    >>> plt.show()
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