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
) -> Tuple[int, int, np.ndarray]:
    """
    Detect elbow point by curvature on a normalized curve.
    Savitzky–Golay の微分出力（deriv=1,2）を直接用いて κ を計算する。
    """
    x = np.asarray(k_list, dtype=float)
    y = np.asarray(inertia_list, dtype=float)
    n = len(x)
    if n < 3:
        raise ValueError("k_list must have length >= 3")

    # 1) 非増加性の強制
    y = np.minimum.accumulate(y)

    # 2) 正規化
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)

    # 3) S-G 用のパラメータ調整（小標本セーフティ）
    #    - 窓長は最大でも n に依存、かつ奇数
    #    - polyorder < window_length を保証
    if smooth and n >= 5:
        wl = min(window_length, (n // 2) * 2 + 1)  # 最大の奇数（≲ n）
        wl = max(5, wl | 1)                        # 下限5、奇数化
        po = min(polyorder, wl - 1)
        po = max(2, po)                            # 下限2（曲率に十分）
        # 4) サンプリング間隔（正規化 x 上）
        dx = float(np.median(np.diff(x_n)))
        if not np.isfinite(dx) or dx <= 0:
            dx = 1.0

        # 5) S-G で解析的に y', y'' を直接計算（端点は補間モード）
        #    y_smooth は出力用途がなければ省略可だが、安定のために 0次も一度通す
        _ = savgol_filter(y_n, window_length=wl, polyorder=po, deriv=0, mode="interp")
        dy  = savgol_filter(y_n, window_length=wl, polyorder=po, deriv=1, delta=dx, mode="interp")
        d2y = savgol_filter(y_n, window_length=wl, polyorder=po, deriv=2, delta=dx, mode="interp")
    else:
        # フォールバック（S-Gを使わない場合）
        dy  = np.gradient(y_n, x_n)
        d2y = np.gradient(dy,  x_n)

    # 6) 曲率 κ = |y''| / (1 + (y')^2)^(3/2)
    kappa = np.abs(d2y) / np.power(1.0 + dy * dy, 1.5)

    # 7) 端点は無視
    kappa[0] = kappa[-1] = -np.inf

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
                label=f"Elbow: k={optimal_k}, inertia={inertias[elbow_idx]:.4f}")
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