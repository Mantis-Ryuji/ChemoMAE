from __future__ import annotations
import numpy as np
import torch
from typing import Optional


@torch.no_grad()
def silhouette_samples_cosine_gpu(
    X: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    *,
    device: str = "cuda",
    chunk: Optional[int] = 1000000,
    dtype: torch.dtype = torch.float32,
    return_numpy: bool = True,
    eps: float = 1e-12,
) -> np.ndarray | torch.Tensor:
    r"""
    sklearn.metrics.silhouette_samples と同等の出力を返す
    「コサイン距離」版の GPU 実装（厳密・O(NK)）。

    Parameters
    ----------
    X : (N, D) array-like
        入力特徴。np.ndarray / torch.Tensor どちらでも可。
        ※ sklearn の cosine と同様、内部で行方向に L2 正規化します。
          （零ベクトルはそのまま 0 ベクトルのまま扱い、内積=0 → 距離=1）
    labels : (N,) array-like of int
        クラスタ割当。非連番でも可（内部で 0..K-1 に圧縮）。

    device : {"cuda","cpu",...}
        計算デバイス。
    chunk : int | None
        b_i 計算時のタイルサイズ（X @ M^T を分割）。VRAM に応じて調整。
        None なら一括。
    dtype : torch.dtype
        計算 dtype（float16 / bfloat16 / float32）。
    return_numpy : bool
        True なら np.ndarray を返す。False なら torch.Tensor を返す。
    eps : float
        数値安定用の微小量。

    Returns
    -------
    s : (N,) same type as `return_numpy`
        各サンプルのシルエット値。範囲は [-1, 1]。
        単独クラスタのサンプルは 0 に設定。

    Notes
    -----
    - 定義：
        - d(x,y) = 1 - cos(x,y)
        - a_i = 自クラスタ内の「除外平均距離」
        - b_i = 他クラスタ平均距離の最小
        - s_i = (b_i - a_i) / max(a_i, b_i)
      ここで cos は行ベクトル L2 正規化後の内積で算出。

    - 計算量：O(ND + KD + NK_chunk)。主に X @ M^T が支配的。
    - メモリ：X, M と一時タイル（chunk）に依存。chunk を小さくすれば省メモリ。
    """
    # ---- 入力を torch に変換 ----
    x_is_numpy = isinstance(X, np.ndarray)
    l_is_numpy = isinstance(labels, np.ndarray)

    if x_is_numpy:
        X_t = torch.as_tensor(X, device=device, dtype=dtype)
    else:
        X_t = X.to(device=device, dtype=dtype, non_blocking=True)

    if l_is_numpy:
        y_t = torch.as_tensor(labels, device=device, dtype=torch.long)
    else:
        y_t = labels.to(device=device, dtype=torch.long, non_blocking=True)

    assert X_t.ndim == 2 and y_t.ndim == 1 and X_t.size(0) == y_t.size(0), "Shape mismatch"
    N, D = X_t.shape

    # ---- ラベルを 0..K-1 に圧縮（非連番対応）----
    uniq, inv = torch.unique(y_t, sorted=True, return_inverse=True)
    y_t = inv  # 0..K-1
    K = int(uniq.numel())

    # ---- 行方向 L2 正規化（零ベクトルはそのまま0）----
    # norm=0 の行はそのまま 0 ベクトル → 任意のベクトルとの cos=0 → 距離=1
    norm = X_t.norm(p=2, dim=1, keepdim=True)
    safe = norm.clamp_min(eps)
    Xn = X_t / safe
    Xn = torch.where(norm > 0, Xn, torch.zeros_like(Xn))

    # ---- クラスタごとの和ベクトル S_k とサイズ n_k ----
    S = torch.zeros(K, D, device=device, dtype=Xn.dtype)
    S.index_add_(0, y_t, Xn)  # scatter-add
    n = torch.bincount(y_t, minlength=K)
    n = torch.clamp(n, min=1)  # ゼロ除算回避
    M = S / n.unsqueeze(1)     # 各クラスタの平均ベクトル（単位化は不要）

    # ---- a_i：除外平均（自クラスタ平均だが自分を除く）----
    S_c = S[y_t]            # (N, D)
    n_c = n[y_t]            # (N,)
    denom_a = torch.clamp(n_c - 1, min=1)      # n_c==1 は後で上書き
    mean_excl = (S_c - Xn) / denom_a.unsqueeze(1)
    cos_in = torch.sum(Xn * mean_excl, dim=1)  # x_i^T mean_excl
    a = 1.0 - cos_in
    # 単独クラスタは a=0（最終的に s=0 に設定）
    a = torch.where(n_c == 1, torch.zeros_like(a), a)

    # ---- b_i：他クラスタ平均距離の最小（= 1 - 最大平均類似度）----
    if chunk is None:
        sims = Xn @ M.t()  # (N, K)
        sims[torch.arange(N, device=device), y_t] = float("-inf")
        best_sim, _ = sims.max(dim=1)
        b = 1.0 - best_sim
    else:
        b = torch.empty(N, device=device, dtype=Xn.dtype)
        start = 0
        while start < N:
            end = min(start + int(chunk), N)
            sims_blk = Xn[start:end] @ M.t()  # (B, K)
            sims_blk[torch.arange(end - start, device=device), y_t[start:end]] = float("-inf")
            best_sim_blk, _ = sims_blk.max(dim=1)
            b[start:end] = 1.0 - best_sim_blk
            start = end

    # ---- silhouette s_i ----
    denom = torch.maximum(a, b).clamp_min(1e-12)
    s = (b - a) / denom
    # 単独クラスタは 0
    s = torch.where(n_c == 1, torch.zeros_like(s), s)

    if return_numpy:
        return s.detach().cpu().numpy()
    return s


@torch.no_grad()
def silhouette_score_cosine_gpu(
    X: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    *,
    return_numpy: bool = True,
    **kwargs,
) -> float:
    """
    便宜関数：サンプルごとの値の平均（= silhouette_score 相当）を返す。
    kwargs は silhouette_samples_cosine_gpu にそのまま渡されます。
    """
    # 強制的にテンソル計算させる
    kw = dict(kwargs)
    kw["return_numpy"] = False
    s = silhouette_samples_cosine_gpu(X, labels, **kw)  # torch.Tensor
    
    # 平均は fp32 で計算
    score_t = s.float().mean(dtype=torch.float32)

    if return_numpy:
        return float(score_t.item())        # Python float (NumPy 互換)
    else:
        return score_t                      # torch.Tensor (float32 scalar)
