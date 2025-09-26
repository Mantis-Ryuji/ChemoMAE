from __future__ import annotations
from typing import Optional, Union
import numpy as np
import torch

@torch.no_grad()
def fps_downsample(
    X: Union[np.ndarray, torch.Tensor],
    *,
    ratio: float = 0.1,
    ensure_unit_sphere: bool = True,
    seed: Optional[int] = None,
    init_index: Optional[int] = None,
    return_numpy: bool = True,
    eps: float = 1e-12,
):
    """
    Farthest-Point Sampling（FPS, cosine-similarity）

    概要
    ------------
    - 「既選集合と一番“似ていない”サンプル」を順に追加していく多様性重視のダウンサンプリングです（コサイン類似度に基づく）。
    - 返すデータは **元のスケール**（内部の正規化は選択のためだけに使います）。
    - CUDA が利用可能なら自動で GPU 実行します。

    Algorithm
    -----------------------
    1) 初期点を 1 つ選ぶ（乱数 seed か init_index で制御可）。
    2) 目標個数 k に達するまで繰り返す（k は max(1, round(N*ratio))）:
       - 各候補について、選ばれた集合に対するコサイン類似度を計算。
       - その中の最大値（= 既選集合の“最も近い相手”）を記録。
       - この最大値が最小のサンプル（= 集合と一番似ていない）を追加。
       - 以後の更新は 1 回の行列×ベクトル積で効率よく再計算。
    ※ 実装はベクトル化されており、GPU では特に高速です。

    Parameters
    ----------
    X : (N, C) np.ndarray | torch.Tensor
        入力特徴。SNV などの前処理後を想定。
    ratio : float, default 0.1
        抜き出し比率。選択個数 k は max(1, round(N * ratio))。
    ensure_unit_sphere : bool, default True
        True のとき内部で行 L2 正規化を行い、コサイン幾何に整合させます。
        ただし返り値は正規化前の元データから抽出します。
    seed : Optional[int]
        初期点の乱択に使う乱数種（None なら現行 RNG を使用）。
    init_index : Optional[int]
        初期点のインデックスを固定したい場合に指定（seed より優先）。
    return_numpy : bool, default True
        True なら np.ndarray を返す。False なら torch.Tensor を返す。
        Torch 入力で False の場合は入力テンソルと同じデバイスに載せて返します。
    eps : float, default 1e-12
        行 L2 正規化時の数値安定項。

    Returns
    -------
    X_downsampled : (k, C) np.ndarray | torch.Tensor
        選ばれたサブセット。型は return_numpy と入力型の組み合わせに従います。

    Notes
    -----
    - 時間計算量は O(N * k)。各反復は 1 回の行列×ベクトル積と要素ごとの最小更新。
    - CUDA のメモリ使用は PyTorch のキャッシュにより「予約量」が増えて見えることがありますが、
      通常はリークではありません。
    - 行のスケーリング（各行の正の定数倍）には、ensure_unit_sphere=True のとき不変です。

    Example
    -------
    >>> X_snv = SNVScaler().transform(X)        # (N, C)
    >>> X_sub = fps_downsample(X_snv, ratio=0.1, ensure_unit_sphere=True, seed=42)
    """
    # ---- 前処理（Torchへ集約、CUDAがあれば使用） -----------------------
    is_numpy = isinstance(X, np.ndarray)
    xt = torch.from_numpy(X) if is_numpy else X
    if xt.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={tuple(xt.shape)}")
    N, C = xt.shape
    if N == 0:
        # 空入力のときは空の同型配列/テンソルを返す
        if is_numpy and return_numpy:
            return X[:0]
        elif is_numpy and not return_numpy:
            return torch.from_numpy(X[:0])
        elif not is_numpy and return_numpy:
            return xt[:0].detach().cpu().numpy()
        else:
            return xt[:0]

    # 最適デバイス決定と型
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = xt.dtype if xt.dtype.is_floating_point else torch.float32
    xt = xt.to(device=dev, dtype=dtype, non_blocking=True)

    # 計算用の単位球表現（必要なら）
    if ensure_unit_sphere:
        n = torch.linalg.vector_norm(xt, dim=1, keepdim=True)
        X_unit = xt / (n + eps)
    else:
        X_unit = xt

    # 取得個数
    k = max(1, int(round(N * float(ratio))))

    # RNG（初期点用）
    gen = torch.Generator(device=dev)
    if seed is not None:
        gen.manual_seed(int(seed))
    else:
        # 非決定的でよければ現在のグローバルシードから派生
        gen.manual_seed(torch.seed())

    # 初期点
    if init_index is None:
        idx0 = int(torch.randint(low=0, high=N, size=(1,), generator=gen, device=dev).item())
    else:
        if not (0 <= init_index < N):
            raise ValueError(f"init_index out of range: {init_index} not in [0,{N})")
        idx0 = int(init_index)

    # ---- FPS 本体（cos距離でO(Nk)、テンポラリは再利用） ---------------
    idx = torch.empty(k, dtype=torch.long, device="cpu")
    idx[0] = idx0

    # 既選集合への最近距離 dmin = 1 - X_unit @ x_sel
    x0 = X_unit[idx0]                 # (C,)
    dmin = 1.0 - (X_unit @ x0)        # (N,)
    dmin.clamp_min_(0.0)
    dmin[idx0] = 0.0

    # 一時バッファ（再利用して予約メモリの増加を抑える）
    sim = torch.empty(N, device=dev)          # sim = X_unit @ x_new
    one_minus = torch.empty(N, device=dev)    # 1 - sim

    for t in range(1, k):
        next_i = int(torch.argmax(dmin).item())
        idx[t] = next_i

        x_new = X_unit[next_i]               # (C,)
        sim.zero_().addmv_(X_unit, x_new)    # sim = X_unit @ x_new
        one_minus.copy_(sim).mul_(-1.0).add_(1.0)  # 1 - sim
        torch.minimum(dmin, one_minus, out=dmin)   # dmin = min(dmin, 1 - sim)
        dmin[next_i] = 0.0

    # ---- 出力（元スケールの値で返す） -----------------------------------
    if return_numpy:
        if is_numpy:
            # numpy入力 → numpyのまま返す（元dtype保持）
            return X[idx.numpy()]
        else:
            # torch入力でもnumpyで返す
            return xt.index_select(0, idx.to(device=dev)).detach().cpu().numpy()
    else:
        if is_numpy:
            # numpy入力だがtorchで返す（計算デバイス上）
            return xt.index_select(0, idx.to(device=dev))
        else:
            # torch入力は入力テンソルと同じデバイスで返す
            x_dev = X.device
            return X.index_select(0, idx.to(device=x_dev))
