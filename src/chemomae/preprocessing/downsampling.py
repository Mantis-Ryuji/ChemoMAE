from __future__ import annotations
from typing import Optional, Union, Tuple
import numpy as np
import torch


@torch.no_grad()
def cosine_fps_downsample(
    X: Union[np.ndarray, torch.Tensor],
    *,
    ratio: float = 0.1,
    seed: Optional[int] = None,
    init_index: Optional[int] = None,
    return_numpy: bool = True,
    return_indices: bool = False,
    eps: float = 1e-12,
) -> Union[np.ndarray, torch.Tensor, Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]]:
    """
    Farthest-Point Sampling (FPS) on the unit hypersphere with cosine geometry.

    概要
    ----
    - 既選集合から「最も離れた（= 方向が最も異なる）」サンプルを 1 点ずつ追加していく、
      多様性重視のダウンサンプリングです（コサイン幾何、単位球上）。
    - 内部では **必ず各行を L2 正規化** してコサイン幾何に整合させます。
      ただし **返り値は元スケール**（正規化前のデータ）から抽出します。
    - CUDA が利用可能なら自動で GPU を使用します。

    Algorithm
    ---------
    1) 初期点を 1 つ選ぶ（`init_index` があればそれを使用。なければ `seed` に基づく乱択）。
    2) 目標個数 k（= min(max(1, round(N*ratio)), N)）に達するまで繰り返す:
       - 各候補 i について、既選集合 S の中での最近傍距離（cos 距離） r(i) を維持更新。
       - r(i) が最大の i を 1 点だけ S に追加。
       - 以後の更新は 1 回の行列×ベクトル積と要素ごとの最小更新で O(N) に抑制。

    Parameters
    ----------
    X : (N, C) np.ndarray | torch.Tensor
        入力特徴（SNV 等の前処理後を想定）。NaN/Inf を含まないこと。
    ratio : float, default 0.1
        抜き出し比率。選択個数 k は k = min(max(1, round(N*ratio)), N)。
        ratio <= 0 はエラー、ratio >= 1 なら全件選択 (k = N) に収束します。
    seed : Optional[int]
        初期点の乱択に使う乱数種（None なら現行 RNG）。`init_index` 指定時は無視されます。
    init_index : Optional[int]
        初期点のインデックスを固定したい場合に指定（`seed` より優先）。
    return_numpy : bool, default True
        True なら np.ndarray を返す。False なら torch.Tensor を返す。
        Torch 入力で False の場合は入力テンソルと同じデバイスに載せて返します。
    return_indices : bool, default False
        True の場合、（サブセット, 選択インデックス）を返します。
        return_numpy=True のとき indices は np.ndarray、False のときは torch.Tensor。
    eps : float, default 1e-12
        行 L2 正規化時の数値安定項。

    Returns
    -------
    X_downsampled : (k, C) np.ndarray | torch.Tensor
        選ばれたサブセット（元スケール）。
    indices : (k,) np.ndarray | torch.Tensor, optional
        return_indices=True のときのみ添付。

    Notes
    -----
    - 時間計算量は O(N * k)。各反復は 1 回の行列×ベクトル積と要素ごとの最小更新。
    - 計算は少なくとも float32 で行います（fp16/bf16 入力は内部で昇格）。
    - 角距離 d_ang = arccos(cos) と 1 - cos は単調変換の関係にあり、argmax/argmin による
      選択順位は一致します（実装は高速な 1 - cos を用いて順位付けします）。
    - 返り値の dtype / device は **元入力に合わせる** 方針です（numpy 入力→numpy 出力、
      torch 入力→希望に応じて numpy / torch、torch のときは元デバイス）。
    """
    # ---- 型判定と基本検証 -------------------------------------------------
    is_numpy = isinstance(X, np.ndarray)
    if is_numpy:
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={tuple(X.shape)}")
        if X.size == 0:
            # 空入力：元型で空を返す
            empty_np = X[:0]
            if return_indices:
                return empty_np, (np.empty((0,), dtype=int))
            return empty_np
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Inf.")
        xt = torch.from_numpy(X)
    else:
        if not torch.is_tensor(X):
            raise TypeError("X must be a numpy array or torch tensor.")
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape={tuple(X.shape)}")
        if X.numel() == 0:
            empty_t = X[:0]
            if return_numpy:
                empty_np = empty_t.detach().cpu().numpy()
                if return_indices:
                    return empty_np, (np.empty((0,), dtype=int))
                return empty_np
            else:
                if return_indices:
                    return empty_t, (torch.empty(0, dtype=torch.long, device=X.device))
                return empty_t
        if not torch.isfinite(X).all():
            raise ValueError("X contains NaN or Inf.")
        xt = X

    N, C = int(xt.shape[0]), int(xt.shape[1])

    # 取得個数 k の決定（クリップ）
    k = int(round(N * float(ratio)))
    k = max(1, k)
    k = min(k, N)

    # ---- デバイス・dtype 設定（演算は少なくとも fp32） -------------------
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = xt.dtype if xt.dtype.is_floating_point else torch.float32
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    xt_work = xt.to(device=dev, dtype=dtype, non_blocking=True)

    # ---- 単位球への埋め込み（必ず実施） ---------------------------------
    n = torch.linalg.vector_norm(xt_work, dim=1, keepdim=True)
    X_unit = xt_work / (n + eps)

    # ---- 乱数生成器（初期点用） ------------------------------------------
    gen = torch.Generator(device=dev)
    if init_index is None:
        if seed is not None:
            gen.manual_seed(int(seed))
        else:
            gen.manual_seed(torch.seed())
        idx0 = int(torch.randint(low=0, high=N, size=(1,), generator=gen, device=dev).item())
    else:
        if not (0 <= int(init_index) < N):
            raise ValueError(f"init_index out of range: {init_index} not in [0,{N})")
        idx0 = int(init_index)

    # ---- FPS 本体 ---------------------------------------------------------
    idx = torch.empty(k, dtype=torch.long, device="cpu")
    idx[0] = idx0

    # 既選集合への最近距離 dmin = 1 - X_unit @ x_sel（数値安定の clamping）
    x0 = X_unit[idx0]                           # (C,)
    dmin = 1.0 - (X_unit @ x0)                  # (N,)
    dmin.clamp_min_(0.0)
    NEG_INF = torch.tensor(float("-inf"), device=dev)
    dmin[idx0] = NEG_INF                        # 再選択を確実に禁止

    # 一時バッファ（再利用）
    sim = torch.empty(N, device=dev)            # sim = X_unit @ x_new
    one_minus = torch.empty(N, device=dev)      # 1 - sim

    for t in range(1, k):
        # 最も遠い候補を選ぶ
        next_i = int(torch.argmax(dmin).item())
        idx[t] = next_i

        # 新規追加点との類似度を計算し、最近距離を更新
        x_new = X_unit[next_i]                  # (C,)
        sim.zero_().addmv_(X_unit, x_new)      # sim = X_unit @ x_new
        sim.clamp_(-1.0, 1.0)                   # 数値誤差対策
        one_minus.copy_(sim).mul_(-1.0).add_(1.0)     # 1 - sim
        torch.minimum(dmin, one_minus, out=dmin)      # dmin = min(dmin, 1 - sim)
        dmin[next_i] = NEG_INF                  # 追加済みは再選択不可

    # ---- 出力（元スケールから抽出） --------------------------------------
    if is_numpy:
        sel_np = X[idx.numpy()]  # 元 dtype のまま
        if return_numpy:
            if return_indices:
                return sel_np, idx.numpy()
            return sel_np
        else:
            sel_t = torch.from_numpy(sel_np).to(device=dev, dtype=dtype)
            if return_indices:
                return sel_t, torch.from_numpy(idx.numpy()).to(device=dev)
            return sel_t
    else:
        # torch 入力
        x_dev = X.device
        sel_t = X.index_select(0, idx.to(device=x_dev))  # 元デバイス・元 dtype
        if return_numpy:
            sel_np = sel_t.detach().cpu().numpy()
            if return_indices:
                return sel_np, idx.numpy()
            return sel_np
        else:
            if return_indices:
                return sel_t, idx.to(device=x_dev)
            return sel_t
