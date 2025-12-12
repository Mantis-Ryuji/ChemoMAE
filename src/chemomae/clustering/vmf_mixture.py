from __future__ import annotations

import gc
import math
from typing import Optional, Tuple, List, Union, Dict, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["VMFMixture", "elbow_vmf", "vmf_logC", "vmf_bessel_ratio"]


def _logIv_small(nu: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Small-κ expansion of log I_ν(κ).
    I_ν(κ) = (κ/2)^ν / Γ(ν+1) * [1 + κ^2/(4(ν+1)) + κ^4/(32(ν+1)(ν+2)) + ...]
    We keep up to κ^4 term and compute log safely.
    """
    eps = 1e-12
    t = (k * 0.5).clamp_min(eps)
    base = nu * torch.log(t) - torch.lgamma(nu + 1.0)
    a1 = (k * k) / (4.0 * (nu + 1.0))
    a2 = (k**4) / (32.0 * (nu + 1.0) * (nu + 2.0))
    series = 1.0 + a1 + a2
    return base + torch.log(series.clamp_min(eps))


def _logIv_large(nu: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Large-κ asymptotic for log I_ν(κ).
    I_ν(κ) ~ e^κ / sqrt(2πκ) * (1 - (μ-1)/(8κ) + (μ-1)(μ-9)/(2!(8κ)^2) - ...)
    with μ = 4ν^2.
    We keep two correction terms inside log for stability.
    """
    eps = 1e-12
    mu = 4.0 * (nu * nu)
    invk = 1.0 / k.clamp_min(1e-6)
    c1 = -(mu - 1.0) * 0.125 * invk
    c2 = (mu - 1.0) * (mu - 9.0) * (invk * invk) / (2.0 * (8.0**2))
    corr = 1.0 + c1 + c2
    return k - 0.5 * (torch.log(2.0 * math.pi * k.clamp_min(1e-6))) + torch.log(
        corr.clamp_min(eps)
    )


def _blend(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # smoothstep-like blend: w in [0,1]
    w = w.clamp(0.0, 1.0)
    return (1.0 - w) * a + w * b


def _logIv_piecewise(nu: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    # thresholds: small < K1, large > K2, blend in between
    K1 = 2.0
    K2 = 12.0
    small = _logIv_small(nu, k)
    large = _logIv_large(nu, k)
    w = ((k - K1) / (K2 - K1))
    return torch.where(k <= K1, small, torch.where(k >= K2, large, _blend(small, large, w)))


def vmf_bessel_ratio(nu: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Approximate R_ν(κ) = I_{ν+1}(κ)/I_ν(κ).
    - small κ: R ~ κ/(2ν+2) * [1 - κ^2/(2(2ν+2)(2ν+4))]
    - large κ: R ~ 1 - (2ν+1)/(2κ) + (4ν^2 - 1)/(8κ^2)
    - mid κ: smooth blend.
    """
    k_cl = k.clamp_min(1e-6)
    two_nu = 2.0 * nu

    a = two_nu + 2.0
    b = two_nu + 4.0
    R_small = (k_cl / a) * (1.0 - (k_cl * k_cl) / (2.0 * a * b))

    R_large = 1.0 - (two_nu + 1.0) / (2.0 * k_cl) + (4.0 * nu * nu - 1.0) / (
        8.0 * (k_cl * k_cl)
    )

    K1 = 2.0
    K2 = 12.0
    w = ((k - K1) / (K2 - K1)).clamp(0.0, 1.0)
    R = torch.where(k <= K1, R_small, torch.where(k >= K2, R_large, _blend(R_small, R_large, w)))
    return R.clamp(1e-6, 1.0 - 1e-6)


def vmf_logC(d: int, kappa: torch.Tensor) -> torch.Tensor:
    """Compute log C_d(kappa) with approximated log I_ν.
    C_d(κ) = κ^ν / [(2π)^{ν+1} I_ν(κ)],   ν = d/2 - 1
    Returns tensor with the same shape as kappa.
    """
    nu = 0.5 * float(d) - 1.0
    nu_t = torch.as_tensor(nu, dtype=kappa.dtype, device=kappa.device)
    logIv = _logIv_piecewise(nu_t, kappa)
    return nu_t * torch.log(kappa.clamp_min(1e-12)) - (nu_t + 1.0) * math.log(2.0 * math.pi) - logIv


class VMFMixture(nn.Module):
    r"""
    von Mises–Fisher (vMF) Mixture Model on the unit hypersphere with chunked EM.

    概要
    ----
    本クラスは **単位球面上の混合モデル** vMF mixture を EM で学習する。
    入力特徴 `X` は行方向 L2 正規化されたベクトル（`||x_i||=1`）として扱い、
    各成分 k は方向 `μ_k`（単位ベクトル）と集中度 `κ_k` を持つ。

    分布
    ----
    - vMF 密度（d 次元）:
      `p(x | μ, κ) = C_d(κ) * exp(κ μ^T x)`,  `||x||=||μ||=1`
    - `C_d(κ)` は正規化定数で、ベッセル関数 `I_ν` を含む:
      `C_d(κ) = κ^ν / ((2π)^(ν+1) I_ν(κ))`,  `ν = d/2 - 1`

    最適化（EM）
    -----------
    目的は対数尤度
    `L = Σ_i log Σ_k π_k * C_d(κ_k) * exp(κ_k μ_k^T x_i)`
    を最大化すること。

    - E-step:
      責務（posterior）
      `γ_{ik} = p(z_i=k | x_i) = softmax_k( log π_k + log C_d(κ_k) + κ_k μ_k^T x_i )`
    - M-step:
      - 混合比: `π_k = (1/N) Σ_i γ_{ik}`
      - 方向:   `s_k = Σ_i γ_{ik} x_i`,  `μ_k = s_k / ||s_k||`
      - 集中度: `R̄_k = ||s_k|| / Σ_i γ_{ik}` から近似更新（本実装は closed-form 近似）

    近似（torch-only）
    -----------------
    vMF の正規化定数や κ 更新に必要なベッセル関数比を、
    小 κ / 大 κ の級数・漸近展開を **滑らかに接続**する近似で実装する。
    - `log I_ν(κ)` の近似（piecewise + blend）
    - `R_ν(κ)=I_{ν+1}(κ)/I_ν(κ)` の近似（piecewise + blend）

    ストリーミング（chunked E-step）
    -------------------------------
    `chunk` を指定すると、E-step を `X` のブロックに分割し、
    `X[s:e]` を逐次 `device` に移して責務と十分統計量 `(N_k, S_k)` を累積する。
    VRAM が厳しい大規模データでの学習を想定。

    Parameters
    ----------
    n_components : int
        混合成分数 K。
    d : int | None, default=None
        特徴次元 D。None の場合は `fit(X)` 時に `X.shape[1]` から決定。
    device : str | torch.device, default="cuda"
        計算デバイス。chunk を使う場合でも十分統計の集約はこの device 上で行う。
    random_state : int | None, default=42
        初期化やサンプリングに用いる乱数シード（CPU Generator 固定）。
    tol : float, default=1e-4
        収束判定閾値（相対改善 `|lb_t-lb_{t-1}|/(|lb_{t-1}|+eps)` が tol 未満で停止）。
    max_iter : int, default=200
        EM 反復回数の上限。
    init : {"kmeans++", "random"}, default="kmeans++"
        初期化方式。`kmeans++` は cosine 距離 `1-cos` に相当するシード選択を行う。
    kappa_init : float, default=10.0
        κ の初期値（全成分共通）。
    kappa_min : float, default=1e-6
        κ の下限（数値安定のため）。
    dtype : torch.dtype, default=torch.float32
        内部計算 dtype。入力が bf16/fp16 でもこの dtype に変換して計算する想定。

    Attributes
    ----------
    K : int
        混合成分数。
    d : int | None
        特徴次元。`fit()` 後は必ず int。
    mus : torch.Tensor, shape (K, d)
        各成分の方向ベクトル（L2 正規化済み）。
    kappas : torch.Tensor, shape (K,)
        各成分の集中度 κ（`>=kappa_min`）。
    logpi : torch.Tensor, shape (K,)
        混合比の対数（内部表現）。`log_softmax` を通した値が posterior で使用される。
    _logC : torch.Tensor, shape (K,)
        `log C_d(κ_k)` のキャッシュ。
    n_iter_ : int
        実行された EM 反復回数。
    lower_bound_ : float
        最終反復の（近似）対数尤度（E-step の `logsumexp` を総和した値）。
    _fitted : bool
        学習済みフラグ。

    Notes
    -----
    - **入力は球面前提**：本実装は `X` を `F.normalize(X, dim=1)` して扱う。
      したがって、ユーザーが事前正規化していても動作は同じ（再正規化される）。
    - κ 更新は厳密な Newton 解ではなく、`R̄` からの近似式を用いる。
      高次元・高 κ 領域では近似誤差が出る可能性があるため、必要なら κ 更新を差し替える。
    - `chunk` ありの場合、初期化 (`_init_params`) でも大規模データ転送を避けるため
      サブサンプルを用いる設計になっている。
    - GPU/CPU 間転送を最小化したい場合、`chunk=None` で `X` を事前に device に載せる。
      VRAM が厳しい場合は `chunk` を指定し `X` は CPU に置いたまま fit する。
    """
    
    def __init__(
        self,
        n_components: int,
        d: Optional[int] = None,
        device: Union[str, torch.device] = "cuda",
        random_state: Optional[int] = 42,
        tol: float = 1e-4,
        max_iter: int = 200,
        init: str = "kmeans++",
        kappa_init: float = 10.0,
        kappa_min: float = 1e-6,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if n_components <= 0:
            raise ValueError("n_components must be positive")

        if init not in ("kmeans++", "random"):
            raise ValueError("init must be 'kmeans++' or 'random'")

        if not (kappa_init > 0):
            raise ValueError("kappa_init must be > 0")

        self.K = int(n_components)
        self.d: Optional[int] = int(d) if d is not None else None
        self.device = torch.device(device)
        self.random_state = random_state
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.init = str(init)
        self.kappa_init = float(kappa_init)
        self.kappa_min = float(kappa_min)
        self.dtype = dtype

        # Parameters (deferred allocation until d is known)
        self.register_buffer("mus", torch.empty(0, 0))   # (K,d)
        self.register_buffer("kappas", torch.empty(0))   # (K,)
        self.register_buffer("logpi", torch.empty(0))    # (K,)
        self.register_buffer("_logC", torch.empty(0))    # (K,)

        # caches / state
        self._nu: Optional[float] = None
        self._fitted: bool = False
        self.n_iter_: int = 0
        self.lower_bound_: float = float("-inf")

        # rng (CPU固定: deviceに依存させない)
        self._g = torch.Generator(device="cpu")
        if self.random_state is not None:
            self._g.manual_seed(int(self.random_state))

    # ------------------------- buffer/device helpers -------------------------
    @torch.no_grad()
    def _ensure_buffers_device_and_shape(self) -> None:
        """Ensure buffers exist on correct device and with correct shapes."""
        assert self.d is not None
        K, D = self.K, self.d
        dev = self.device
        dt = self.dtype

        if self.mus.numel() == 0 or self.mus.shape != (K, D) or self.mus.device != dev or self.mus.dtype != dt:
            self.mus = torch.empty(K, D, device=dev, dtype=dt)

        if self.kappas.numel() != K or self.kappas.device != dev or self.kappas.dtype != dt:
            self.kappas = torch.empty(K, device=dev, dtype=dt)

        if self.logpi.numel() != K or self.logpi.device != dev or self.logpi.dtype != dt:
            self.logpi = torch.zeros(K, device=dev, dtype=dt)

        if self._logC.numel() != K or self._logC.device != dev or self._logC.dtype != dt:
            self._logC = torch.empty(K, device=dev, dtype=dt)

    @torch.no_grad()
    def _allocate_buffers(self) -> None:
        assert self.d is not None, "d is not initialized"
        self._ensure_buffers_device_and_shape()

    @torch.no_grad()
    def _refresh_logC(self) -> None:
        assert self.d is not None
        self._logC.copy_(vmf_logC(int(self.d), self.kappas))

    # ------------------------- initialization -------------------------
    @torch.no_grad()
    def _init_params(self, X: torch.Tensor, *, chunk: Optional[int] = None) -> None:
        """Initialize parameters.

        Notes
        -----
        If ``chunk`` is provided and ``X`` is very large, initialization uses a
        random subset to avoid moving the full dataset to GPU.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {tuple(X.shape)}")

        N, D = int(X.size(0)), int(X.size(1))
        if self.d is None:
            self.d = D
        elif self.d != D:
            raise ValueError(f"X dim mismatch: expected {self.d}, got {D}")

        self._nu = 0.5 * float(self.d) - 1.0
        self._allocate_buffers()

        # --------------------------------------------------
        # Choose data for initialization (subset when chunked)
        #   - indices sampling is CPU-fixed to avoid generator/device mismatch
        # --------------------------------------------------
        X_src = X
        if chunk is not None and N > max(int(chunk), 0):
            m = min(N, max(10_000, 50 * self.K))
            perm = torch.randperm(N, generator=self._g)  # CPU
            idx = perm[:m].to(X.device)
            X_src = X.index_select(0, idx)

        Xn = F.normalize(X_src.to(self.device, dtype=self.dtype), dim=1)
        Nn = int(Xn.size(0))

        if self.init == "random":
            idx = torch.randint(0, Nn, (self.K,), generator=self._g, device="cpu").to(self.device)
            C = Xn.index_select(0, idx)
            self.mus.copy_(F.normalize(C, dim=1))
        else:
            # cosine k-means++ like seeding
            C = torch.empty(self.K, self.d, device=self.device, dtype=self.dtype)
            idx0 = int(torch.randint(0, Nn, (1,), generator=self._g, device="cpu").item())
            C[0] = Xn[idx0]
            dmin = 1.0 - (Xn @ C[0:1].T).squeeze(1)  # 1 - cos

            for k in range(1, self.K):
                probs = dmin.clamp_min(1e-8)
                probs = probs / probs.sum()
                idxk = int(torch.multinomial(probs, 1, generator=self._g).item())  # multinomial is device-safe here
                C[k] = Xn[idxk]
                d = 1.0 - (Xn @ C[k:k + 1].T).squeeze(1)
                dmin = torch.minimum(dmin, d)

            self.mus.copy_(F.normalize(C, dim=1))

        self.kappas.fill_(float(self.kappa_init))
        self.logpi.fill_(-math.log(float(self.K)))
        self._refresh_logC()

    # ------------------------- E / M with chunking -------------------------
    @torch.inference_mode()
    def _validate_chunk(self, chunk: Optional[int]) -> Optional[int]:
        if chunk is None:
            return None
        if not isinstance(chunk, int):
            raise ValueError(f"chunk must be int or None, got {type(chunk)}")
        if chunk <= 0:
            raise ValueError(f"chunk must be positive, got {chunk}")
        return chunk

    @torch.inference_mode()
    def _e_step_chunk(
        self,
        X: torch.Tensor,
        chunk: Optional[int],
        *,
        return_gamma: bool = True,
    ) -> Tuple[Optional[torch.Tensor], float, torch.Tensor, torch.Tensor]:
        """Chunked E-step (streaming)."""
        chunk = self._validate_chunk(chunk)
        N = int(X.size(0))
        K = self.K
        logpi = self.logpi.log_softmax(dim=0)  # (K,)

        gam = torch.empty(N, K, device=self.device, dtype=self.dtype) if return_gamma else None
        Nk = torch.zeros(K, device=self.device, dtype=self.dtype)
        Sk = torch.zeros(K, self.d, device=self.device, dtype=self.dtype)

        lb_total = 0.0

        def block(s: int, e: int) -> float:
            xb = X[s:e].to(self.device, dtype=self.dtype)
            xb = F.normalize(xb, dim=1)

            dot = xb @ self.mus.T  # (b,K)
            loglik_components = dot * self.kappas.unsqueeze(0) + self._logC.unsqueeze(0)  # (b,K)
            logpost = loglik_components + logpi.unsqueeze(0)  # (b,K)

            gb = torch.softmax(logpost, dim=1)  # (b,K)
            if return_gamma:
                gam[s:e] = gb

            Nk.add_(gb.sum(dim=0))
            Sk.add_(gb.T @ xb)

            return float(torch.logsumexp(logpost, dim=1).sum().item())

        if chunk is None or N <= chunk:
            lb_total += block(0, N)
        else:
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                lb_total += block(s, e)

        return gam, float(lb_total), Nk, Sk

    @torch.no_grad()
    def _m_step_from_stats(self, Nk: torch.Tensor, Sk: torch.Tensor, eps: float = 1e-8) -> None:
        """M-step using sufficient statistics."""
        Nk = Nk.to(self.device, dtype=self.dtype).clamp_min(eps)
        Sk = Sk.to(self.device, dtype=self.dtype)

        pi = Nk / Nk.sum()
        self.logpi.copy_(torch.log(pi.clamp_min(1e-20)))

        Sk_norm = torch.linalg.vector_norm(Sk, dim=1).clamp_min(eps)  # (K,)
        mu = Sk / Sk_norm.unsqueeze(1)
        self.mus.copy_(F.normalize(mu, dim=1))

        # kappa update
        Rbar = (Sk_norm / Nk).clamp(1e-6, 1.0 - 1e-6)
        Df = float(self.d)
        kappa = (Rbar * (Df - Rbar**2)) / (1.0 - Rbar**2 + 1e-8)
        self.kappas.copy_(kappa.clamp_min(self.kappa_min))
        self._refresh_logC()

    # ------------------------- Public API -------------------------
    @torch.no_grad()
    def fit(self, X: torch.Tensor, *, chunk: Optional[int] = None) -> "VMFMixture":
        r"""
        Fit the vMF mixture model by EM.

        Parameters
        ----------
        X : torch.Tensor, shape (N, d)
            入力特徴行列。`N` はサンプル数、`d` は特徴次元。
            NaN/Inf を含んではならない。内部で `F.normalize(X, dim=1)` を適用するため、
            入力はゼロベクトルを含まないことが望ましい。
        chunk : int | None, default=None
            E-step のチャンクサイズ。None の場合は全量を一括処理。
            int の場合は `X` を `chunk` 行ごとに device に転送して責務を計算し、
            十分統計 `(N_k, S_k)` を累積する（ストリーミング学習）。

        Returns
        -------
        self : VMFMixture
            学習済みインスタンス（チェーン可能）。

        Notes
        -----
        - `d` が None の場合は `X.shape[1]` で自動決定し、以後固定される。
        - 収束判定は lower bound（E-step での `logsumexp` 総和）の改善に基づく。
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {tuple(X.shape)}")
        if not torch.isfinite(X).all():
            raise ValueError("X contains NaN/Inf")

        self._init_params(X, chunk=chunk)

        prev: Optional[float] = None
        last_lb: float = -float("inf")

        for t in range(self.max_iter):
            _, lb, Nk, Sk = self._e_step_chunk(X, chunk, return_gamma=False)
            self._m_step_from_stats(Nk, Sk)

            self.n_iter_ = t + 1
            last_lb = float(lb)

            if prev is not None:
                rel = abs(last_lb - prev) / (abs(prev) + 1e-12)
                if (rel < self.tol) or (abs(last_lb - prev) < 1e-6):
                    break
            prev = last_lb

        self.lower_bound_ = float(last_lb)
        self._fitted = True
        return self

    @torch.no_grad()
    def predict_proba(self, X: torch.Tensor, *, chunk: Optional[int] = None) -> torch.Tensor:
        r"""
        Compute posterior responsibilities γ_{ik} for each sample.

        Parameters
        ----------
        X : torch.Tensor, shape (N, d)
            入力特徴。`fit()` と同じ次元 d が必要。
            内部で `F.normalize(X, dim=1)` を適用する。
        chunk : int | None, default=None
            E-step と同様のチャンク処理。大規模 `X` に対して VRAM を節約できる。

        Returns
        -------
        gamma : torch.Tensor, shape (N, K)
            各サンプル i に対する各成分 k の責務 `γ_{ik}`（行方向 softmax、総和 1）。

        Raises
        ------
        RuntimeError
            モデルが未学習（`fit()` 未実行）の場合。
        ValueError
            入力形状が不正、または `X.shape[1] != d` の場合。
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        if X.ndim != 2 or (self.d is not None and X.size(1) != self.d):
            raise ValueError(f"X must be (N,{self.d}), got {tuple(X.shape)}")
        gam, _, _, _ = self._e_step_chunk(X, chunk, return_gamma=True)
        assert gam is not None
        return gam

    @torch.no_grad()
    def predict(self, X: torch.Tensor, *, chunk: Optional[int] = None) -> torch.Tensor:
        r"""
        Predict hard cluster assignments (MAP component index).

        Parameters
        ----------
        X : torch.Tensor, shape (N, d)
            入力特徴。
        chunk : int | None, default=None
            `predict_proba` と同様。

        Returns
        -------
        labels : torch.Tensor, shape (N,)
            `argmax_k γ_{ik}` による割当ラベル（0..K-1）。
        """
        return self.predict_proba(X, chunk=chunk).argmax(dim=1)

    @torch.inference_mode()
    def loglik(self, X: torch.Tensor, *, chunk: Optional[int] = None, average: bool = False) -> float:
        r"""
        Compute (approximate) log-likelihood under the fitted model.

        Parameters
        ----------
        X : torch.Tensor, shape (N, d)
            入力特徴。内部で L2 正規化される。
        chunk : int | None, default=None
            大規模 `X` 用のチャンク計算。
        average : bool, default=False
            True の場合は 1 サンプル当たり平均（mean log-likelihood）を返す。

        Returns
        -------
        ll : float
            `Σ_i log Σ_k π_k C_d(κ_k) exp(κ_k μ_k^T x_i)` の総和、または平均。

        Notes
        -----
        - `logpi` は `log_softmax` を通した混合比として扱う（数値安定）。
        - 内部で `_logC` を用いる（ベッセル近似の影響を受ける）。
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        chunk = self._validate_chunk(chunk)
        N = int(X.size(0))
        logpi = self.logpi.log_softmax(dim=0).unsqueeze(0)  # (1,K)

        def block(s: int, e: int) -> torch.Tensor:
            xb = X[s:e].to(self.device, dtype=self.dtype)
            xb = F.normalize(xb, dim=1)
            dot = xb @ self.mus.T
            loglik_components = dot * self.kappas.unsqueeze(0) + self._logC.unsqueeze(0)
            return torch.logsumexp(loglik_components + logpi, dim=1).sum()

        if chunk is None or N <= chunk:
            total = block(0, N)
        else:
            total = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                total = total + block(s, e)

        total_val = float(total.item())
        return (total_val / N) if average else total_val

    @torch.inference_mode()
    def num_params(self) -> int:
        assert self.d is not None
        return int(self.K * self.d + (self.K - 1))

    @torch.inference_mode()
    def bic(self, X: torch.Tensor, *, chunk: Optional[int] = None) -> float:
        r"""
        Compute Bayesian Information Criterion (BIC).

        BIC = -2 * loglik(X) + p * log(N)

        Parameters
        ----------
        X : torch.Tensor, shape (N, d)
            入力特徴。
        chunk : int | None, default=None
            `loglik` と同様のチャンク計算。

        Returns
        -------
        bic : float
            BIC 値（小さいほど良い）。

        Notes
        -----
        - パラメータ数 `p` は簡易的に `K*d + (K-1)` を用いる（本実装の `num_params()`）。
        厳密には κ の自由度や制約（||μ||=1）を考慮した有効自由度の定義もあり得る。
        """
        if self.d is None:
            raise RuntimeError("Model not fitted (d is None)")
        if X.ndim != 2 or X.size(1) != self.d:
            raise ValueError(f"X must be (N,{self.d}), got {tuple(X.shape)}")
        N = int(X.size(0))
        ll = self.loglik(X, chunk=chunk, average=False)
        p = self.num_params()
        return -2.0 * ll + p * math.log(N)

    # ------------------------- Save / Load -------------------------
    def state_dict_vmf(self) -> Dict[str, Any]:
        r"""
        Export a lightweight state dict for vMF mixture parameters.

        Returns
        -------
        state : dict
            次を含む辞書（CPU tensor とメタ情報）:
            - K, d, device, dtype
            - mus, kappas, logpi, _logC （すべて CPU clone）
            - n_iter_, lower_bound_, _fitted
            - random_state, rng_state
            - tol, max_iter, init, kappa_init, kappa_min

        Notes
        -----
        - `torch.nn.Module.state_dict()` と異なり、本クラス固有の永続化形式を提供する。
        - 返り値は `torch.save()` でそのまま保存可能。
        """
        return {
            "K": self.K,
            "d": int(self.d) if self.d is not None else None,
            "device": str(self.device),
            "dtype": str(self.dtype).replace("torch.", ""),
            "mus": self.mus.detach().clone().cpu(),
            "kappas": self.kappas.detach().clone().cpu(),
            "logpi": self.logpi.detach().clone().cpu(),
            "_logC": self._logC.detach().clone().cpu(),
            "n_iter_": self.n_iter_,
            "lower_bound_": self.lower_bound_,
            "_fitted": self._fitted,
            "random_state": self.random_state,
            "rng_state": self._g.get_state(),
            "tol": self.tol,
            "max_iter": self.max_iter,
            "init": self.init,
            "kappa_init": self.kappa_init,
            "kappa_min": self.kappa_min,
        }

    @torch.no_grad()
    def save(self, path: str) -> None:
        r"""
        Save the model to a file.

        Parameters
        ----------
        path : str
            保存先パス。`torch.save(self.state_dict_vmf(), path)` を実行する。

        Notes
        -----
        - すべて CPU tensor として保存されるため、環境依存（GPU 有無）の影響を受けにくい。
        """
        torch.save(self.state_dict_vmf(), path)

    @classmethod
    @torch.no_grad()
    def load(cls, path: str, map_location: Union[str, torch.device, None] = None) -> "VMFMixture":
        r"""
        Load a saved vMF mixture model.

        Parameters
        ----------
        path : str
            `save()` で保存したファイルパス。
        map_location : str | torch.device | None, default=None
            `torch.load` の `map_location`。CPU でロードしたい場合は "cpu" を指定。

        Returns
        -------
        model : VMFMixture
            復元されたモデル。`mus/kappas/logpi/_logC` を含み、`_fitted=True` となる。

        Raises
        ------
        RuntimeError
            保存データが不正で `d=None` のまま復元できない場合など。

        Notes
        -----
        - `device` と `dtype` は保存データの値を優先する（ただし map_location により tensor の実体は変わり得る）。
        - `rng_state` が保存されていれば CPU Generator の状態も復元する。
        """
        sd = torch.load(path, map_location=map_location)
        K = int(sd["K"])
        d = sd["d"]
        device = sd.get("device", "cpu")
        dtype_str = sd.get("dtype", "float32")
        dtype = getattr(torch, dtype_str, torch.float32)

        obj = cls(
            n_components=K,
            d=d,
            device=device,
            dtype=dtype,
            random_state=sd.get("random_state", None),
            tol=float(sd.get("tol", 1e-4)),
            max_iter=int(sd.get("max_iter", 200)),
            init=str(sd.get("init", "kmeans++")),
            kappa_init=float(sd.get("kappa_init", 10.0)),
            kappa_min=float(sd.get("kappa_min", 1e-6)),
        )

        if obj.d is None:
            raise RuntimeError("Loaded model has d=None; cannot allocate buffers")
        obj._allocate_buffers()

        obj.mus.copy_(sd["mus"].to(obj.device, dtype=obj.dtype))
        obj.kappas.copy_(sd["kappas"].to(obj.device, dtype=obj.dtype))
        obj.logpi.copy_(sd["logpi"].to(obj.device, dtype=obj.dtype))
        obj._logC.copy_(sd["_logC"].to(obj.device, dtype=obj.dtype))
        obj._nu = 0.5 * float(obj.d) - 1.0
        obj.n_iter_ = int(sd.get("n_iter_", 0))
        obj.lower_bound_ = float(sd.get("lower_bound_", float("-inf")))
        obj._fitted = bool(sd.get("_fitted", True))

        rng_state = sd.get("rng_state", None)
        if rng_state is not None:
            obj._g.set_state(rng_state)
        return obj


@torch.no_grad()
def elbow_vmf(
    cluster_module: Callable[..., "VMFMixture"],
    X: torch.Tensor,
    device: str = "cuda",
    k_max: int = 50,
    chunk: Optional[int] = None,
    verbose: bool = True,
    random_state: int = 42,
    criterion: str = "bic",   # {"bic", "nll"}
) -> Tuple[List[int], List[float], int, int, float]:
    r"""
    Sweep K and compute model selection scores (BIC or mean NLL), then estimate an elbow by curvature.

    Parameters
    ----------
    cluster_module : Callable[..., VMFMixture]
        `VMFMixture` を返す callable（通常は VMFMixture クラス自身）。
        例: `elbow_vmf(VMFMixture, X, ...)`
    X : torch.Tensor, shape (N, d)
        入力特徴。
    device : str, default="cuda"
        各 K のモデル学習に用いるデバイス。
    k_max : int, default=50
        K を 1..k_max で走査する。
    chunk : int | None, default=None
        None の場合、`X` を一度 device に載せて各 K で再利用する（高速だが VRAM 使用）。
        int の場合、`X` は CPU のままでもよく、fit/predict の E-step を chunk ストリーミングで処理する。
    verbose : bool, default=True
        各 K のスコアを表示する。
    random_state : int, default=42
        各 K の初期化に用いる乱数シード。
    criterion : {"bic", "nll"}, default="bic"
        - "bic": `vmf.bic(X)` を評価（小さいほど良い）
        - "nll": `-vmf.loglik(X, average=True)` を評価（小さいほど良い）

    Returns
    -------
    k_list : list[int]
        走査した K のリスト（1..k_max）。
    scores : list[float]
        各 K のスコア（criterion に依存）。小さいほど良い。
    K_elbow : int
        曲率ベースの elbow 推定による最適 K。
    idx_elbow : int
        `k_list` 上の elbow インデックス。
    curvature : float
        elbow 推定に使った曲率スカラー。

    Notes
    -----
    - 曲率計算は `find_elbow_curvature` を利用し、内部ではスコア系列を符号反転して
      “増加系列”として扱っている（実装依存）。
    - BIC と elbow の最適 K は一致しないことがあるため、用途に応じて採用基準を決める。
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D")

    if chunk is None:
        X_input = X.to(device, non_blocking=True)
    else:
        X_input = X

    if criterion not in ("bic", "nll"):
        raise ValueError("criterion must be 'bic' or 'nll'")

    scores: List[float] = []
    k_list = list(range(1, k_max + 1))

    for k in k_list:
        vmf = cluster_module(
            n_components=k,
            d=None,
            device=device,
            random_state=random_state,
            tol=1e-4,
            max_iter=200,
        )
        vmf.fit(X_input, chunk=chunk)

        if criterion == "bic":
            val = float(vmf.bic(X, chunk=chunk))
            tag = "BIC"
        else:
            nll = -float(vmf.loglik(X, chunk=chunk, average=True))
            val = nll
            tag = "mean_NLL"

        scores.append(val)
        if verbose:
            print(f"k={k}, {tag}={val:.6f}")

        gc.collect()
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    from .ops import find_elbow_curvature
    series_for_curv = [-s for s in scores]
    K, idx, kappa = find_elbow_curvature(k_list, series_for_curv)

    if verbose:
        best_idx = int(min(range(len(scores)), key=lambda i: scores[i]))
        print(f"Optimal k (curvature): {K}  |  Best-by-{criterion}: k={k_list[best_idx]}, score={scores[best_idx]:.6f}")

    return k_list, scores, K, idx, kappa