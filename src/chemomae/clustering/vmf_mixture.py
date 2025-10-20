from __future__ import annotations
import gc
import math
from typing import Optional, Tuple, List, Union, Dict, Any, Callable

import torch
import torch.nn as nn

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
    a2 = (k ** 4) / (32.0 * (nu + 1.0) * (nu + 2.0))
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
    c2 = (mu - 1.0) * (mu - 9.0) * (invk * invk) / (2.0 * (8.0 ** 2))
    corr = 1.0 + c1 + c2
    return k - 0.5 * (torch.log(2.0 * math.pi * k.clamp_min(1e-6))) + torch.log(corr.clamp_min(eps))


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
    # weight based on κ
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
    # small-k approx
    a = two_nu + 2.0
    b = two_nu + 4.0
    R_small = (k_cl / a) * (1.0 - (k_cl * k_cl) / (2.0 * a * b))
    # large-k approx
    R_large = 1.0 - (two_nu + 1.0) / (2.0 * k_cl) + (4.0 * nu * nu - 1.0) / (8.0 * (k_cl * k_cl))
    # blend
    K1 = 2.0
    K2 = 12.0
    w = ((k - K1) / (K2 - K1)).clamp(0.0, 1.0)
    R = torch.where(k <= K1, R_small, torch.where(k >= K2, R_large, _blend(R_small, R_large, w)))
    return R.clamp(0.0 + 1e-6, 1.0 - 1e-6)


def vmf_logC(d: int, kappa: torch.Tensor) -> torch.Tensor:
    """Compute log C_d(kappa) with approximated log I_ν.
    C_d(κ) = κ^ν / [(2π)^{ν+1} I_ν(κ)],   ν = d/2 - 1
    Returns tensor with the same shape as kappa.
    """
    nu = 0.5 * float(d) - 1.0
    nu_t = torch.as_tensor(nu, dtype=kappa.dtype, device=kappa.device)
    logIv = _logIv_piecewise(nu_t, kappa)
    return nu_t * torch.log(kappa.clamp_min(1e-12)) - (nu_t + 1.0) * math.log(2.0 * math.pi) - logIv


# VMFMixture: EM with chunked E-step / sufficient-statistics accumulation.
class VMFMixture(nn.Module):
    r"""
    vMF Mixture Model with chunked E-step and torch-only Bessel approximations.

    概要
    ----
    - 分布: vMF(x | μ_k, κ_k) ∝ exp(κ_k μ_k^T x),  ‖x‖=‖μ_k‖=1
    - 近似: log I_ν, I_{ν+1}/I_ν を小/大 κ の級数・漸近展開で近似し滑らかに接続
    - E-step: チャンク処理で責務 γ_{ik}
    - M-step: μ は重み付き平均→正規化, κ は R̄ から近似アップデート

    Parameters
    ----------
    n_components : int
        混合成分数 K。
    d : Optional[int]
        特徴次元（単位球面 S^{d-1}）。**省略時は fit(X) で X.size(1) から自動決定**。
    device : str | torch.device, default="cuda"
        学習・推論デバイス。
    random_state : int | None, default=42
        乱数シード。
    tol : float, default=1e-4
        収束判定（対数尤度の相対変化）。
    max_iter : int, default=200
        EM 反復の最大回数。
    init : {"kmeans++", "random"}, default="kmeans++"
        μ 初期化方式。"kmeans++" は球面コサイン版。
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
    ) -> None:
        super().__init__()
        if n_components <= 0:
            raise ValueError("n_components must be positive")

        self.K = int(n_components)
        self.d: Optional[int] = int(d) if d is not None else None
        self.device = torch.device(device)
        self.random_state = random_state
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.init = init

        # Parameters (deferred allocation until d is known)
        self.register_buffer("mus", torch.empty(0, 0))   # (K,d) after init
        self.register_buffer("kappas", torch.empty(0))   # (K,)
        self.register_buffer("logpi", torch.empty(0))    # (K,)
        self.register_buffer("_logC", torch.empty(0))    # (K,)

        # caches / state
        self._nu: Optional[float] = None
        self._fitted: bool = False
        self.n_iter_: int = 0
        self.lower_bound_: float = float("-inf")

        # rng
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
        # mus
        if (not hasattr(self, "mus")) or self.mus.numel() == 0 or self.mus.shape != (K, D) or self.mus.device != dev:
            self.mus = torch.empty(K, D, device=dev)
        else:
            self.mus = self.mus.to(dev)
            if self.mus.shape != (K, D):
                self.mus = torch.empty(K, D, device=dev)
        # kappas
        if (not hasattr(self, "kappas")) or self.kappas.numel() != K or self.kappas.device != dev:
            self.kappas = torch.empty(K, device=dev)
        else:
            self.kappas = self.kappas.to(dev)
            if self.kappas.numel() != K:
                self.kappas = torch.empty(K, device=dev)
        # logpi
        if (not hasattr(self, "logpi")) or self.logpi.numel() != K or self.logpi.device != dev:
            self.logpi = torch.zeros(K, device=dev)
        else:
            self.logpi = self.logpi.to(dev)
            if self.logpi.numel() != K:
                self.logpi = torch.zeros(K, device=dev)
        # _logC
        if (not hasattr(self, "_logC")) or self._logC.numel() != K or self._logC.device != dev:
            self._logC = torch.empty(K, device=dev)
        else:
            self._logC = self._logC.to(dev)
            if self._logC.numel() != K:
                self._logC = torch.empty(K, device=dev)

    @torch.no_grad()
    def _allocate_buffers(self) -> None:
        assert self.d is not None, "d is not initialized"
        self._ensure_buffers_device_and_shape()

    # ------------------------- initialization -------------------------
    @torch.no_grad()
    def _init_params(self, X: torch.Tensor) -> None:
        N, D = int(X.size(0)), int(X.size(1))
        if self.d is None:
            self.d = D
        elif self.d != D:
            raise ValueError(f"X dim mismatch: expected {self.d}, got {D}")

        self._nu = 0.5 * float(self.d) - 1.0
        self._allocate_buffers()

        # normalize data
        Xn = torch.nn.functional.normalize(X.to(self.device, dtype=torch.float32), dim=1)

        # init mus by cosine k-means++ like seeding
        C = torch.empty(self.K, self.d, device=self.device)
        idx0 = torch.randint(0, N, (1,), generator=self._g).item()
        C[0] = Xn[idx0]
        dmin = 1.0 - (Xn @ C[0:1].T).squeeze(1)  # 1 - cos
        dmin = dmin.clamp_min_(1e-12)
        probs = dmin / (dmin.sum() + 1e-12)
        for k in range(1, self.K):
            idx = torch.multinomial(probs.cpu(), 1, generator=self._g).item()
            C[k] = Xn[idx]
            dk = 1.0 - (Xn @ C[k:k+1].T).squeeze(1)
            dmin = torch.minimum(dmin, dk).clamp_min_(1e-12)
            probs = dmin / (dmin.sum() + 1e-12)
        self.mus.copy_(torch.nn.functional.normalize(C, dim=1))

        # init kappas by average resultant length per cluster assignment once
        labels = (Xn @ self.mus.T).argmax(dim=1)
        kappa = torch.empty(self.K, device=self.device)
        for k in range(self.K):
            mask = labels == k
            if mask.any():
                r = Xn[mask].mean(dim=0)
                Rbar = r.norm().clamp(1e-6, 1 - 1e-6)  # in [0,1]
                Df = float(self.d)
                kappa[k] = (Rbar * (Df - Rbar**2)) / (1.0 - Rbar**2)
            else:
                kappa[k] = 1.0
        self.kappas.copy_(kappa.clamp_min(1e-6))
        self.logpi.copy_(torch.full((self.K,), -math.log(self.K), device=self.device))
        self._refresh_logC()

    @torch.no_grad()
    def _refresh_logC(self) -> None:
        assert self.d is not None, "d is not initialized"
        self._logC.copy_(vmf_logC(self.d, self.kappas))

    # ------------------------- E / M with chunking -------------------------
    @torch.inference_mode()
    def _e_step_chunk(self, X: torch.Tensor, chunk: Optional[int]) -> Tuple[torch.Tensor, float]:
        """Return responsibilities γ (N,K) and lower bound (approx log-lik)."""
        X = torch.nn.functional.normalize(X.to(self.device, dtype=torch.float32), dim=1)
        N = int(X.size(0))
        K = self.K
        logpi = self.logpi.log_softmax(dim=0)  # (K,)
        gam = torch.empty(N, K, device=self.device, dtype=torch.float32)

        def block(s: int, e: int) -> float:
            x = X[s:e]
            dot = x @ self.mus.T                     # (b,K)
            loglik_components = dot * self.kappas.unsqueeze(0) + self._logC.unsqueeze(0)
            logpost = loglik_components + logpi.unsqueeze(0)
            gam[s:e] = torch.softmax(logpost, dim=1)  # responsibilities
            return torch.logsumexp(logpost, dim=1).sum().item()

        if (chunk is None) or (chunk <= 0) or (N <= (chunk or 0)):
            lb = block(0, N)
        else:
            lb = 0.0
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                lb += block(s, e)
        return gam, float(lb)

    @torch.no_grad()
    def _m_step(self, X: torch.Tensor, gam: torch.Tensor, eps: float = 1e-8) -> None:
        X = torch.nn.functional.normalize(X.to(self.device, dtype=torch.float32), dim=1)
        Nk = gam.sum(dim=0).clamp_min(eps)              # (K,)
        # mixing weights
        pi = Nk / Nk.sum()
        self.logpi.copy_(torch.log(pi))
        # mean directions
        mu = (gam.T @ X) / Nk.unsqueeze(1)              # (K,d)
        mu = torch.nn.functional.normalize(mu, dim=1)
        self.mus.copy_(mu)
        # update kappa via resultant length
        Rnorm = torch.linalg.vector_norm((gam.T @ X), dim=1) / Nk  # (K,)
        Rbar = Rnorm.clamp(1e-6, 1 - 1e-6)
        Df = float(self.d)
        kappa = (Rbar * (Df - Rbar**2)) / (1.0 - Rbar**2 + 1e-8)
        self.kappas.copy_(kappa.clamp_min(1e-6))
        self._refresh_logC()

    # ------------------------- Public API -------------------------
    @torch.no_grad()
    def fit(self, X: torch.Tensor, *, chunk: Optional[int] = None) -> "VMFMixture":
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {tuple(X.shape)}")
        if not torch.isfinite(X).all():
            raise ValueError("X contains NaN/Inf")

        # (X is moved to device inside _init_params/_e_step_chunk too; this is safe)
        self._init_params(X)
        prev = None
        for t in range(self.max_iter):
            gam, lb = self._e_step_chunk(X, chunk)
            self._m_step(X, gam)
            self.n_iter_ = t + 1
            # relative improvement
            if prev is not None:
                rel = abs(lb - prev) / (abs(prev) + 1e-12)
                if (rel < self.tol) or (abs(lb - prev) < 1e-6):
                    prev = lb
                    break
            prev = lb
        self.lower_bound_ = float(prev if prev is not None else lb)
        self._fitted = True
        return self

    @torch.no_grad()
    def predict_proba(self, X: torch.Tensor, *, chunk: Optional[int] = None) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        if X.ndim != 2 or (self.d is not None and X.size(1) != self.d):
            raise ValueError(f"X must be (N,{self.d}), got {tuple(X.shape)}")
        gam, _ = self._e_step_chunk(X, chunk)
        return gam

    @torch.no_grad()
    def predict(self, X: torch.Tensor, *, chunk: Optional[int] = None) -> torch.Tensor:
        return self.predict_proba(X, chunk=chunk).argmax(dim=1)

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        """Rejection sampling on S^{d-1} for each component, mixing by π."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        device = self.device
        d = int(self.d) if self.d is not None else None
        if d is None:
            raise RuntimeError("d is not initialized")
        pi = self.logpi.log_softmax(dim=0).exp()
        comp = torch.multinomial(pi, n, replacement=True, generator=self._g)
        out = torch.empty(n, d, device=device)
        for k in range(self.K):
            m = (comp == k).sum().item()
            if m == 0:
                continue
            mu = self.mus[k]
            kappa = self.kappas[k]
            out[comp == k] = _sample_vmf(mu, kappa, m)
        return out

    # ------------------------- Model diagnostics -------------------------
    @torch.inference_mode()
    def loglik(self, X: torch.Tensor, *, chunk: Optional[int] = None, average: bool = False) -> float:
        """Total (or average) log-likelihood under current parameters."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        Xn = torch.nn.functional.normalize(X.to(self.device, dtype=torch.float32), dim=1)
        N = int(Xn.size(0))
        logpi = self.logpi.log_softmax(dim=0).unsqueeze(0)  # (1,K)

        def block(s: int, e: int) -> torch.Tensor:
            x = Xn[s:e]
            dot = x @ self.mus.T
            loglik_components = dot * self.kappas.unsqueeze(0) + self._logC.unsqueeze(0)
            return torch.logsumexp(loglik_components + logpi, dim=1).sum()

        if (chunk is None) or (chunk <= 0) or (N <= (chunk or 0)):
            total = block(0, N)
        else:
            total = torch.tensor(0.0, device=self.device)
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                total = total + block(s, e)
        total_val = float(total.item())
        return (total_val / N) if average else total_val

    @torch.inference_mode()
    def num_params(self) -> int:
        """自由度 p for BIC.
        Each component: μ has (d-1) dof on hypersphere + κ (1). Mixing weights contribute (K-1).
        → p = K*(d-1+1) + (K-1) = K*d + (K-1).
        """
        assert self.d is not None
        return int(self.K * self.d + (self.K - 1))

    @torch.inference_mode()
    def bic(self, X: torch.Tensor, *, chunk: Optional[int] = None) -> float:
        """Bayesian Information Criterion for the fitted model on X.
        BIC = -2 * loglik + p * log(N)
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
        """Return a lightweight, torch.save-friendly state dict."""
        return {
            "K": self.K,
            "d": int(self.d) if self.d is not None else None,
            "device": str(self.device),
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
        }

    @torch.no_grad()
    def save(self, path: str) -> None:
        torch.save(self.state_dict_vmf(), path)

    @classmethod
    @torch.no_grad()
    def load(cls, path: str, map_location: Union[str, torch.device, None] = None) -> "VMFMixture":
        sd = torch.load(path, map_location=map_location)
        K = int(sd["K"])
        d = sd["d"]
        device = sd.get("device", "cpu")
        obj = cls(n_components=K, d=d, device=device,
                  random_state=sd.get("random_state", None),
                  tol=float(sd.get("tol", 1e-4)),
                  max_iter=int(sd.get("max_iter", 200)),
                  init=str(sd.get("init", "kmeans++")))
        # allocate buffers on device with correct shapes
        if obj.d is None:
            raise RuntimeError("Loaded model has d=None; cannot allocate buffers")
        obj._allocate_buffers()

        # restore tensors (move to device)
        obj.mus.copy_(sd["mus"].to(obj.device))
        obj.kappas.copy_(sd["kappas"].to(obj.device))
        obj.logpi.copy_(sd["logpi"].to(obj.device))
        obj._logC.copy_(sd["_logC"].to(obj.device))
        obj._nu = 0.5 * float(obj.d) - 1.0
        obj.n_iter_ = int(sd.get("n_iter_", 0))
        obj.lower_bound_ = float(sd.get("lower_bound_", float("-inf")))
        obj._fitted = bool(sd.get("_fitted", True))
        rng_state = sd.get("rng_state", None)
        if rng_state is not None:
            obj._g.set_state(rng_state)
        return obj


# vMF sampler (eager mode)
@torch.no_grad()
def _sample_vmf(mu: torch.Tensor, kappa: torch.Tensor, n: int) -> torch.Tensor:
    """Wood's method (approx) with Householder transform, torch-only.
    mu: (d,), ‖mu‖=1,  kappa>0
    return: (n,d)
    """
    d = mu.numel()
    device = mu.device
    dtype = mu.dtype

    # sample on R^{d-1}
    v = torch.randn(n, d - 1, device=device, dtype=dtype)
    v = torch.nn.functional.normalize(v, dim=1)

    # Beta/Uniform proposals (distributions.* は TorchScript 非対応なので eager で使用)
    a = (d - 1) * 0.5
    beta = torch.distributions.Beta(torch.as_tensor(a, device=device, dtype=dtype),
                                    torch.as_tensor(a, device=device, dtype=dtype))
    unif = torch.distributions.Uniform(torch.as_tensor(0.0, device=device, dtype=dtype),
                                       torch.as_tensor(1.0, device=device, dtype=dtype))

    out_w = torch.empty(0, device=device, dtype=dtype)
    need = n
    bb = torch.sqrt(4.0 * (kappa * kappa) + (d - 1) ** 2)
    B = (bb - 2.0 * kappa) / (d - 1)
    A = (bb + 2.0 * kappa + (d - 1)) * 0.25
    D = (4.0 * A * B) / (1.0 + B) - (d - 1) * torch.log(torch.as_tensor(float(d - 1), device=device, dtype=dtype))

    while out_w.numel() < n:
        m = max(4, 2 * need)
        eps = beta.sample((m,))
        u = unif.sample((m,))
        w = (1.0 - (1.0 + B) * eps) / (1.0 - (1.0 - B) * eps)
        t = (2.0 * A * B) / (1.0 - (1.0 - B) * eps)
        det = (d - 1) * torch.log(t.clamp_min(1e-12)) - t + D - torch.log(u.clamp_min(1e-12))
        acc = w[det >= 0]
        if acc.numel() > 0:
            out_w = torch.cat([out_w, acc])
            need = n - out_w.numel()

    w = out_w[:n].unsqueeze(1)
    x = torch.cat([w, torch.sqrt((1.0 - w * w).clamp_min(0.0)) * v], dim=1)  # (n,d)

    # Householder: map e1 -> mu
    e1 = torch.zeros(d, 1, device=device, dtype=dtype)
    e1[0, 0] = 1.0
    u = e1 - mu.view(-1, 1)
    u_norm = torch.linalg.vector_norm(u, dim=0).clamp_min(1e-12)
    u = u / u_norm
    Hx = x - 2.0 * (x @ u) @ u.T
    return Hx


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
    Sweep K=1..k_max for vMF Mixture and pick an elbow by curvature.

    概要
    ----
    - `K = 1..k_max` について VMFMixture（`cluster_module`）を学習し、評価値を記録。
      - `criterion="bic"`:  BIC（小さいほど良い）
      - `criterion="nll"`:  平均負対数尤度 = -mean loglik（小さいほど良い）
    - エルボー推定は `find_elbow_curvature(k_list, series_decreasing)` に委譲。
      ここで
        - bic の場合は `series_decreasing = [-bic for bic in series]`（単調減少に変換）
        - nll の場合は `series_decreasing = [-nll for nll in series]`（単調減少に変換）
    - 返り値は `(k_list, scores, optimal_k, elbow_idx, kappa)`。

    Parameters
    ----------
    cluster_module : Callable[..., VMFMixture]
        `VMFMixture` 互換のコンストラクタ。
        呼び出し側で `n_components`, `device`, `random_state` などを渡せる必要があります。
    X : torch.Tensor, shape (N, D)
        入力特徴行列。内部で `device` へ転送します（VMFMixture 側でも正規化されます）。
    device : {"cuda", "cpu"} | torch.device, default="cuda"
        学習に用いるデバイス。
    k_max : int, default=50
        試すクラスタ数の上限（1..k_max を走査）。
    chunk : int | None, default=None
        ストリーミング学習のチャンクサイズ。`None` ならフルデバイス、
        `>0` なら CPU→GPU ストリーミング（`VMFMixture.fit(..., chunk=...)` に委譲）。
    verbose : bool, default=True
        各 K での評価値をログ表示。
    random_state : int, default=42
        乱数シード（初期化に影響）。
    criterion : {"bic", "nll"}, default="bic"
        エルボー探索に用いる指標。

    Returns
    -------
    k_list : List[int]
        走査したクラスタ数（1..k_max）。
    scores : List[float]
        各 K に対する評価値（`criterion` で選んだもの）。
        - bic:  BIC（小さいほど良い）
        - nll:  平均負対数尤度（小さいほど良い）
    optimal_k : int
        曲率（“折れ曲がり”）により選ばれた推奨クラスタ数。
    elbow_idx : int
        `k_list[elbow_idx] == optimal_k` を満たすインデックス。
    kappa : float
        エルボー点の曲率スコア（大きいほどエルボーが明確）。

    Notes
    -----
    - BIC は単調減少でないことがあるため、曲率計算では -BIC を使います（形状だけ利用）。
      実際の「最良BIC」は `min(scores)` も参考にしてください。
    - 大きな N の場合、`chunk` を設定すると VRAM を抑えて計算できます。
    - `find_elbow_curvature` はローカル util（循環回避のため関数内 import）。
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D")

    X = X.to(device, non_blocking=True)

    if criterion not in ("bic", "nll"):
        raise ValueError("criterion must be 'bic' or 'nll'")

    scores: List[float] = []
    k_list = list(range(1, k_max + 1))

    for k in k_list:
        vmf = cluster_module(
            n_components=k,
            d=None,                    # fit 時に自動検出
            device=device,
            random_state=random_state,
            tol=1e-4,
            max_iter=200,
        )
        vmf.fit(X, chunk=chunk)

        if criterion == "bic":
            val = float(vmf.bic(X, chunk=chunk))
            tag = "BIC"
        else:
            # 平均 NLL = - 平均 loglik
            nll = -float(vmf.loglik(X, chunk=chunk, average=True))
            val = nll
            tag = "mean_NLL"

        scores.append(val)
        if verbose:
            print(f"k={k}, {tag}={val:.6f}")

        # メモリ掃除（GPUを使っている時のみ）
        gc.collect()
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 曲率ベースのエルボー（単調減少列にしてから）
    from .ops import find_elbow_curvature
    series_for_curv = [-s for s in scores]  # “良いほうが大きい”形に変換
    K, idx, kappa = find_elbow_curvature(k_list, series_for_curv)

    if verbose:
        # 参考に min-BIC / min-NLL も出しておく
        best_idx = int(min(range(len(scores)), key=lambda i: scores[i]))
        print(f"Optimal k (curvature): {K}  |  Best-by-{criterion}: k={k_list[best_idx]}, score={scores[best_idx]:.6f}")

    return k_list, scores, K, idx, kappa