from __future__ import annotations

import gc
import math
from typing import Optional, Tuple, List, Union, Dict, Any, Callable

import numpy as np
from scipy import special
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["VMFMixture", "elbow_vmf", "vmf_logC", "vmf_bessel_ratio"]


def _log_bessel_series(nu: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Log of the positive series after factoring out (k/2)^nu / Gamma(nu+1).

    Used for small k and scaled-Bessel underflow. Log-space summation avoids
    underflow in high dimensions; the decreasing-term tail bounds truncation.
    See https://dlmf.nist.gov/10.25.E2.
    """
    log_term = np.zeros_like(k)
    log_sum = np.zeros_like(k)
    log_z = 2.0 * (np.log(k) - math.log(2.0))
    log_eps = math.log(np.finfo(np.float64).eps)
    for m in range(1, 10_001):
        log_term += log_z - math.log(m) - np.log(nu + m)
        log_sum = np.logaddexp(log_sum, log_term)
        log_next_ratio = log_z - math.log(m + 1) - np.log(nu + m + 1)
        if np.all(log_next_ratio < 0):
            log_tail = log_term + log_next_ratio - np.log(-np.expm1(log_next_ratio))
            if np.all(log_tail - log_sum <= log_eps):
                return log_sum
    raise FloatingPointError("Bessel series did not converge within 10000 terms")


def _bessel_inputs(nu: torch.Tensor, k: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Validate the vMF domain and broadcast in CPU float64."""
    if not k.is_floating_point() or nu.is_complex():
        raise ValueError("Bessel inputs must be real, with floating-point kappa")
    orders, values = np.broadcast_arrays(
        nu.detach().to(device="cpu", dtype=torch.float64).numpy(),
        k.detach().to(device="cpu", dtype=torch.float64).numpy(),
    )
    if not np.all(np.isfinite(orders) & (orders >= 0)):
        raise ValueError("nu must be finite and nonnegative (d >= 2)")
    if not np.all(np.isfinite(values) & (values >= 0)):
        raise ValueError("kappa must be finite and nonnegative")
    return orders, values


def vmf_bessel_ratio(nu: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Compute I_{nu+1}(k)/I_nu(k), including the exact zero limit.

    SciPy scaled Bessel functions and a convergent-series fallback are evaluated
    on CPU in float64. The result has k's dtype/device and the broadcast shape.
    This numerical helper does not build an autograd graph.
    """
    orders, values = _bessel_inputs(nu, k)
    result = np.zeros_like(values)
    positive = values > 0
    v, x = orders[positive], values[positive]
    denominator = special.ive(v, x)
    numerator = special.ive(v + 1.0, x)
    regular = (x > 1.0) & (denominator > np.finfo(np.float64).tiny) & (
        numerator > np.finfo(np.float64).tiny
    )
    ratio = np.empty_like(x)
    ratio[regular] = numerator[regular] / denominator[regular]
    fallback = ~regular
    if np.any(fallback):
        vf, xf = v[fallback], x[fallback]
        ratio[fallback] = (xf / (2.0 * (vf + 1.0))) * np.exp(
            _log_bessel_series(vf + 1.0, xf) - _log_bessel_series(vf, xf)
        )
    if not np.all(np.isfinite(ratio)):
        raise FloatingPointError("Bessel ratio is not finite for the supplied inputs")
    result[positive] = np.clip(ratio, 0.0, 1.0)
    return torch.as_tensor(result, dtype=k.dtype, device=k.device)


def vmf_logC(d: int, kappa: torch.Tensor) -> torch.Tensor:
    """Compute log C_d(kappa), including the uniform-sphere limit at zero.
    C_d(κ) = κ^ν / [(2π)^{ν+1} I_ν(κ)],   ν = d/2 - 1

    CPU float64 scaled Bessel evaluation falls back to a convergent positive
    series for small kappa or underflow. Returns kappa's shape, dtype and device;
    this numerical helper does not build an autograd graph.
    """
    if not isinstance(d, int) or isinstance(d, bool) or d < 2:
        raise ValueError("d must be an integer >= 2")
    nu = 0.5 * float(d) - 1.0
    orders, values = _bessel_inputs(torch.tensor(nu, dtype=torch.float64), kappa)
    uniform = special.gammaln(0.5 * d) - math.log(2.0) - 0.5 * d * math.log(math.pi)
    result = np.full_like(values, uniform)
    positive = values > 0
    x = values[positive]
    scaled = special.ive(nu, x)
    regular = (x > 1.0) & (scaled > np.finfo(np.float64).tiny)
    logc = np.empty_like(x)
    logc[regular] = (
        nu * np.log(x[regular]) - (nu + 1.0) * math.log(2.0 * math.pi)
        - np.log(scaled[regular]) - x[regular]
    )
    fallback = ~regular
    if np.any(fallback):
        logc[fallback] = uniform - _log_bessel_series(orders[positive][fallback], x[fallback])
    if not np.all(np.isfinite(logc)):
        raise FloatingPointError("log C_d is not finite for the supplied inputs")
    result[positive] = logc
    return torch.as_tensor(result, dtype=kappa.dtype, device=kappa.device)


class VMFMixture(nn.Module):
    r"""
    von Mises–Fisher (vMF) Mixture Model on the unit hypersphere with chunked EM.

    Overview
    ----
    Fit a vMF mixture on the unit hypersphere by EM.
    Input features `X` are treated as row-wise L2-normalized vectors (`||x_i||=1`).
    Each component k has a unit direction `μ_k` and concentration `κ_k`.

    Distribution
    ----
    - vMF density in d dimensions:
      `p(x | μ, κ) = C_d(κ) * exp(κ μ^T x)`,  `||x||=||μ||=1`
    - The normalization constant `C_d(κ)` contains the Bessel function `I_ν`:
      `C_d(κ) = κ^ν / ((2π)^(ν+1) I_ν(κ))`,  `ν = d/2 - 1`

    Optimization (EM)
    -----------
    Maximize the log-likelihood
    `L = Σ_i log Σ_k π_k * C_d(κ_k) * exp(κ_k μ_k^T x_i)`

    - E-step:
      Posterior responsibilities
      `γ_{ik} = p(z_i=k | x_i) = softmax_k( log π_k + log C_d(κ_k) + κ_k μ_k^T x_i )`
    - M-step:
      - Mixture weights: `π_k = (1/N) Σ_i γ_{ik}`
      - Directions: `s_k = Σ_i γ_{ik} x_i`, `μ_k = s_k / ||s_k||`
      - Concentrations: update approximately from `R̄_k = ||s_k|| / Σ_i γ_{ik}`
        using a closed-form approximation.

    Numerics
    --------
    Compute normalization constants using SciPy scaled Bessel functions in CPU float64.
    For small κ or underflow, use the convergent defining series instead.
    Log densities and likelihood aggregation also use float64; directions and
    sufficient statistics use the configured dtype.

    Streaming (chunked E-step)
    -------------------------------
    With `chunk`, partition the E-step into blocks of `X`, transfer each `X[s:e]`
    to `device`, and accumulate responsibilities and sufficient statistics `(N_k, S_k)`.
    This supports large datasets under constrained VRAM.

    Parameters
    ----------
    n_components : int
        Number of mixture components K.
    d : int | None, default=None
        Feature dimension D; None infers it from `X.shape[1]` during `fit(X)`.
    device : str | torch.device, default="cuda"
        Computation device; sufficient statistics are aggregated here even with chunks.
    random_state : int | None, default=42
        Initialization seed for a fixed CPU Generator.
    tol : float, default=1e-4
        Converge when nonnegative relative improvement is below tol, or nonnegative
        absolute improvement is below 1e-6. Likelihood decreases stop separately.
    max_iter : int, default=200
        Maximum number of EM iterations.
    init : {"kmeans++", "random"}, default="kmeans++"
        Initialization method; `kmeans++` selects seeds using cosine distance `1-cos`.
    kappa_init : float, default=10.0
        Initial κ shared by all components.
    kappa_min : float, default=1e-6
        Positive lower bound for κ, also used for zero-resultant components.
    dtype : torch.dtype, default=torch.float32
        Direction/statistics dtype (float32 or float64); inputs are cast to this dtype.

    Attributes
    ----------
    K : int
        Number of mixture components.
    d : int | None
        Feature dimension, always an int after `fit()`.
    mus : torch.Tensor, shape (K, d)
        L2-normalized component direction vectors.
    kappas : torch.Tensor, shape (K,)
        Component concentrations κ (`>=kappa_min`).
    logpi : torch.Tensor, shape (K,)
        Internal log mixture weights; the posterior uses their `log_softmax` values.
    _logC : torch.Tensor, shape (K,)
        Cached `log C_d(κ_k)`.
    n_iter_ : int
        Number of completed EM iterations.
    lower_bound_ : float
        Total log-likelihood recomputed from final parameters; NaN on legacy restoration.
    converged_ : bool
        Whether nonnegative improvement satisfied a convergence criterion.
    stop_reason_ : str | None
        "tol", "likelihood_decreased", "max_iter", or None before fit/on legacy restoration.
    _fitted : bool
        Fitted-state flag.

    Notes
    -----
    - Inputs are assumed spherical and are L2-normalized row by row, including
      inputs already normalized by the caller.
    - The κ update uses an approximation based on `R̄` rather than an exact Newton solution.
      Approximation error can arise at high dimension/concentration; replace the
      κ update if required.
    - Chunked initialization (`_init_params`) also uses a subsample to avoid large transfers.
    - To minimize CPU/GPU transfers, use `chunk=None` with `X` already on device.
      Under constrained VRAM, use chunks while keeping `X` on CPU.
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
        if int(n_components) != n_components:
            raise ValueError("n_components must be an integer")
        if d is not None and (int(d) != d or d < 2):
            raise ValueError("d must be an integer >= 2")
        if int(max_iter) != max_iter or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        if not math.isfinite(tol) or tol < 0:
            raise ValueError("tol must be finite and nonnegative")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")

        if init not in ("kmeans++", "random"):
            raise ValueError("init must be 'kmeans++' or 'random'")

        if not math.isfinite(kappa_min) or kappa_min <= 0:
            raise ValueError("kappa_min must be finite and > 0")
        if kappa_min < torch.finfo(dtype).tiny * torch.finfo(dtype).eps:
            raise ValueError("kappa_min must be representable as a positive value in dtype")
        if not math.isfinite(kappa_init) or kappa_init < kappa_min:
            raise ValueError("kappa_init must be finite and >= kappa_min")
        if kappa_init > torch.finfo(dtype).max:
            raise ValueError("kappa_init must be representable in dtype")

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
        self.converged_: bool = False
        self.stop_reason_: Optional[str] = None

        # Keep RNG on CPU, independent of the computation device.
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

        if self._logC.numel() != K or self._logC.device != dev or self._logC.dtype != torch.float64:
            self._logC = torch.empty(K, device=dev, dtype=torch.float64)

    @torch.no_grad()
    def _allocate_buffers(self) -> None:
        assert self.d is not None, "d is not initialized"
        self._ensure_buffers_device_and_shape()

    @torch.no_grad()
    def _refresh_logC(self) -> None:
        assert self.d is not None
        self._logC.copy_(vmf_logC(int(self.d), self.kappas.to(torch.float64)))

    def _validate_X(self, X: torch.Tensor) -> None:
        """Validate shape without making a full-size converted copy."""
        if X.ndim != 2 or X.size(0) == 0 or X.size(1) < 2:
            raise ValueError(f"X must be nonempty (N, d) with d >= 2, got {tuple(X.shape)}")
        if self.d is not None and X.size(1) != self.d:
            raise ValueError(f"X dim mismatch: expected {self.d}, got {X.size(1)}")
        if X.is_complex():
            raise ValueError("X must be real")

    def _normalize_block(self, X: torch.Tensor) -> torch.Tensor:
        """Reject invalid rows and normalize without overflow or epsilon clipping."""
        xb = X.to(self.device, dtype=self.dtype)
        if not torch.isfinite(xb).all():
            raise ValueError("X contains NaN/Inf or overflows the model dtype")
        scale = xb.abs().amax(dim=1, keepdim=True)
        if (scale == 0).any():
            raise ValueError("X contains a zero-norm row in the model dtype")
        xb = xb / scale
        return xb / torch.linalg.vector_norm(xb, dim=1, keepdim=True)

    def _logpost(self, xb: torch.Tensor) -> torch.Tensor:
        """Evaluate component log densities in float64 before cancellation."""
        dot = (xb @ self.mus.T).to(torch.float64).clamp(-1.0, 1.0)
        return (
            dot * self.kappas.to(torch.float64).unsqueeze(0)
            + self._logC.unsqueeze(0)
            + self.logpi.to(torch.float64).log_softmax(dim=0).unsqueeze(0)
        )

    # ------------------------- initialization -------------------------
    @torch.no_grad()
    def _init_params(self, X: torch.Tensor, *, chunk: Optional[int] = None) -> None:
        """Initialize parameters.

        Notes
        -----
        If ``chunk`` is provided and ``X`` is very large, initialization uses a
        random subset to avoid moving the full dataset to GPU.
        """
        chunk = self._validate_chunk(chunk)
        self._validate_X(X)

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

        Xn = self._normalize_block(X_src)
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
                # Sampling always uses the CPU generator, including CUDA fits.
                idxk = int(torch.multinomial(probs.cpu(), 1, generator=self._g).item())
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
        self._validate_X(X)
        N = int(X.size(0))
        K = self.K

        gam = torch.empty(N, K, device=self.device, dtype=self.dtype) if return_gamma else None
        Nk = torch.zeros(K, device=self.device, dtype=self.dtype)
        Sk = torch.zeros(K, self.d, device=self.device, dtype=self.dtype)

        lb_total = 0.0

        def block(s: int, e: int) -> float:
            xb = self._normalize_block(X[s:e])
            logpost = self._logpost(xb)

            gb = torch.softmax(logpost, dim=1).to(self.dtype)  # (b,K)
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
        # Preserve true zeros and tiny positive masses; epsilon must not turn an
        # empty component into a maximally concentrated one.
        Nk = Nk.to(self.device, dtype=torch.float64)
        Sk = Sk.to(self.device, dtype=torch.float64)
        if not torch.isfinite(Nk).all() or not torch.isfinite(Sk).all():
            raise FloatingPointError("Non-finite vMF sufficient statistics")
        if (Nk < 0).any() or Nk.sum() <= 0:
            raise ValueError("Component masses must be nonnegative with positive total")

        pi = Nk / Nk.sum()
        self.logpi.copy_(torch.log(pi.clamp_min(1e-20)))

        Sk_scale = Sk.abs().amax(dim=1)
        scaled = Sk / torch.where(Sk_scale > 0, Sk_scale, 1.0).unsqueeze(1)
        scaled_norm = torch.linalg.vector_norm(scaled, dim=1)
        Sk_norm = Sk_scale * scaled_norm
        occupied = Nk > 0
        directed = occupied & (Sk_scale > 0)
        self.mus[directed] = (scaled[directed] / scaled_norm[directed, None]).to(self.dtype)

        # Keep the existing closed-form approximation and upper resultant guard.
        Rbar = (Sk_norm[occupied] / Nk[occupied]).clamp(0.0, 1.0 - 1e-6)
        Df = float(self.d)
        kappa = (Rbar * (Df - Rbar**2)) / (1.0 - Rbar**2 + eps)
        self.kappas[occupied] = kappa.clamp_min(self.kappa_min).to(self.dtype)
        # Zero resultant: keep the previous unit direction, kappa = kappa_min.
        # Zero mass: keep both direction and concentration; retain the weight floor.
        self._refresh_logC()

    # ------------------------- Public API -------------------------
    @torch.no_grad()
    def fit(self, X: torch.Tensor, *, chunk: Optional[int] = None) -> "VMFMixture":
        r"""
        Fit the vMF mixture model by EM.

        Parameters
        ----------
        X : torch.Tensor, shape (N, d)
            Input features with `N` samples and dimension `d`.
            Values must be finite; rows are L2-normalized internally.
            Empty data and zero vectors raise ValueError.
        chunk : int | None, default=None
            E-step chunk size; None processes all rows together.
            An int transfers `chunk` rows at a time to device, computes responsibilities,
            and accumulates sufficient statistics `(N_k, S_k)` for streaming training.

        Returns
        -------
        self : VMFMixture
            Fitted instance for method chaining.

        Notes
        -----
        - If `d` is None, infer it from `X.shape[1]` and keep it fixed afterward.
        - Convergence uses improvement in the lower bound (E-step `logsumexp` sum).
        """
        chunk = self._validate_chunk(chunk)
        self._validate_X(X)
        self._fitted = False
        self.n_iter_ = 0
        self.lower_bound_ = -float("inf")
        self.converged_ = False
        self.stop_reason_ = None
        self._init_params(X, chunk=chunk)
        _, lb, Nk, Sk = self._e_step_chunk(X, chunk, return_gamma=False)
        if not math.isfinite(lb):
            raise FloatingPointError("Non-finite initial vMF log-likelihood")

        for t in range(self.max_iter):
            self._m_step_from_stats(Nk, Sk)
            _, updated_lb, Nk, Sk = self._e_step_chunk(X, chunk, return_gamma=False)
            if not math.isfinite(updated_lb):
                raise FloatingPointError("Non-finite vMF log-likelihood after M-step")
            self.n_iter_ = t + 1
            self.lower_bound_ = updated_lb
            improvement = updated_lb - lb
            if improvement < 0:
                self.stop_reason_ = "likelihood_decreased"
                break
            if improvement / (abs(lb) + 1e-12) < self.tol or improvement < 1e-6:
                self.converged_ = True
                self.stop_reason_ = "tol"
                break
            lb = updated_lb
        else:
            self.stop_reason_ = "max_iter"

        self._fitted = True
        return self

    @torch.no_grad()
    def predict_proba(self, X: torch.Tensor, *, chunk: Optional[int] = None) -> torch.Tensor:
        r"""
        Compute posterior responsibilities γ_{ik} for each sample.

        Parameters
        ----------
        X : torch.Tensor, shape (N, d)
            Input features with the same dimension d as `fit()`.
            Rows are L2-normalized internally.
        chunk : int | None, default=None
            Chunking as in the E-step, reducing VRAM use for large `X`.

        Returns
        -------
        gamma : torch.Tensor, shape (N, K)
            Responsibilities `γ_{ik}` for sample i/component k; row-wise softmax sums to one.

        Raises
        ------
        RuntimeError
            The model is not fitted.
        ValueError
            Invalid input shape or `X.shape[1] != d`.
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
            Input features.
        chunk : int | None, default=None
            As in `predict_proba`.

        Returns
        -------
        labels : torch.Tensor, shape (N,)
            Assigned labels (0..K-1) from `argmax_k γ_{ik}`.
        """
        return self.predict_proba(X, chunk=chunk).argmax(dim=1)

    @torch.inference_mode()
    def loglik(self, X: torch.Tensor, *, chunk: Optional[int] = None, average: bool = False) -> float:
        r"""
        Compute (approximate) log-likelihood under the fitted model.

        Parameters
        ----------
        X : torch.Tensor, shape (N, d)
            Input features, L2-normalized internally.
        chunk : int | None, default=None
            Chunked computation for large `X`.
        average : bool, default=False
            Return mean log-likelihood per sample when true.

        Returns
        -------
        ll : float
            Total or per-sample mean log-likelihood,
            `Σ_i log Σ_k π_k C_d(κ_k) exp(κ_k μ_k^T x_i)`.

        Notes
        -----
        - Treat `logpi` as mixture weights after `log_softmax` for numerical stability.
        - Use float64 `_logC` internally.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted")

        chunk = self._validate_chunk(chunk)
        self._validate_X(X)
        N = int(X.size(0))

        def block(s: int, e: int) -> float:
            xb = self._normalize_block(X[s:e])
            return float(torch.logsumexp(self._logpost(xb), dim=1).sum().item())

        if chunk is None or N <= chunk:
            total = block(0, N)
        else:
            total = 0.0
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                total = total + block(s, e)

        return (total / N) if average else total

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
            Input features.
        chunk : int | None, default=None
            Chunked computation as in `loglik`.

        Returns
        -------
        bic : float
            BIC value; lower is better.

        Notes
        -----
        - `p = K*d + (K-1)` counts K*(d-1) direction, K concentration,
          and K-1 mixture-weight parameters.
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
            Dictionary of CPU tensors and metadata containing:
            - K, d, device, dtype
            - mus, kappas, logpi, _logC (all CPU clones)
            - numerics_version, n_iter_, lower_bound_, converged_, stop_reason_, _fitted
            - random_state, rng_state
            - tol, max_iter, init, kappa_init, kappa_min

        Notes
        -----
        - Provides a class-specific persistence format rather than `torch.nn.Module.state_dict()`.
        - The returned dictionary can be saved directly with `torch.save()`.
        """
        return {
            "numerics_version": 2,
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
            "converged_": self.converged_,
            "stop_reason_": self.stop_reason_,
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
            Destination path for `torch.save(self.state_dict_vmf(), path)`.

        Notes
        -----
        - Saving all tensors on CPU reduces dependence on GPU availability.
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
            Path to a file written by `save()`.
        map_location : str | torch.device | None, default=None
            Restored computation device; None uses the saved device. Use "cpu" for CPU loading.

        Returns
        -------
        model : VMFMixture
            Restored model with `mus/kappas/logpi/_logC` and `_fitted=True`.

        Raises
        ------
        RuntimeError
            Invalid saved data, for example when `d=None` prevents buffer allocation.

        Notes
        -----
        - Explicit map_location takes precedence over the saved device; use the saved dtype.
        - Rebuild `_logC` with current numerics; do not reuse legacy likelihood/convergence data.
        - Restore the CPU Generator state when `rng_state` was saved.
        """
        # The serialized tensors and generator state are CPU data. Move only
        # model buffers to the requested computation device after deserialization.
        sd = torch.load(path, map_location="cpu", weights_only=True)
        K = int(sd["K"])
        d = sd["d"]
        device = map_location if map_location is not None else sd.get("device", "cpu")
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
        norms = torch.linalg.vector_norm(obj.mus.to(torch.float64), dim=1)
        if not torch.isfinite(obj.mus).all() or not torch.allclose(
            norms, torch.ones_like(norms), rtol=1e-5, atol=1e-5
        ):
            raise ValueError("Saved vMF directions must be finite unit vectors; refit invalid models")
        if not torch.isfinite(obj.kappas).all() or (obj.kappas < obj.kappa_min).any():
            raise ValueError("Saved vMF concentrations must be finite and >= kappa_min")
        if not torch.isfinite(obj.logpi).all():
            raise ValueError("Saved vMF log weights must be finite")
        obj._refresh_logC()
        obj._nu = 0.5 * float(obj.d) - 1.0
        obj.n_iter_ = int(sd.get("n_iter_", 0))
        current_numerics = sd.get("numerics_version") == 2
        obj.lower_bound_ = float(sd.get("lower_bound_", float("nan"))) if current_numerics else float("nan")
        obj.converged_ = bool(sd.get("converged_", False)) if current_numerics else False
        obj.stop_reason_ = sd.get("stop_reason_", None) if current_numerics else None
        obj._fitted = bool(sd.get("_fitted", True))

        rng_state = sd.get("rng_state", None)
        if rng_state is not None:
            obj._g.set_state(rng_state.cpu())
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
        Callable returning `VMFMixture`, usually the class itself.
        For example: `elbow_vmf(VMFMixture, X, ...)`.
    X : torch.Tensor, shape (N, d)
        Input features.
    device : str, default="cuda"
        Training device for each K.
    k_max : int, default=50
        Sweep K from 1..k_max. Must be an integer >= 3.
    chunk : int | None, default=None
        None transfers `X` to device once for reuse across K, trading VRAM for speed.
        With an int, `X` may remain on CPU while fit/predict stream chunked E-steps.
    verbose : bool, default=True
        Print the score for each K.
    random_state : int, default=42
        Initialization seed for each K.
    criterion : {"bic", "nll"}, default="bic"
        - "bic": evaluate `vmf.bic(X)` (lower is better).
        - "nll": evaluate `-vmf.loglik(X, average=True)` (lower is better).

    Returns
    -------
    k_list : list[int]
        Evaluated K values (1..k_max).
    scores : list[float]
        Score for each K, depending on criterion; lower is better.
    K_elbow : int
        Heuristic K from curvature-based elbow estimation, not the minimum-score K.
    idx_elbow : int
        Elbow index in `k_list`.
    curvature : float
        Scalar curvature used for elbow estimation.

    Notes
    -----
    - Both criteria pass their lower-is-better scores directly to
      `find_elbow_curvature`; returned scores are unchanged.
    - The helper replaces score increases with the cumulative minimum before
      estimating curvature. Inspect the original curve because this can hide
      nonmonotonic behavior. A flat curve selects K=2 with zero curvature and
      does not provide evidence for a preferred K.
    - Minimum-score and elbow estimates can select different K; use the
      comparison protocol chosen for the application.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if isinstance(k_max, bool) or not isinstance(k_max, (int, np.integer)) or k_max < 3:
        raise ValueError("k_max must be an integer >= 3")
    if criterion not in ("bic", "nll"):
        raise ValueError("criterion must be 'bic' or 'nll'")

    if chunk is None:
        X_input = X.to(device, non_blocking=True)
    else:
        X_input = X

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
    K, idx, kappa = find_elbow_curvature(k_list, scores)

    if verbose:
        best_idx = int(min(range(len(scores)), key=lambda i: scores[i]))
        print(f"Optimal k (curvature): {K}  |  Best-by-{criterion}: k={k_list[best_idx]}, score={scores[best_idx]:.6f}")

    return k_list, scores, K, idx, kappa
