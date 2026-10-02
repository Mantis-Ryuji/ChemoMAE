from __future__ import annotations

import os
import gc
import math
from numbers import Integral, Real
from typing import Optional, Tuple, Union, Callable, List, Literal

import torch
import torch.nn as nn

from .ops import l2_normalize_rows, cosine_dissimilarity

__all__ = ["CosineKMeans", "elbow_ckmeans"]


def _validated_config(
    n_components: object, tol: object, max_iter: object, random_state: object
) -> tuple[int, float, int, int | None]:
    for name, value in (("n_components", n_components), ("max_iter", max_iter)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(tol, bool) or not isinstance(tol, Real) or not math.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")
    if random_state is not None and (
        isinstance(random_state, bool) or not isinstance(random_state, Integral)
        or not -(2**63) <= random_state <= 2**64 - 1
    ):
        raise ValueError("random_state must be an integer Torch seed or None")
    return int(n_components), float(tol), int(max_iter), int(random_state) if random_state is not None else None


def _validate_features(X: torch.Tensor) -> None:
    if not isinstance(X, torch.Tensor):
        raise TypeError("X must be a Torch tensor")
    if X.ndim != 2 or X.size(0) == 0 or X.size(1) == 0:
        raise ValueError("X must be a nonempty two-dimensional feature matrix")
    if X.layout != torch.strided or X.is_complex() or X.dtype == torch.bool:
        raise ValueError("X must be a dense real numeric tensor")
    if not torch.isfinite(X).all():
        raise ValueError("X contains NaN/Inf")


def _validate_chunk(chunk: int | None) -> None:
    if chunk is not None and (
        isinstance(chunk, bool) or not isinstance(chunk, Integral) or chunk <= 0
    ):
        raise ValueError("chunk must be a positive integer or None")


def _validated_diagnostics(
    inertia: object, n_iter: object, converged: object, stop_reason: object, max_iter: int
) -> tuple[float, int, bool, Literal["tolerance", "max_iter"]]:
    if isinstance(inertia, bool) or not isinstance(inertia, Real) or not math.isfinite(inertia):
        raise ValueError("inertia_ must be finite")
    if isinstance(n_iter, bool) or not isinstance(n_iter, Integral) or not 1 <= n_iter <= max_iter:
        raise ValueError("n_iter_ must be an integer from one through max_iter")
    if type(converged) is not bool:
        raise ValueError("converged_ must be boolean")
    if stop_reason not in ("tolerance", "max_iter"):
        raise ValueError("stop_reason_ must be 'tolerance' or 'max_iter'")
    if converged != (stop_reason == "tolerance") or (not converged and n_iter != max_iter):
        raise ValueError("convergence diagnostics are inconsistent")
    reason: Literal["tolerance", "max_iter"] = "tolerance" if converged else "max_iter"
    return float(inertia), int(n_iter), converged, reason


def _validated_centers(centers: object, n_components: int, latent_dim: object) -> torch.Tensor:
    if isinstance(latent_dim, bool) or not isinstance(latent_dim, Integral) or latent_dim <= 0:
        raise ValueError("latent_dim must be a positive integer")
    if not isinstance(centers, torch.Tensor):
        raise ValueError("centroids must be a Torch tensor")
    if (centers.layout != torch.strided or centers.dtype != torch.float32
            or centers.ndim != 2 or centers.shape != (n_components, latent_dim)):
        raise ValueError("centroids must be a nonempty FP32 matrix matching K and latent_dim")
    if not torch.isfinite(centers).all():
        raise ValueError("centroids contain NaN/Inf")
    return centers


class CosineKMeans(nn.Module):
    r"""
    Cosine (hyperspherical) K-Means with k-means++ init and optional streaming.

    Parameters
    ----------
    n_components : int, default=8
        Positive cluster count K.
    tol : float, default=1e-4
        Finite nonnegative tolerance. Stop when successive pre-update objectives
        change relatively by less than tol or absolutely by less than tol*1e-3.
        Zero disables tolerance-based stopping.
    max_iter : int, default=500
        Positive maximum number of completed centroid updates.
    device : str | torch.device, default="cuda"
        Computation device.
    random_state : int | None, default=42
        Seed for the instance-owned CPU initialization generator. Its stream
        advances across fits; None leaves the generator's default seed unchanged.

    Attributes
    ----------
    centroids : torch.Tensor, shape (K, D)
        FP32 fitted centers registered as a buffer, using the existing row
        normalization convention. Zero sums remain zero.
    latent_dim : int or None
        Feature dimension, inferred during fitting.
    inertia_ : float
        Mean nearest-center cosine dissimilarity against the final stored
        centers, after their final update and normalization. This is not SSE.
    n_iter_ : int
        Completed centroid updates; zero before fitting.
    converged_ : bool
        True only when the tolerance condition stopped the most recent fit.
    stop_reason_ : {"tolerance", "max_iter"} or None
        Most recent fit's stop reason; None before fitting.

    Notes
    -----
    Inputs are row-normalized and computed in FP32. Assignments maximize cosine
    similarity; centroid means are normalized after updates. Empty classes use
    the samples farthest from their currently nearest centers.
    Row normalization uses eps=1e-6; sufficiently small or zero vectors can
    retain a norm below one. Persistence validates values without normalizing.

    A positive chunk streams CPU features to CUDA, but full labels and maximum
    similarities remain resident on the compute device. CPU fitting retains the
    full feature matrix. Final-objective evaluation adds one assignment pass.

    save_centroids/load_centroids preserve fitted centers and diagnostics
    exactly without renormalization. They restore prediction state, not an
    in-progress fit or the advanced initialization-generator state.
    """
    def __init__(
        self,
        n_components: int = 8,
        tol: float = 1e-4,
        max_iter: int = 500,
        device: Union[str, torch.device] = "cuda",
        random_state: Optional[int] = 42
    ) -> None:
        super().__init__()
        self.n_components, self.tol, self.max_iter, self.random_state = _validated_config(
            n_components, tol, max_iter, random_state
        )
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available; select device='cpu'")

        # Register an empty buffer until fitting determines the feature dimension.
        self.register_buffer("centroids", torch.empty(0, 0, device=self.device, dtype=torch.float32))

        self._generator = torch.Generator(device="cpu")
        if random_state is not None:
            self._generator.manual_seed(int(random_state))

        self.latent_dim: Optional[int] = None
        # The objective is mean cosine dissimilarity rather than SSE.
        self.inertia_: float = float("inf")
        self.n_iter_: int = 0
        self.converged_: bool = False
        self.stop_reason_: Literal["tolerance", "max_iter"] | None = None
        self._fitted: bool = False

    # ----------------------------- init (k-means++) -----------------------------
    @torch.no_grad()
    def _init_centroids_kmeanspp(self, Xn: torch.Tensor) -> torch.Tensor:
        """Full-device k-means++ (Xn: L2-normalized, on device, preferably FP32)."""
        N = int(Xn.size(0))
        if self.n_components > N:
            raise ValueError(f"n_components ({self.n_components}) must be <= N ({N}).")
        K, d = self.n_components, int(Xn.size(1))
        C = torch.empty(K, d, device=Xn.device, dtype=Xn.dtype)

        # First center.
        idx0 = torch.randint(0, N, (1,), generator=self._generator)
        C[0] = Xn[idx0.to(Xn.device)]

        # Subsequent centers.
        dmin = cosine_dissimilarity(Xn, C[0:1]).squeeze(1).clamp_min_(1e-12)
        probs = (dmin / (dmin.sum() + 1e-12)).clamp_min_(0)

        for k in range(1, K):
            idx_cpu = torch.multinomial(probs.detach().cpu(), num_samples=1, generator=self._generator)
            idx = idx_cpu.to(Xn.device)
            C[k] = Xn[idx]
            dk = cosine_dissimilarity(Xn, C[k:k + 1]).squeeze(1)
            dmin = torch.minimum(dmin, dk).clamp_min_(1e-12)
            probs = (dmin / (dmin.sum() + 1e-12)).clamp_min_(0)

        return l2_normalize_rows(C)

    @torch.no_grad()
    def _init_centroids_kmeanspp_stream(self, X_cpu: torch.Tensor, chunk: int) -> torch.Tensor:
        """Streaming k-means++: keep X on CPU and transfer chunks to device."""
        N = int(X_cpu.size(0))
        if self.n_components > N:
            raise ValueError(f"n_components ({self.n_components}) must be <= N ({N}).")
        K, d = self.n_components, int(X_cpu.size(1))
        C = torch.empty(K, d, device=self.device, dtype=torch.float32)

        idx0 = torch.randint(0, N, (1,), generator=self._generator)
        C[0] = l2_normalize_rows(X_cpu[idx0].to(self.device, dtype=torch.float32))[0]

        dmin = torch.full((N,), float("inf"), dtype=torch.float32)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            x = l2_normalize_rows(X_cpu[s:e].to(self.device, dtype=torch.float32))
            d = cosine_dissimilarity(x, C[0:1]).squeeze(1).float().cpu()
            dmin[s:e] = torch.minimum(dmin[s:e], d)
            del x, d

        for k in range(1, K):
            w = dmin.clamp_min_(1e-12)
            probs = w / (w.sum() + 1e-12)
            idx_cpu = torch.multinomial(probs, num_samples=1, generator=self._generator)
            C[k] = l2_normalize_rows(X_cpu[idx_cpu].to(self.device, dtype=torch.float32))[0]

            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                x = l2_normalize_rows(X_cpu[s:e].to(self.device, dtype=torch.float32))
                d = cosine_dissimilarity(x, C[k:k + 1]).squeeze(1).float().cpu()
                dmin[s:e] = torch.minimum(dmin[s:e], d)
                del x, d

        return l2_normalize_rows(C)

    # ----------------------------- E / M (with streaming variants) -----------------------------
    @torch.no_grad()
    def _assign_in_chunks_cpu(self, X_cpu: torch.Tensor, C: torch.Tensor, chunk: int):
        N = int(X_cpu.size(0))
        device, dtype = C.device, C.dtype
        labels = torch.empty(N, dtype=torch.long, device=device)
        max_sim = torch.empty(N, dtype=dtype, device=device)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            x = l2_normalize_rows(X_cpu[s:e].to(device, dtype=dtype, non_blocking=True))
            sim = x @ C.T
            m, l = sim.max(dim=1)
            labels[s:e] = l
            max_sim[s:e] = m
            del x, sim
        return labels, max_sim

    @torch.no_grad()
    def _update_centroids_in_chunks_cpu(self, X_cpu: torch.Tensor, labels: torch.Tensor, K: int, chunk: int):
        device = labels.device
        d = self.latent_dim if self.latent_dim is not None else int(X_cpu.size(1))
        C_new = torch.zeros(K, d, device=device, dtype=torch.float32)
        counts = torch.zeros(K, device=device, dtype=torch.float32)
        N = int(X_cpu.size(0))
        ones = None
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            x = l2_normalize_rows(X_cpu[s:e].to(device, dtype=torch.float32, non_blocking=True))
            l = labels[s:e]
            C_new.index_add_(0, l, x)  # fp32 accumulate
            if (ones is None) or (ones.numel() != (e - s)):
                ones = torch.ones((e - s,), device=device, dtype=torch.float32)
            counts.scatter_add_(0, l, ones)
            del x, l
        non_empty = counts > 0
        if non_empty.any():
            C_new[non_empty] = l2_normalize_rows(C_new[non_empty] / counts[non_empty].unsqueeze(1))
        return C_new, counts

    # ----------------------------- Fit / Predict -----------------------------
    @torch.no_grad()
    def fit(self, X: torch.Tensor, chunk: Optional[int] = None) -> "CosineKMeans":
        r"""
        Fit spherical centers and record final-objective and stop diagnostics.

        Parameters
        ----------
        X : torch.Tensor, shape (N, D)
            Nonempty finite real features, converted to FP32 internally.
        chunk : int | None, default=None
            Positive streaming chunk size on CUDA. None uses the full device;
            CPU fitting uses the full matrix even when a chunk is supplied.

        Returns
        -------
        self : CosineKMeans
            Fitted instance. n_iter_ counts completed updates; inertia_ is
            evaluated against the final stored centers in an extra pass.

        Raises
        ------
        ValueError
            Invalid shape, nonfinite features, invalid chunk, or K greater than N.
        """
        _validate_features(X)
        _validate_chunk(chunk)
        _validated_config(self.n_components, self.tol, self.max_iter, self.random_state)
        if self.n_components > X.size(0):
            raise ValueError(f"n_components ({self.n_components}) must be <= N ({X.size(0)})")
        # Infer the feature dimension; an interrupted new fit is not fitted state.
        self.latent_dim = int(X.size(1))
        self._fitted = False
        self.n_iter_ = 0
        self.converged_ = False
        self.stop_reason_ = None
        self.inertia_ = float("inf")

        stream = (chunk is not None) and (self.device.type == "cuda")

        # Initialize in FP32 using the existing k-means++ policy.
        if stream:
            X_cpu = X.to("cpu", dtype=torch.float32)
            if not torch.isfinite(X_cpu).all():
                raise ValueError("X cannot be represented as finite FP32 features")
            C = self._init_centroids_kmeanspp_stream(X_cpu, chunk)
        else:
            X_fp32 = X.to(self.device, dtype=torch.float32)
            if not torch.isfinite(X_fp32).all():
                raise ValueError("X cannot be represented as finite FP32 features")
            Xn = l2_normalize_rows(X_fp32)
            del X_fp32
            C = self._init_centroids_kmeanspp(Xn)

        prev = None
        for iteration in range(1, self.max_iter + 1):
            # E-step
            if stream:
                labels, max_sim = self._assign_in_chunks_cpu(X_cpu, C, chunk)
                mean_J = (1.0 - max_sim).mean().item()
            else:
                sim = Xn @ C.T
                labels = sim.argmax(dim=1)
                max_sim = sim.gather(1, labels.unsqueeze(1)).squeeze(1)
                mean_J = (1.0 - max_sim).mean().item()

            # Accumulate centroid updates in FP32.
            if stream:
                C_new, counts = self._update_centroids_in_chunks_cpu(X_cpu, labels, self.n_components, chunk)
            else:
                counts = torch.bincount(labels, minlength=self.n_components).to(torch.float32)
                C_new = torch.zeros_like(C, dtype=torch.float32)
                C_new.index_add_(0, labels, Xn.to(torch.float32))
                non_empty = counts > 0
                if non_empty.any():
                    C_new[non_empty] = l2_normalize_rows(C_new[non_empty] / counts[non_empty].unsqueeze(1))

            # Refill empty classes with the farthest currently assigned samples.
            non_empty = counts > 0
            if (~non_empty).any():
                num_empty = int((~non_empty).sum().item())
                nearest_d = 1.0 - max_sim  # distance
                far_idx = torch.argsort(nearest_d, descending=True)[:num_empty]
                empty_ids = (~non_empty).nonzero(as_tuple=False).squeeze(1)
                if stream:
                    xfar = l2_normalize_rows(X_cpu[far_idx.cpu()].to(self.device, dtype=torch.float32))
                    C_new[empty_ids] = xfar
                else:
                    C_new[empty_ids] = l2_normalize_rows(Xn[far_idx].to(torch.float32))

            self.n_iter_ = iteration
            # Preserve the existing pre-update objective stopping criterion.
            if prev is not None:
                rel = abs(prev - mean_J) / (abs(prev) + 1e-12)
                if (rel < self.tol) or (abs(prev - mean_J) < self.tol * 1e-3):
                    C = C_new
                    prev = mean_J
                    self.converged_ = True
                    self.stop_reason_ = "tolerance"
                    break
            C = C_new
            prev = mean_J
        if not self.converged_:
            self.stop_reason_ = "max_iter"

        # Normalize and update the registered prediction buffer.
        C = l2_normalize_rows(C).to(self.device, dtype=torch.float32)
        if not torch.isfinite(C).all():
            raise RuntimeError("Fitting produced nonfinite centers")
        if self.centroids.shape != C.shape:
            self.centroids.resize_(C.shape)
        self.centroids.copy_(C)

        # Evaluate the reported objective against the exact prediction buffer,
        # including the last M-step and its final normalization.
        if stream:
            _, final_max_sim = self._assign_in_chunks_cpu(X_cpu, self.centroids, chunk)
        else:
            del sim
            final_max_sim = (Xn @ self.centroids.T).max(dim=1).values
        self.inertia_ = float((1.0 - final_max_sim).mean().item())
        if not math.isfinite(self.inertia_):
            raise RuntimeError("Final-center objective is nonfinite")
        self._fitted = True

        gc.collect()
        if stream and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    @torch.no_grad()
    def fit_predict(self, X: torch.Tensor, chunk: Optional[int] = None) -> torch.Tensor:
        r"""
        Fit on `X` and return labels.

        Parameters
        ----------
        X : torch.Tensor, shape (N, D)
            Input features.
        chunk : int | None, default=None
            Streaming chunk size.

        Returns
        -------
        labels : torch.Tensor, shape (N,), dtype=torch.long
            Assigned cluster IDs.
        """
        self.fit(X, chunk=chunk)
        return self.predict(X, chunk=chunk)

    @torch.no_grad()
    def predict(
        self,
        X: torch.Tensor,
        return_dist: bool = False,
        chunk: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""
        Predict cluster labels (and optionally distances) for `X`.

        Parameters
        ----------
        X : torch.Tensor, shape (N, D)
            Nonempty finite real feature matrix.
        return_dist : bool, default=False
            Also return the full (N, K) cosine-dissimilarity matrix.
        chunk : int | None, default=None
            Positive CUDA streaming chunk size; None uses the full device.

        Returns
        -------
        labels : torch.Tensor, shape (N,), dtype=torch.long
            Nearest-center IDs on the centers' device.
        dist : torch.Tensor, shape (N, K), optional
            Cosine dissimilarities; allocated in full even with streaming.

        Raises
        ------
        RuntimeError
            Fitted centers are unavailable or invalid.
        ValueError
            Invalid features or chunk, or a different feature dimension.
        """
        if (
            not self._fitted
            or self.centroids is None
            or self.centroids.numel() == 0
            or not torch.isfinite(self.centroids).all()
        ):
            raise RuntimeError("Centroids are not initialized. Call fit() or load_centroids() first.")

        _validate_features(X)
        _validate_chunk(chunk)
        if self.latent_dim is None:
            raise RuntimeError("latent_dim is undefined. Call fit() or load_centroids() first.")
        if int(X.size(1)) != self.latent_dim:
            raise ValueError(f"X dim mismatch: expected {self.latent_dim}, got {int(X.size(1))}")

        stream = (chunk is not None) and (self.centroids.device.type == "cuda")
        if stream:
            X_cpu = X.to("cpu", dtype=torch.float32)
            if not torch.isfinite(X_cpu).all():
                raise ValueError("X cannot be represented as finite FP32 features")
            N = int(X_cpu.size(0))
            labels = torch.empty(N, dtype=torch.long, device=self.centroids.device)
            dist_all = None
            if return_dist:
                dist_all = torch.empty(N, self.centroids.size(0), dtype=torch.float32, device=self.centroids.device)
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                x = l2_normalize_rows(X_cpu[s:e].to(self.centroids.device, dtype=torch.float32, non_blocking=True))
                sim = x @ self.centroids.T
                l = sim.argmax(dim=1)
                labels[s:e] = l
                if return_dist:
                    dist_all[s:e] = 1.0 - sim
                del x, sim
            return (labels, dist_all) if return_dist else labels
        else:
            X_fp32 = X.to(self.centroids.device, dtype=torch.float32)
            if not torch.isfinite(X_fp32).all():
                raise ValueError("X cannot be represented as finite FP32 features")
            Xn = l2_normalize_rows(X_fp32)
            del X_fp32
            sim = Xn @ self.centroids.T
            labels = sim.argmax(dim=1)
            if return_dist:
                return labels, (1.0 - sim)
            return labels

    # ----------------------------- Centroids I/O  -----------------------------
    @torch.no_grad()
    def save_centroids(self, path: str | bytes | "os.PathLike[str]") -> None:
        r"""
        Save exact fitted centers, constructor configuration and diagnostics.

        Parameters
        ----------
        path : str | PathLike
            Destination for the versioned Torch fitted-state file.

        Raises
        ------
        RuntimeError
            No fitted state is available.
        ValueError
            Centers, configuration or fitted diagnostics are invalid.

        Notes
        -----
        Centers are copied to CPU without normalization or dtype conversion.
        This file restores fitted prediction state; it excludes device choice
        and the advanced initialization-generator state, and cannot resume fit.
        """
        if not self._fitted:
            raise RuntimeError("Model is not fitted; no centroids to save.")
        k, tol, max_iter, seed = _validated_config(
            self.n_components, self.tol, self.max_iter, self.random_state
        )
        centers = _validated_centers(self.centroids, k, self.latent_dim)
        inertia, n_iter, converged, reason = _validated_diagnostics(
            self.inertia_, self.n_iter_, self.converged_, self.stop_reason_, max_iter
        )
        payload = {
            "format_version": 1,
            "config": {
                "n_components": k, "tol": tol, "max_iter": max_iter,
                "random_state": seed,
            },
            "centroids": centers.detach().cpu().clone(),
            "latent_dim": self.latent_dim,
            "inertia_": inertia,
            "n_iter_": n_iter,
            "converged_": converged,
            "stop_reason_": reason,
        }
        torch.save(payload, path)

    @torch.no_grad()
    def load_centroids(
        self, path: str | bytes | "os.PathLike[str]", *, strict_k: bool = True
    ) -> "CosineKMeans":
        r"""
        Restore validated fitted prediction state on this instance's device.

        Parameters
        ----------
        path : str | PathLike
            File produced by save_centroids with format_version=1. Historical
            unversioned center-only payloads are rejected explicitly.
        strict_k : bool, default=True
            Require saved K to match the instance's K. False adopts saved K.
            Other saved constructor settings and diagnostics are restored.

        Returns
        -------
        self : CosineKMeans
            Restored instance. Centers are preserved without renormalization.

        Raises
        ------
        ValueError
            Unsupported format, invalid configuration/state, or a K mismatch.

        Notes
        -----
        Validation finishes before fitted attributes change. The initialization
        generator is recreated from the saved random_state; its advanced stream
        position is not stored. Subsequent fitting starts a new fit.
        """
        if type(strict_k) is not bool:
            raise ValueError("strict_k must be boolean")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (not isinstance(payload, dict) or type(payload.get("format_version")) is not int
                or payload["format_version"] != 1):
            raise ValueError("Unsupported fitted-state format; expected format_version=1")
        config = payload.get("config")
        if not isinstance(config, dict):
            raise ValueError("fitted-state config must be a dictionary")
        k, tol, max_iter, seed = _validated_config(
            config.get("n_components"), config.get("tol"),
            config.get("max_iter"), config.get("random_state")
        )
        if "random_state" not in config:
            raise ValueError("fitted-state config is missing random_state")
        C = _validated_centers(payload.get("centroids"), k, payload.get("latent_dim"))
        inertia, n_iter, converged, reason = _validated_diagnostics(
            payload.get("inertia_"), payload.get("n_iter_"),
            payload.get("converged_"), payload.get("stop_reason_"), max_iter
        )
        if strict_k and k != self.n_components:
            raise ValueError(f"n_components mismatch: expected {self.n_components}, file has {k}")
        restored_centers = C.to(self.device).clone()
        restored_generator = torch.Generator(device="cpu")
        if seed is not None:
            restored_generator.manual_seed(seed)
        self.centroids = restored_centers
        self.n_components, self.tol, self.max_iter, self.random_state = k, tol, max_iter, seed
        self._generator = restored_generator
        self.latent_dim = int(C.size(1))
        self.inertia_ = inertia
        self.n_iter_, self.converged_, self.stop_reason_ = n_iter, converged, reason
        self._fitted = True
        return self

# ----------------------------- Model selection (elbow sweep) -----------------------------
@torch.no_grad()
def elbow_ckmeans(
    cluster_module: Callable[..., "CosineKMeans"],
    X: torch.Tensor,
    device: str = "cuda",
    k_max: int = 50,
    chunk: Optional[int] = None,
    verbose: bool = True,
    random_state: int = 42,
) -> Tuple[List[int], List[float], int, int, float]:
    r"""
    Sweep K from 1..k_max for cosine k-means and pick an elbow by curvature.

    Overview
    ----
    - Fit CosineKMeans (`cluster_module`) for `K = 1..k_max` and record
      the objective `inertia = mean(1 - cos(x, c))`.
    - Select a curvature-based elbow with `find_elbow_curvature(k_list, inertias)`.
    - Return `(k_list, inertias, optimal_k, elbow_idx, kappa)`.

    Parameters
    ----------
    cluster_module : Callable[..., CosineKMeans]
        CosineKMeans-compatible constructor, such as `CosineKMeans` itself.
        Must accept `n_components`, `device`, `random_state`, and related arguments.
    X : torch.Tensor, shape (N, D)
        Input features, transferred internally to `device` without a copy if already there.
    device : {"cuda", "cpu"} | torch.device, default="cuda"
        Training device.
    k_max : int, default=50
        Maximum cluster count to evaluate (sweeps 1..k_max).
    chunk : int | None, default=None
        Streaming chunk size. `None` uses the full device; a positive value uses
        CPU-to-GPU streaming to reduce VRAM, delegated to `CosineKMeans.fit(..., chunk=...)`.
    verbose : bool, default=True
        Print `mean_inertia` for each K.
    random_state : int, default=42
        Random seed controlling k-means++ initialization.

    Returns
    -------
    k_list : List[int]
        Evaluated cluster counts (1..k_max).
    inertias : List[float]
        `mean(1 - cos)` for each K; generally decreases as K increases.
    optimal_k : int
        Recommended cluster count selected by curvature (the bend in the curve).
    elbow_idx : int
        Index satisfying `k_list[elbow_idx] == optimal_k`.
    kappa : float
        Implementation-dependent nonnegative curvature at the selected point.
        Larger values indicate a clearer elbow.

    Notes
    -----
    - The inertia objective is `J = mean(1 - cos(x, c))`, rather than SSE.
    - For large N, set `chunk` to reduce VRAM use.
    - Garbage collection and GPU memory cleanup follow training for each K.
    - Elbow estimation delegates to a locally imported `find_elbow_curvature`.

    Examples
    --------
    >>> X = torch.randn(500, 16)  # Features; rows are normalized internally.
    >>> from chemomae.clustering import CosineKMeans
    >>> ks, js, K, idx, kappa = elbow_ckmeans(
    ...     CosineKMeans, X, device="cpu", k_max=6, chunk=128, verbose=False,
    ... )
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D")

    # Avoid transferring all data to GPU when streaming.
    if chunk is None:
        X_input = X.to(device, non_blocking=True)
    else:
        # Keep chunked input on CPU; fit handles the transfers.
        X_input = X

    inertias: List[float] = []
    k_list = list(range(1, k_max + 1))

    for k in k_list:
        ckm = cluster_module(
            n_components=k,
            tol=1e-4,
            max_iter=500,
            device=device,
            random_state=random_state,
        )
        ckm.fit(X_input, chunk=chunk)
        inertias.append(float(ckm.inertia_))
        if verbose:
            print(f"k={k}, mean_inertia={ckm.inertia_:.6f}")

        # Clear GPU memory only when using GPU.
        gc.collect()
        if torch.device(device).type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Import locally to avoid circular dependencies.
    from .ops import find_elbow_curvature
    K, idx, kappa = find_elbow_curvature(k_list, inertias)
    if verbose:
        print(f"Optimal k (curvature): {K}")
    return k_list, inertias, K, idx, kappa
