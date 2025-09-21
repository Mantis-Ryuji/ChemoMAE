from __future__ import annotations
import gc
from typing import Optional, Tuple, Union, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import l2_normalize_rows, cosine_dissimilarity

__all__ = ["CosineKMeans", "elbow_ckmeans"]


class CosineKMeans(nn.Module):
    """Cosine (spherical) K-Means with k-means++ init and optional streaming.

    Args:
        n_clusters: number of clusters K
        latent_dim: feature dimension d (must match X.size(1))
        tol: convergence tolerance on mean objective (relative OR absolute)
        max_iter: maximum number of EM iterations
        device: "cuda" / "cpu" or torch.device
        random_state: int seed for deterministic multinomial
        use_squared_init: if True, use D(x)^2 in k-means++ probabilities
    """

    def __init__(
        self,
        n_clusters: int = 8,
        latent_dim: int = 64,
        tol: float = 1e-3,
        max_iter: int = 500,
        device: Union[str, torch.device] = "cuda",
        random_state: Optional[int] = 42,
        use_squared_init: bool = False,
    ) -> None:
        super().__init__()
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        self.n_clusters = int(n_clusters)
        self.latent_dim = int(latent_dim)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.device = torch.device(device)
        self.random_state = random_state
        self.use_squared_init = bool(use_squared_init)

        self.register_buffer(
            "centroids",
            F.normalize(torch.randn(self.n_clusters, self.latent_dim, device=self.device), dim=1),
        )
        self._generator = torch.Generator(device="cpu")
        if random_state is not None:
            self._generator.manual_seed(int(random_state))

        self.inertia_: float = float("inf")
        self._fitted: bool = False

    # ----------------------------- init (k-means++) -----------------------------
    @torch.no_grad()
    def _init_centroids_kmeanspp(self, Xn: torch.Tensor) -> torch.Tensor:
        """Full-device k-means++ (Xn: L2-normalized, on device)."""
        N = int(Xn.size(0))
        if self.n_clusters > N:
            raise ValueError(f"n_clusters ({self.n_clusters}) must be <= N ({N}).")
        K, d = self.n_clusters, self.latent_dim
        C = torch.empty(K, d, device=Xn.device, dtype=Xn.dtype)

        idx0 = torch.randint(0, N, (1,), generator=self._generator)
        C[0] = Xn[idx0.to(Xn.device)]

        dmin = cosine_dissimilarity(Xn, C[0:1]).squeeze(1).clamp_min_(1e-12)
        w = dmin.square() if self.use_squared_init else dmin
        probs = (w / (w.sum() + 1e-12)).clamp_min_(0)

        for k in range(1, K):
            idx_cpu = torch.multinomial(probs.detach().cpu(), num_samples=1, generator=self._generator)
            idx = idx_cpu.to(Xn.device)
            C[k] = Xn[idx]
            dk = cosine_dissimilarity(Xn, C[k:k + 1]).squeeze(1)
            dmin = torch.minimum(dmin, dk).clamp_min_(1e-12)
            w = dmin.square() if self.use_squared_init else dmin
            probs = (w / (w.sum() + 1e-12)).clamp_min_(0)
        return l2_normalize_rows(C)

    @torch.no_grad()
    def _init_centroids_kmeanspp_stream(self, X_cpu: torch.Tensor, chunk: int) -> torch.Tensor:
        """Streaming k-means++: X stays on CPU; chunks are sent to device."""
        N = int(X_cpu.size(0))
        if self.n_clusters > N:
            raise ValueError(f"n_clusters ({self.n_clusters}) must be <= N ({N}).")
        K, d = self.n_clusters, self.latent_dim
        C = torch.empty(K, d, device=self.device, dtype=X_cpu.dtype)

        idx0 = torch.randint(0, N, (1,), generator=self._generator)
        C[0] = l2_normalize_rows(X_cpu[idx0].to(self.device))[0]

        dmin = torch.full((N,), float("inf"), dtype=torch.float32)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            x = l2_normalize_rows(X_cpu[s:e].to(self.device))
            d = cosine_dissimilarity(x, C[0:1]).squeeze(1).float().cpu()
            dmin[s:e] = torch.minimum(dmin[s:e], d)
            del x, d

        for k in range(1, K):
            w = dmin.clamp_min_(1e-12)
            if self.use_squared_init:
                w = w * w
            probs = w / (w.sum() + 1e-12)
            idx_cpu = torch.multinomial(probs, num_samples=1, generator=self._generator)
            C[k] = l2_normalize_rows(X_cpu[idx_cpu].to(self.device))[0]

            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                x = l2_normalize_rows(X_cpu[s:e].to(self.device))
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
        dtype = self.centroids.dtype
        C_new = torch.zeros(K, self.latent_dim, device=device, dtype=dtype)
        counts = torch.zeros(K, device=device, dtype=dtype)
        N = int(X_cpu.size(0))
        ones = None
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            x = l2_normalize_rows(X_cpu[s:e].to(device, dtype=dtype, non_blocking=True))
            l = labels[s:e]
            C_new.index_add_(0, l, x)
            if (ones is None) or (ones.numel() != (e - s)):
                ones = torch.ones((e - s,), device=device, dtype=dtype)
            counts.scatter_add_(0, l, ones)
            del x, l
        non_empty = counts > 0
        if non_empty.any():
            C_new[non_empty] = l2_normalize_rows(C_new[non_empty] / counts[non_empty].unsqueeze(1))
        return C_new, counts

    # ----------------------------- Fit / Predict -----------------------------
    @torch.no_grad()
    def fit(self, X: torch.Tensor, chunk: Optional[int] = None) -> "CosineKMeans":
        """Fit centroids on X.
        - chunk is None: full-device (X moved to device)
        - chunk > 0: CPU→GPU streaming (VRAM saving)
        """
        if X.ndim != 2 or X.size(1) != self.latent_dim:
            raise ValueError(f"X must be (N,{self.latent_dim}), got {tuple(X.shape)}")
        if not torch.isfinite(X).all():
            raise ValueError("X contains NaN/Inf")
        if self.n_clusters > X.size(0):
            raise ValueError(f"n_clusters ({self.n_clusters}) must be <= N ({X.size(0)})")
        if chunk is not None and chunk <= 0:
            raise ValueError("chunk must be positive")

        stream = (chunk is not None) and (self.device.type == "cuda")
        N = int(X.size(0))

        if stream:
            X_cpu = X.to("cpu", dtype=self.centroids.dtype)
            C = self._init_centroids_kmeanspp_stream(X_cpu, chunk)
        else:
            Xn = l2_normalize_rows(X.to(self.device, dtype=self.centroids.dtype))
            C = self._init_centroids_kmeanspp(Xn)

        prev = None
        last = None
        for _ in range(self.max_iter):
            # E-step
            if stream:
                labels, max_sim = self._assign_in_chunks_cpu(X_cpu, C, chunk)
                mean_J = (1.0 - max_sim).mean().item()
            else:
                sim = Xn @ C.T
                labels = sim.argmax(dim=1)
                max_sim = sim.gather(1, labels.unsqueeze(1)).squeeze(1)
                mean_J = (1.0 - max_sim).mean().item()

            # M-step
            if stream:
                C_new, counts = self._update_centroids_in_chunks_cpu(X_cpu, labels, self.n_clusters, chunk)
            else:
                counts = torch.bincount(labels, minlength=self.n_clusters).to(Xn.dtype)
                C_new = torch.zeros_like(C)
                C_new.index_add_(0, labels, Xn)
                non_empty = counts > 0
                if non_empty.any():
                    C_new[non_empty] = l2_normalize_rows(C_new[non_empty] / counts[non_empty].unsqueeze(1))

            # empty cluster handling: steal farthest points
            non_empty = counts > 0
            if (~non_empty).any():
                num_empty = int((~non_empty).sum().item())
                nearest_d = 1.0 - max_sim  # distance
                far_idx = torch.argsort(nearest_d, descending=True)[:num_empty]
                empty_ids = (~non_empty).nonzero(as_tuple=False).squeeze(1)
                if stream:
                    xfar = l2_normalize_rows(X_cpu[far_idx.cpu()].to(self.device, dtype=C_new.dtype))
                    C_new[empty_ids] = xfar
                else:
                    C_new[empty_ids] = l2_normalize_rows(Xn[far_idx])

            # convergence
            if prev is not None:
                rel = abs(prev - mean_J) / (abs(prev) + 1e-12)
                if (rel < self.tol) or (abs(prev - mean_J) < 1e-7):
                    C = C_new; prev = mean_J; last = mean_J
                    break
            C = C_new
            prev = mean_J
            last = mean_J

        self.centroids.copy_(l2_normalize_rows(C))
        self._fitted = True
        self.inertia_ = float(last if last is not None else prev)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    @torch.no_grad()
    def fit_predict(self, X: torch.Tensor, chunk: Optional[int] = None) -> torch.Tensor:
        self.fit(X, chunk=chunk)
        return self.predict(X, chunk=chunk)

    @torch.no_grad()
    def predict(
        self,
        X: torch.Tensor,
        return_dist: bool = False,
        chunk: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Predict labels (and optionally distances) for X."""
        # --- Centroid 初期化確認 ---
        if (
            not self._fitted
            or self.centroids is None
            or self.centroids.numel() == 0
            or not torch.isfinite(self.centroids).all()
        ):
            raise RuntimeError("Centroids are not initialized. Call fit() or load_state_dict() first.")
        
        if X.ndim != 2 or X.size(1) != self.latent_dim:
            raise ValueError(f"X must be (N,{self.latent_dim}), got {tuple(X.shape)}")

        stream = (chunk is not None) and (self.centroids.device.type == "cuda")
        if stream:
            X_cpu = X.to("cpu", dtype=self.centroids.dtype)
            N = int(X_cpu.size(0))
            labels = torch.empty(N, dtype=torch.long, device=self.centroids.device)
            dist_all = None
            if return_dist:
                dist_all = torch.empty(N, self.centroids.size(0), dtype=self.centroids.dtype, device=self.centroids.device)
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                x = l2_normalize_rows(X_cpu[s:e].to(self.centroids.device, non_blocking=True))
                sim = x @ self.centroids.T
                l = sim.argmax(dim=1)
                labels[s:e] = l
                if return_dist:
                    dist_all[s:e] = 1.0 - sim
                del x, sim
            return (labels, dist_all) if return_dist else labels
        else:
            Xn = l2_normalize_rows(X.to(self.centroids.device, dtype=self.centroids.dtype))
            sim = Xn @ self.centroids.T
            labels = sim.argmax(dim=1)
            if return_dist:
                return labels, (1.0 - sim)
            return labels
        
    def state_dict(self, destination=None, prefix="", keep_vars=False):
        out = super().state_dict(destination, prefix, keep_vars)
        out["inertia_"] = torch.tensor(self.inertia_)
        return out
    
    def load_state_dict(self, state_dict: dict, strict: bool = True):
        """Load centroids and mark the model as fitted."""
        super().load_state_dict(state_dict, strict=strict)
        # centroids が読み込まれたら fitted 状態にする
        self._fitted = True
        # inertia_ が state_dict に含まれていれば復元
        if "inertia_" in state_dict:
            self.inertia_ = float(state_dict["inertia_"])
        
        # 追加: 数値揺れや古いチェックポイント対策として正規化
        with torch.no_grad():
            self.centroids = torch.nn.functional.normalize(self.centroids, dim=1)
        return self


# ----------------------------- Model selection (elbow sweep) -----------------------------
@torch.no_grad()
def elbow_ckmeans(
    cluster_module: Callable[..., CosineKMeans],
    X: torch.Tensor,
    device: str = "cuda",
    k_max: int = 50,
    chunk: Optional[int] = None,
    verbose: bool = True,
):
    """Sweep k=1..k_max, record mean inertia, choose K by curvature.
    Returns: (k_list, inertias, optimal_k, elbow_idx, kappa)
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    d = int(X.size(1))
    inertias: List[float] = []
    k_list = list(range(1, k_max + 1))

    for k in k_list:
        ckm = cluster_module(
            n_clusters=k,
            latent_dim=d,
            tol=1e-3,
            max_iter=500,
            device=device,
            random_state=42,
        )
        ckm.fit(X, chunk=chunk)
        inertias.append(float(ckm.inertia_))
        if verbose:
            print(f"k={k}, mean_inertia={ckm.inertia_:.6f}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    from .ops import find_elbow_curvature  # local import to avoid cycles in type hints
    K, idx, kappa = find_elbow_curvature(k_list, inertias)
    if verbose:
        print(f"Optimal k (curvature): {K}")
    return k_list, inertias, K, idx, kappa
