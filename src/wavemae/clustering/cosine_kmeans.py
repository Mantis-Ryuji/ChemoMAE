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
        tol: convergence tolerance on mean objective (relative OR absolute)
        max_iter: maximum number of EM iterations
        device: "cuda" / "cpu" or torch.device
        random_state: int seed for deterministic multinomial
        use_squared_init: if True, use D(x)^2 for k-means++ probs (通常は False 推奨)

    Notes:
        - 特徴次元 latent_dim は fit(X) 時に X.size(1) から自動決定。
        - 目的関数は平均コサイン不類似度 J = mean(1 - cos(x, c))。
        - E-step: argmax cos、M-step: クラスタ平均→L2正規化（球面 k-means の標準形）。
        - 内部計算は fp32 で実行（half/bf16 入力でも内部で昇格）。
        - 学習後の推論再利用は save_centroids / load_centroids を使用（最終中心のみ保存）。
    """

    def __init__(
        self,
        n_clusters: int = 8,
        tol: float = 1e-3,
        max_iter: int = 500,
        device: Union[str, torch.device] = "cuda",
        random_state: Optional[int] = 42,
        use_squared_init: bool = False,
    ) -> None:
        super().__init__()
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive")

        self.n_clusters = int(n_clusters)
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.device = torch.device(device)
        self.random_state = random_state
        self.use_squared_init = bool(use_squared_init)

        # 次元未確定のため空バッファで登録（state_dict に載せるため register_buffer を使用）
        self.register_buffer("centroids", torch.empty(0, 0, device=self.device))

        self._generator = torch.Generator(device="cpu")
        if random_state is not None:
            self._generator.manual_seed(int(random_state))

        self.latent_dim: Optional[int] = None
        # inertia_ は mean(1 - cos)（SSE ではない点に注意）
        self.inertia_: float = float("inf")
        self._fitted: bool = False

    # ----------------------------- init (k-means++) -----------------------------
    @torch.no_grad()
    def _init_centroids_kmeanspp(self, Xn: torch.Tensor) -> torch.Tensor:
        """Full-device k-means++ (Xn: L2-normalized, device 上, fp32 推奨)."""
        N = int(Xn.size(0))
        if self.n_clusters > N:
            raise ValueError(f"n_clusters ({self.n_clusters}) must be <= N ({N}).")
        K, d = self.n_clusters, int(Xn.size(1))
        C = torch.empty(K, d, device=Xn.device, dtype=Xn.dtype)

        # 1点目
        idx0 = torch.randint(0, N, (1,), generator=self._generator)
        C[0] = Xn[idx0.to(Xn.device)]

        # 2点目以降
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
        """Streaming k-means++: X は CPU のまま、チャンクを device へ送る。"""
        N = int(X_cpu.size(0))
        if self.n_clusters > N:
            raise ValueError(f"n_clusters ({self.n_clusters}) must be <= N ({N}).")
        K, d = self.n_clusters, int(X_cpu.size(1))
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
            if self.use_squared_init:
                w = w * w
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
        """Fit centroids on X.
        - chunk is None: full-device (X moved to device)
        - chunk > 0: CPU→GPU streaming (VRAM saving)
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {tuple(X.shape)}")
        if not torch.isfinite(X).all():
            raise ValueError("X contains NaN/Inf")
        if self.n_clusters > X.size(0):
            raise ValueError(f"n_clusters ({self.n_clusters}) must be <= N ({X.size(0)})")
        if chunk is not None and chunk <= 0:
            raise ValueError("chunk must be positive")

        # 次元の自動確定
        self.latent_dim = int(X.size(1))

        stream = (chunk is not None) and (self.device.type == "cuda")

        # k-means++ 初期化（内部計算は fp32）
        if stream:
            X_cpu = X.to("cpu", dtype=torch.float32)
            C = self._init_centroids_kmeanspp_stream(X_cpu, chunk)
        else:
            Xn = l2_normalize_rows(X.to(self.device, dtype=torch.float32))
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

            # M-step（累積は fp32）
            if stream:
                C_new, counts = self._update_centroids_in_chunks_cpu(X_cpu, labels, self.n_clusters, chunk)
            else:
                counts = torch.bincount(labels, minlength=self.n_clusters).to(torch.float32)
                C_new = torch.zeros_like(C, dtype=torch.float32)
                C_new.index_add_(0, labels, Xn.to(torch.float32))
                non_empty = counts > 0
                if non_empty.any():
                    C_new[non_empty] = l2_normalize_rows(C_new[non_empty] / counts[non_empty].unsqueeze(1))

            # 空クラスタ対応：最遠サンプルを盗む
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

            # 収束判定（相対/絶対）
            if prev is not None:
                rel = abs(prev - mean_J) / (abs(prev) + 1e-12)
                if (rel < self.tol) or (abs(prev - mean_J) < 1e-7):
                    C = C_new
                    prev = mean_J
                    last = mean_J
                    break
            C = C_new
            prev = mean_J
            last = mean_J

        # centroids を L2 正規化してバッファに反映（register_buffer を保持）
        C = l2_normalize_rows(C).to(self.device, dtype=torch.float32)
        if self.centroids.numel() == 0:
            self.register_buffer("centroids", C)
        else:
            self.centroids.resize_(C.shape)
            self.centroids.copy_(C)

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
        """
        Predict labels (and optionally distances) for X.
        return_dist=True の場合、(N, K) の 1 - cos 行列を返す（メモリ使用量に注意）。
        """
        if (
            not self._fitted
            or self.centroids is None
            or self.centroids.numel() == 0
            or not torch.isfinite(self.centroids).all()
        ):
            raise RuntimeError("Centroids are not initialized. Call fit() or load_centroids() first.")

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {tuple(X.shape)}")
        if self.latent_dim is None:
            raise RuntimeError("latent_dim is undefined. Call fit() or load_centroids() first.")
        if int(X.size(1)) != self.latent_dim:
            raise ValueError(f"X dim mismatch: expected {self.latent_dim}, got {int(X.size(1))}")

        stream = (chunk is not None) and (self.centroids.device.type == "cuda")
        if stream:
            X_cpu = X.to("cpu", dtype=torch.float32)
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
            Xn = l2_normalize_rows(X.to(self.centroids.device, dtype=torch.float32))
            sim = Xn @ self.centroids.T
            labels = sim.argmax(dim=1)
            if return_dist:
                return labels, (1.0 - sim)
            return labels

    # ----------------------------- Centroids I/O  -----------------------------
    @torch.no_grad()
    def save_centroids(self, path: str | bytes | "os.PathLike[str]"):
        """最終クラスタ中心だけを保存（推論再利用向けの最小構成）。"""
        if not self._fitted or self.centroids.numel() == 0:
            raise RuntimeError("Model is not fitted; no centroids to save.")
        payload = {
            "centroids": l2_normalize_rows(self.centroids.detach().to("cpu", dtype=torch.float32)),
            "inertia_": float(self.inertia_),
        }
        torch.save(payload, path)

    @torch.no_grad()
    def load_centroids(self, path: str | bytes | "os.PathLike[str]", *, strict_k: bool = True):
        """
        保存した最終クラスタ中心を読み込み、即 predict 可能な状態にする。
        Args:
            path: torch.save で保存したファイル
            strict_k: True のとき、ファイル内の K と self.n_clusters が異なる場合にエラー
        """
        payload = torch.load(path, map_location=self.device)
        if "centroids" not in payload:
            raise KeyError("payload has no 'centroids'.")

        C = payload["centroids"].to(self.device, dtype=torch.float32)
        if C.ndim != 2 or C.size(0) <= 0:
            raise ValueError(f"Invalid centroids shape: {tuple(C.shape)}")
        K, d = int(C.size(0)), int(C.size(1))

        if strict_k and (K != self.n_clusters):
            raise ValueError(f"n_clusters mismatch: expected {self.n_clusters}, file has {K}")

        C = l2_normalize_rows(C)
        if self.centroids.numel() == 0:
            self.register_buffer("centroids", C)
        else:
            self.centroids.resize_(C.shape)
            self.centroids.copy_(C)

        self.latent_dim = d
        self.inertia_ = float(payload.get("inertia_", float("inf")))
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
    """
    Sweep k=1..k_max, record mean inertia, choose K by curvature.
    Returns: (k_list, inertias, optimal_k, elbow_idx, kappa)
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D")

    # 入力を実行デバイスへ（すでに同一ならコピーは発生しない）
    X = X.to(device, non_blocking=True)

    inertias: List[float] = []
    k_list = list(range(1, k_max + 1))

    for k in k_list:
        ckm = cluster_module(
            n_clusters=k,
            tol=1e-3,
            max_iter=500,
            device=device,
            random_state=random_state,
        )
        ckm.fit(X, chunk=chunk)
        inertias.append(float(ckm.inertia_))
        if verbose:
            print(f"k={k}, mean_inertia={ckm.inertia_:.6f}")

        # メモリ掃除（GPUを使っている時のみ）
        gc.collect()
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 局所 import（循環参照回避）
    from .ops import find_elbow_curvature
    K, idx, kappa = find_elbow_curvature(k_list, inertias)
    if verbose:
        print(f"Optimal k (curvature): {K}")
    return k_list, inertias, K, idx, kappa
