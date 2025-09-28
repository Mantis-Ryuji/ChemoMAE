import math
import numpy as np
import pytest

import torch
from sklearn.metrics import silhouette_samples as sk_silhouette_samples

from chemomae.clustering.metric import (
    silhouette_samples_cosine_gpu,
    silhouette_score_cosine_gpu,
)


def _make_blob_data(n_per_cluster=30, d=16, k=3, seed=0, add_zeros=False):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12
    X_list = []
    y_list = []
    for c in range(k):
        Xc = centers[c] + 0.1 * rng.normal(size=(n_per_cluster, d)).astype(np.float32)
        X_list.append(Xc)
        y_list.append(np.full(n_per_cluster, c, dtype=np.int64))
    X = np.vstack(X_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.int64)
    if add_zeros:
        n_zero = max(1, len(X) // 15)
        X[:n_zero] = 0.0
    return X, y


def test_equivalence_to_sklearn_cpu_small():
    """CPU: numpy I/O equivalence to sklearn silhouette_samples(metric='cosine')."""
    X, y = _make_blob_data(n_per_cluster=20, d=8, k=3, seed=123)
    ours = silhouette_samples_cosine_gpu(
        X, y, device="cpu", chunk=None, return_numpy=True, dtype=torch.float64
    )
    ref = sk_silhouette_samples(X, y, metric="cosine")
    np.testing.assert_allclose(ours, ref, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("chunk", [None, 1024, 17])
def test_chunking_invariance_cpu(chunk):
    """CPU: invariance w.r.t. chunk size."""
    X, y = _make_blob_data(n_per_cluster=25, d=12, k=4, seed=7)
    base = silhouette_samples_cosine_gpu(X, y, device="cpu", chunk=None, return_numpy=True)
    test = silhouette_samples_cosine_gpu(X, y, device="cpu", chunk=chunk, return_numpy=True)
    np.testing.assert_allclose(test, base, rtol=0, atol=0)


def test_nonconsecutive_labels_equivalence():
    """Non-consecutive labels produce identical results to remapped labels."""
    X, y = _make_blob_data(n_per_cluster=10, d=7, k=3, seed=42)
    y_non = y.copy()
    y_non[y == 0] = 10
    y_non[y == 1] = 30
    y_non[y == 2] = 20
    s_non = silhouette_samples_cosine_gpu(X, y_non, device="cpu", chunk=None, return_numpy=True)
    s_seq = silhouette_samples_cosine_gpu(X, y, device="cpu", chunk=None, return_numpy=True)
    np.testing.assert_allclose(s_non, s_seq, rtol=0, atol=0)


def test_singleton_cluster_yields_zero():
    """Singleton cluster sample should yield silhouette 0."""
    X, y = _make_blob_data(n_per_cluster=15, d=10, k=3, seed=0)
    X = np.vstack([X, X[:1] + 0.0])
    y = np.concatenate([y, np.array([99], dtype=np.int64)])
    s = silhouette_samples_cosine_gpu(X, y, device="cpu", chunk=None, return_numpy=True)
    assert s[-1] == 0.0


def test_zero_vectors_match_sklearn():
    """Zero rows behave like sklearn (cosine: distance 1 to all)."""
    X, y = _make_blob_data(n_per_cluster=12, d=9, k=3, seed=5, add_zeros=True)
    ours = silhouette_samples_cosine_gpu(
        X, y, device="cpu", chunk=None, return_numpy=True, dtype=torch.float64
    )
    ref = sk_silhouette_samples(X, y, metric="cosine")
    np.testing.assert_allclose(ours, ref, rtol=1e-7, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_matches_cpu_and_chunking():
    """GPU results match CPU and are invariant to chunk size."""
    X, y = _make_blob_data(n_per_cluster=40, d=32, k=5, seed=77)
    cpu = silhouette_samples_cosine_gpu(
        X, y, device="cpu", chunk=None, return_numpy=False
    ).cpu().numpy()
    gpu_full = silhouette_samples_cosine_gpu(X, y, device="cuda", chunk=None, return_numpy=True)
    gpu_chunk = silhouette_samples_cosine_gpu(X, y, device="cuda", chunk=12345, return_numpy=True)
    np.testing.assert_allclose(gpu_full, cpu, rtol=0, atol=0)
    np.testing.assert_allclose(gpu_chunk, cpu, rtol=0, atol=0)


@pytest.mark.parametrize("return_numpy", [True, False])
def test_return_type_and_score(return_numpy):
    """Return type and mean score check."""
    X, y = _make_blob_data(n_per_cluster=18, d=11, k=3, seed=9)
    s = silhouette_samples_cosine_gpu(X, y, device="cpu", chunk=None, return_numpy=return_numpy)

    if return_numpy:
        assert isinstance(s, np.ndarray)
        # NumPy 側も float32 で平均（既定は float64）
        score = float(np.asarray(s, dtype=np.float32).mean(dtype=np.float32))
        tol = 1e-7   # reduce 実装差による極小差分は許容
    else:
        assert torch.is_tensor(s)
        score = float(s.float().mean(dtype=torch.float32).item())
        tol = 0.0    # 同じ Torch 経路なので完全一致期待

    # silhouette_score_cosine_gpu は内部計算を常に Torch(fp32) 経路に統一
    score2 = silhouette_score_cosine_gpu(X, y, device="cpu", chunk=None)

    assert math.isclose(score, score2, rel_tol=0.0, abs_tol=tol)



@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dtype", [torch.float16, getattr(torch, "bfloat16", torch.float16)])
def test_low_precision_dtypes(dtype):
    """Half/BFloat16 consistency against float32 (allowing small tolerance)."""
    X, y = _make_blob_data(n_per_cluster=32, d=64, k=4, seed=11)
    s_low = silhouette_samples_cosine_gpu(
        X, y, device="cuda", chunk=4096, return_numpy=True, dtype=dtype
    )
    s_ref = silhouette_samples_cosine_gpu(
        X, y, device="cuda", chunk=None, return_numpy=True, dtype=torch.float32
    )
    np.testing.assert_allclose(s_low, s_ref, rtol=5e-4, atol=5e-4)
