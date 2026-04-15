import math

import numpy as np
import pytest
import torch
from sklearn.metrics import silhouette_samples as sk_silhouette_samples

from chemomae.clustering.metric import (
    silhouette_samples_cosine_gpu,
    silhouette_score_cosine_gpu,
)


def _make_blob_data(
    n_per_cluster: int = 30,
    d: int = 16,
    k: int = 3,
    seed: int = 0,
    add_zeros: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for c in range(k):
        xc = centers[c] + 0.1 * rng.normal(size=(n_per_cluster, d)).astype(np.float32)
        x_list.append(xc)
        y_list.append(np.full(n_per_cluster, c, dtype=np.int64))

    x = np.vstack(x_list).astype(np.float32)
    y = np.concatenate(y_list).astype(np.int64)

    if add_zeros:
        n_zero = max(1, len(x) // 15)
        x[:n_zero] = 0.0

    return x, y


def test_equivalence_to_sklearn_cpu_small() -> None:
    """CPU: numpy I/O equivalence to sklearn silhouette_samples(metric='cosine')."""
    x, y = _make_blob_data(n_per_cluster=20, d=8, k=3, seed=123)
    ours = silhouette_samples_cosine_gpu(
        x,
        y,
        device="cpu",
        chunk=None,
        return_numpy=True,
        dtype=torch.float64,
    )
    ref = sk_silhouette_samples(x, y, metric="cosine")
    np.testing.assert_allclose(ours, ref, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("chunk", [None, 1024, 17])
def test_chunking_invariance_cpu(chunk: int | None) -> None:
    """CPU: invariance w.r.t. chunk size."""
    x, y = _make_blob_data(n_per_cluster=25, d=12, k=4, seed=7)
    base = silhouette_samples_cosine_gpu(
        x,
        y,
        device="cpu",
        chunk=None,
        return_numpy=True,
    )
    test = silhouette_samples_cosine_gpu(
        x,
        y,
        device="cpu",
        chunk=chunk,
        return_numpy=True,
    )
    np.testing.assert_allclose(test, base, rtol=0.0, atol=0.0)


def test_nonconsecutive_labels_equivalence() -> None:
    """Non-consecutive labels produce identical results to remapped labels."""
    x, y = _make_blob_data(n_per_cluster=10, d=7, k=3, seed=42)

    y_non = y.copy()
    y_non[y == 0] = 10
    y_non[y == 1] = 30
    y_non[y == 2] = 20

    s_non = silhouette_samples_cosine_gpu(
        x,
        y_non,
        device="cpu",
        chunk=None,
        return_numpy=True,
    )
    s_seq = silhouette_samples_cosine_gpu(
        x,
        y,
        device="cpu",
        chunk=None,
        return_numpy=True,
    )
    np.testing.assert_allclose(s_non, s_seq, rtol=0.0, atol=0.0)


def test_singleton_cluster_yields_zero() -> None:
    """Singleton cluster sample should yield silhouette 0."""
    x, y = _make_blob_data(n_per_cluster=15, d=10, k=3, seed=0)
    x = np.vstack([x, x[:1] + 0.0])
    y = np.concatenate([y, np.array([99], dtype=np.int64)])

    s = silhouette_samples_cosine_gpu(
        x,
        y,
        device="cpu",
        chunk=None,
        return_numpy=True,
    )
    assert s[-1] == 0.0


def test_zero_vectors_match_sklearn() -> None:
    """Zero rows behave like sklearn (cosine: distance 1 to all)."""
    x, y = _make_blob_data(n_per_cluster=12, d=9, k=3, seed=5, add_zeros=True)
    ours = silhouette_samples_cosine_gpu(
        x,
        y,
        device="cpu",
        chunk=None,
        return_numpy=True,
        dtype=torch.float64,
    )
    ref = sk_silhouette_samples(x, y, metric="cosine")
    np.testing.assert_allclose(ours, ref, rtol=1e-7, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_matches_cpu_and_chunking() -> None:
    """GPU results match CPU within float32-level tolerance and are chunk-invariant."""
    x, y = _make_blob_data(n_per_cluster=40, d=32, k=5, seed=77)

    cpu = silhouette_samples_cosine_gpu(
        x,
        y,
        device="cpu",
        chunk=None,
        return_numpy=False,
    ).cpu().numpy()

    gpu_full = silhouette_samples_cosine_gpu(
        x,
        y,
        device="cuda",
        chunk=None,
        return_numpy=True,
    )
    gpu_chunk = silhouette_samples_cosine_gpu(
        x,
        y,
        device="cuda",
        chunk=12345,
        return_numpy=True,
    )

    # CPU/GPU reductions need not be bitwise identical.
    np.testing.assert_allclose(gpu_full, cpu, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gpu_chunk, cpu, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gpu_chunk, gpu_full, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("return_numpy", [True, False])
def test_return_type_and_score(return_numpy: bool) -> None:
    """Return type and mean score check."""
    x, y = _make_blob_data(n_per_cluster=18, d=11, k=3, seed=9)
    s = silhouette_samples_cosine_gpu(
        x,
        y,
        device="cpu",
        chunk=None,
        return_numpy=return_numpy,
    )

    if return_numpy:
        assert isinstance(s, np.ndarray)
        score = float(np.asarray(s, dtype=np.float32).mean(dtype=np.float32))
        score2 = float(
            silhouette_score_cosine_gpu(
                x,
                y,
                device="cpu",
                chunk=None,
                return_numpy=True,
            )
        )
        # NumPy(fp32) vs torch(fp32) reduce path may differ slightly.
        assert math.isclose(score, score2, rel_tol=0.0, abs_tol=2e-7)
    else:
        assert torch.is_tensor(s)
        score = float(s.float().mean(dtype=torch.float32).item())
        score2_t = silhouette_score_cosine_gpu(
            x,
            y,
            device="cpu",
            chunk=None,
            return_numpy=False,
        )
        assert torch.is_tensor(score2_t)
        assert score2_t.numel() == 1
        score2 = float(score2_t.item())
        assert math.isclose(score, score2, rel_tol=0.0, abs_tol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        getattr(torch, "bfloat16", torch.float16),
    ],
)
def test_low_precision_dtypes(dtype: torch.dtype) -> None:
    """Half/BFloat16 consistency against float32 with dtype-aware tolerance.

    Note
    ----
    Low-precision CUDA reductions are not expected to match float32 bitwise.
    In particular, bfloat16 has much coarser mantissa precision than float16,
    so a looser tolerance is required.
    """
    x, y = _make_blob_data(n_per_cluster=32, d=64, k=4, seed=11)

    s_low = silhouette_samples_cosine_gpu(
        x,
        y,
        device="cuda",
        chunk=4096,
        return_numpy=True,
        dtype=dtype,
    )
    s_ref = silhouette_samples_cosine_gpu(
        x,
        y,
        device="cuda",
        chunk=None,
        return_numpy=True,
        dtype=torch.float32,
    )

    assert isinstance(s_low, np.ndarray)
    assert isinstance(s_ref, np.ndarray)

    if dtype == torch.float16:
        rtol = 1e-3
        atol = 1e-3
    else:
        # bfloat16 is substantially coarser than fp16 on mantissa precision.
        rtol = 1.5e-2
        atol = 7e-3

    np.testing.assert_allclose(s_low, s_ref, rtol=rtol, atol=atol)