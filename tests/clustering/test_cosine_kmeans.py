import math
import torch
import numpy as np
import pytest

from wavemae.clustering.cosine_kmeans import CosineKMeans, elbow_ckmeans
from wavemae.clustering.ops import l2_normalize_rows


def _make_spherical_blobs(n_per=40, noise=0.10, seed=0):
    """
    2次元の単位円上に 3 クラスタ（120度間隔）。ノイズを加えて行正規化。
    戻り値: (X, true_centers)
    """
    rng = np.random.default_rng(seed)
    centers = np.array([
        [1.0, 0.0],
        [-0.5,  math.sqrt(3)/2],
        [-0.5, -math.sqrt(3)/2],
    ], dtype=np.float32)
    Xs = []
    for c in centers:
        pts = c + noise * rng.standard_normal(size=(n_per, 2)).astype(np.float32)
        # 行正規化して球面上に
        pts = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12)
        Xs.append(pts)
    X = np.concatenate(Xs, axis=0).astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(centers.astype(np.float32))


def test_fit_predict_basic_properties_cpu():
    X, _ = _make_spherical_blobs(n_per=30, noise=0.08, seed=1)
    model = CosineKMeans(n_clusters=3, device="cpu", random_state=42, tol=1e-4, max_iter=200)
    model.fit(X)
    assert model._fitted is True
    assert model.centroids.shape == (3, X.shape[1])
    # セントロイドは単位ベクトル
    norms = model.centroids.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    labels = model.predict(X)
    assert labels.shape == (X.shape[0],)
    # inertia は mean(1 - cos) ∈ [0, 2]
    assert 0.0 <= model.inertia_ <= 2.0
    # すべてのクラスタが非空
    counts = torch.bincount(labels, minlength=3)
    assert torch.all(counts > 0)


def test_predict_before_fit_raises_and_dim_mismatch():
    X, _ = _make_spherical_blobs()
    m = CosineKMeans(n_clusters=3, device="cpu")
    # 未fitで predict → 例外
    try:
        _ = m.predict(X)
        raised = False
    except RuntimeError:
        raised = True
    assert raised

    # 違う次元の入力でエラー
    m.fit(X)
    with pytest.raises(ValueError):
        _ = m.predict(torch.randn(X.size(0), X.size(1) + 1))


def test_save_and_load_centroids_and_strict_k(tmp_path):
    X, _ = _make_spherical_blobs(n_per=20, noise=0.05, seed=7)
    m1 = CosineKMeans(n_clusters=3, device="cpu", random_state=0).fit(X)
    path = tmp_path / "centroids.pt"
    m1.save_centroids(path)

    # 同一Kでロード → predict 可能
    m2 = CosineKMeans(n_clusters=3, device="cpu", random_state=0).load_centroids(path, strict_k=True)
    assert m2._fitted and m2.latent_dim == X.shape[1]
    _ = m2.predict(X)  # smoke

    # K不一致で strict_k=True → 例外
    m3 = CosineKMeans(n_clusters=2, device="cpu", random_state=0)
    with pytest.raises(ValueError):
        m3.load_centroids(path, strict_k=True)


def test_predict_return_dist_shape_and_values():
    X, _ = _make_spherical_blobs(n_per=15, noise=0.1, seed=11)
    m = CosineKMeans(n_clusters=3, device="cpu", random_state=0).fit(X)
    labels, dist = m.predict(X, return_dist=True)
    assert labels.shape == (X.shape[0],)
    assert dist.shape == (X.shape[0], 3)
    # 1 - cos の範囲確認
    assert torch.all(dist >= -1e-6) and torch.all(dist <= 2.0 + 1e-6)


def test_elbow_ckmeans_smoke_cpu():
    X, _ = _make_spherical_blobs(n_per=10, noise=0.12, seed=3)
    # 注意: elbow_ckmeans は内部で device へ移すので device="cpu" を指定
    k_list, inertias, K, idx, kappa = elbow_ckmeans(
        CosineKMeans, X, device="cpu", k_max=6, chunk=None, verbose=False, random_state=0
    )
    assert isinstance(k_list, list) and isinstance(inertias, list)
    assert len(k_list) == len(inertias) == 6
    assert 1 <= K <= 6 and 0 <= idx < 6
    assert isinstance(kappa, float) or np.isscalar(kappa) or hasattr(kappa, "__float__")
