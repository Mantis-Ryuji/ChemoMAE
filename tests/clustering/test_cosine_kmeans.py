import math
from copy import deepcopy
from pathlib import Path
import torch
import numpy as np
import pytest

from chemomae.clustering.cosine_kmeans import CosineKMeans, elbow_ckmeans


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
    model = CosineKMeans(n_components=3, device="cpu", random_state=42, tol=1e-4, max_iter=200)
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
    m = CosineKMeans(n_components=3, device="cpu")
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
    m1 = CosineKMeans(n_components=3, device="cpu", random_state=0).fit(X)
    path = tmp_path / "centroids.pt"
    m1.save_centroids(path)

    # 同一Kでロード → predict 可能
    m2 = CosineKMeans(n_components=3, device="cpu", random_state=0).load_centroids(path, strict_k=True)
    assert m2._fitted and m2.latent_dim == X.shape[1]
    assert torch.equal(m2.centroids, m1.centroids)
    assert torch.equal(m2.predict(X), m1.predict(X))
    assert m2.inertia_ == m1.inertia_
    assert (m2.n_iter_, m2.converged_, m2.stop_reason_) == (
        m1.n_iter_, m1.converged_, m1.stop_reason_
    )

    # K不一致で strict_k=True → 例外
    m3 = CosineKMeans(n_components=2, device="cpu", random_state=0)
    with pytest.raises(ValueError):
        m3.load_centroids(path, strict_k=True)
    m3.load_centroids(path, strict_k=False)
    assert m3.n_components == 3
    assert torch.equal(m3.centroids, m1.centroids)


def test_predict_return_dist_shape_and_values():
    X, _ = _make_spherical_blobs(n_per=15, noise=0.1, seed=11)
    m = CosineKMeans(n_components=3, device="cpu", random_state=0).fit(X)
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


def test_one_update_objective_uses_final_centers_and_reports_limit() -> None:
    # Either initial center gives the pre-update objective 0.5. The updated
    # direction is the diagonal, with the smaller objective 1 - sqrt(0.5).
    X = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    model = CosineKMeans(n_components=1, max_iter=1, tol=0, device="cpu").fit(X)
    labels, distances = model.predict(X, return_dist=True)
    expected = distances.gather(1, labels[:, None]).mean().item()
    assert model.inertia_ == expected
    assert model.inertia_ == pytest.approx(1 - math.sqrt(0.5), abs=1e-7)
    assert model.inertia_ != pytest.approx(0.5)
    assert model.n_iter_ == 1 and model.converged_ is False
    assert model.stop_reason_ == "max_iter"


def test_convergence_diagnostics_count_updates_and_reset_on_refit() -> None:
    model = CosineKMeans(n_components=1, max_iter=5, device="cpu")
    assert model.n_iter_ == 0 and model.converged_ is False and model.stop_reason_ is None
    X = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    model.fit(X)
    assert model.n_iter_ == 2 and model.converged_ is True
    assert model.stop_reason_ == "tolerance" and model.inertia_ == 0.0
    model.max_iter = 1
    model.fit(X)
    assert model.n_iter_ == 1 and model.converged_ is False
    assert model.stop_reason_ == "max_iter"


def test_fitted_state_roundtrip_does_not_renormalize_or_cast_centers(tmp_path: Path) -> None:
    X = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    source = CosineKMeans(
        n_components=1, max_iter=1, tol=0.003, random_state=17, device="cpu"
    ).fit(X)
    # A near-unit imported predictor whose FP32 values would change if normalized.
    source.centroids.copy_(torch.tensor([[1.0, 0.001]]))
    _, distances = source.predict(X, return_dist=True)
    source.inertia_ = distances.min(dim=1).values.mean().item()
    before = source.centroids.clone()
    path = tmp_path / "fitted.pt"
    source.save_centroids(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["format_version"] == 1
    assert torch.equal(payload["centroids"], before)
    restored = CosineKMeans(n_components=1, device="cpu").load_centroids(path)
    assert restored.centroids.dtype == source.centroids.dtype
    assert torch.equal(restored.centroids, before)
    assert torch.equal(source.centroids, before)
    assert restored.tol == source.tol and restored.max_iter == source.max_iter
    assert restored.random_state == source.random_state
    assert (restored.n_iter_, restored.converged_, restored.stop_reason_) == (
        source.n_iter_, source.converged_, source.stop_reason_
    )
    old_labels, old_distances = source.predict(X, return_dist=True)
    new_labels, new_distances = restored.predict(X, return_dist=True)
    assert torch.equal(new_labels, old_labels) and torch.equal(new_distances, old_distances)


def _fitted_payload() -> dict[str, object]:
    return {
        "format_version": 1,
        "config": {"n_components": 1, "tol": 1e-4, "max_iter": 1, "random_state": 42},
        "centroids": torch.tensor([[1.0, 0.0]]),
        "latent_dim": 2, "inertia_": 0.0, "n_iter_": 1,
        "converged_": False, "stop_reason_": "max_iter",
    }


@pytest.mark.parametrize("field, value, message", [
    ("format_version", 2, "format"),
    ("centroids", [[1.0, 0.0]], "Torch tensor"),
    ("centroids", torch.tensor([[float("nan"), 0.0]]), "NaN/Inf"),
    ("centroids", torch.tensor([[1.0, float("inf")]]), "NaN/Inf"),
    ("centroids", torch.ones(1, 2, dtype=torch.float64), "FP32"),
    ("centroids", torch.ones(1, 0), "FP32"),
    ("centroids", torch.ones(2, 2), "matching"),
    ("latent_dim", 3, "matching"),
    ("latent_dim", 0, "positive integer"),
    ("inertia_", float("inf"), "finite"),
    ("n_iter_", 0, "n_iter_"),
    ("converged_", True, "inconsistent"),
    ("stop_reason_", "other", "stop_reason_"),
])
def test_invalid_fitted_payloads_fail_without_mutating_model(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    model = CosineKMeans(n_components=1, max_iter=1, device="cpu").fit(torch.tensor([[0.0, 1.0]]))
    centers = model.centroids.clone()
    old_diagnostics = (model.n_iter_, model.converged_, model.stop_reason_, model.inertia_)
    payload = _fitted_payload()
    payload[field] = value
    path = tmp_path / "invalid.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match=message):
        model.load_centroids(path)
    assert torch.equal(model.centroids, centers)
    assert (model.n_iter_, model.converged_, model.stop_reason_, model.inertia_) == old_diagnostics
    assert model._fitted is True


@pytest.mark.parametrize("config", [
    {"n_components": 0}, {"n_components": 1.5}, {"n_components": True},
    {"max_iter": 0}, {"max_iter": -1}, {"max_iter": 1.5}, {"max_iter": True},
    {"tol": -1}, {"tol": float("nan")}, {"tol": float("inf")},
    {"random_state": 1.5}, {"random_state": True}, {"random_state": 2**64},
])
def test_constructor_and_saved_configuration_validation(
    tmp_path: Path, config: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        CosineKMeans(device="cpu", **config)
    payload = deepcopy(_fitted_payload())
    saved_config = payload["config"]
    assert isinstance(saved_config, dict)
    saved_config.update(config)
    path = tmp_path / "invalid-config.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError):
        CosineKMeans(n_components=1, device="cpu").load_centroids(path)


def test_unversioned_center_files_and_unfitted_save_are_rejected(tmp_path: Path) -> None:
    model = CosineKMeans(n_components=1, device="cpu")
    with pytest.raises(RuntimeError, match="not fitted"):
        model.save_centroids(tmp_path / "unfitted.pt")
    path = tmp_path / "old.pt"
    torch.save({"centroids": torch.tensor([[1.0, 0.0]]), "inertia_": 0.0}, path)
    with pytest.raises(ValueError, match="format_version=1"):
        model.load_centroids(path)


def test_zero_center_convention_survives_fitted_state_roundtrip(tmp_path: Path) -> None:
    X = torch.zeros(2, 3)
    source = CosineKMeans(n_components=1, device="cpu").fit(X)
    assert source.inertia_ == 1.0 and torch.equal(source.centroids, torch.zeros(1, 3))
    path = tmp_path / "zero-center.pt"
    source.save_centroids(path)
    restored = CosineKMeans(n_components=1, device="cpu").load_centroids(path)
    assert torch.equal(restored.centroids, source.centroids)
    assert torch.equal(restored.predict(X), source.predict(X))


@pytest.mark.parametrize("features", [
    torch.empty(0, 2), torch.empty(2, 0), torch.ones(2),
    torch.tensor([[float("nan"), 0.0]]), torch.tensor([[float("inf"), 0.0]]),
    torch.ones(1, 2, dtype=torch.complex64), torch.ones(1, 2, dtype=torch.bool),
    torch.tensor([[1e100, 0.0]], dtype=torch.float64),
])
def test_fit_and_predict_reject_invalid_features(features: torch.Tensor) -> None:
    model = CosineKMeans(n_components=1, device="cpu")
    with pytest.raises(ValueError):
        model.fit(features)
    model.fit(torch.tensor([[1.0, 0.0]]))
    with pytest.raises(ValueError):
        model.predict(features)


@pytest.mark.parametrize("chunk", [0, -1, 1.5, True])
def test_fit_and_predict_reject_invalid_chunks(chunk: object) -> None:
    X = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    model = CosineKMeans(n_components=1, device="cpu")
    with pytest.raises(ValueError, match="chunk"):
        model.fit(X, chunk=chunk)
    model.fit(X)
    with pytest.raises(ValueError, match="chunk"):
        model.predict(X, chunk=chunk)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_streaming_final_objective_matches_saved_predictor() -> None:
    X, _ = _make_spherical_blobs(n_per=10, seed=29)
    model = CosineKMeans(n_components=3, max_iter=2, device="cuda").fit(X, chunk=7)
    labels, distances = model.predict(X, return_dist=True, chunk=7)
    expected = distances.gather(1, labels[:, None]).mean().item()
    assert model.inertia_ == pytest.approx(expected, abs=1e-7)
    assert 1 <= model.n_iter_ <= 2
