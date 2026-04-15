import numpy as np
import pytest
import torch

from chemomae.preprocessing.downsampling import cosine_fps_downsample


def _make_unit_sphere_data(
    N: int = 500,
    C: int = 64,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, C)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X


def _scale_rows(X: np.ndarray, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scales = rng.lognormal(
        mean=0.0,
        sigma=0.5,
        size=(X.shape[0], 1),
    ).astype(np.float32)
    return X * scales


# -----------------------------
# 形状・型・比率境界の基本動作
# -----------------------------
@pytest.mark.parametrize("ratio", [0.0, 0.1, 0.25, 1.0])
@pytest.mark.parametrize("return_numpy", [True, False])
def test_shape_and_type_numpy_input(ratio: float, return_numpy: bool) -> None:
    X = _make_unit_sphere_data(N=123, C=7, seed=7)
    k = min(max(1, int(round(X.shape[0] * ratio))), X.shape[0])

    out = cosine_fps_downsample(
        X,
        ratio=ratio,
        return_numpy=return_numpy,
        seed=42,
    )

    if return_numpy:
        assert isinstance(out, np.ndarray)
    else:
        assert isinstance(out, torch.Tensor)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        assert str(out.device).startswith(dev)

    assert out.shape == (k, X.shape[1])


@pytest.mark.parametrize("ratio", [0.0, 0.1, 0.25, 1.0])
@pytest.mark.parametrize("return_numpy", [True, False])
def test_shape_and_type_torch_input(ratio: float, return_numpy: bool) -> None:
    X = torch.as_tensor(_make_unit_sphere_data(N=200, C=5, seed=0))
    k = min(max(1, int(round(X.shape[0] * ratio))), X.shape[0])

    out = cosine_fps_downsample(
        X,
        ratio=ratio,
        return_numpy=return_numpy,
        seed=123,
    )

    if return_numpy:
        assert isinstance(out, np.ndarray)
    else:
        assert isinstance(out, torch.Tensor)
        assert out.device == X.device

    assert out.shape == (k, X.shape[1])


# -----------------------------
# 再現性（seed / init_index）
# -----------------------------
def test_reproducibility_with_seed() -> None:
    X = _make_unit_sphere_data(N=300, C=16, seed=9)
    A = cosine_fps_downsample(X, ratio=0.2, seed=111, return_numpy=True)
    B = cosine_fps_downsample(X, ratio=0.2, seed=111, return_numpy=True)
    assert np.allclose(A, B)


def test_init_index_controls_first_choice() -> None:
    t = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    X = np.stack([np.cos(t), np.sin(t)], axis=1).astype(np.float32)

    out = cosine_fps_downsample(
        X,
        ratio=0.25,
        init_index=0,
        return_numpy=True,
    )

    expected = X[[0, 4]]
    assert np.allclose(out, expected, atol=1e-6)


# -----------------------------
# 単位球正規化に対する不変性（内部で常に実施）
# -----------------------------
def test_invariance_to_row_scaling_internal_unit_normalization() -> None:
    X = _make_unit_sphere_data(N=256, C=8, seed=123)
    X_scaled = _scale_rows(X, seed=456)

    A = cosine_fps_downsample(X, ratio=0.1, seed=7, return_numpy=True)
    B = cosine_fps_downsample(X_scaled, ratio=0.1, seed=7, return_numpy=True)

    def unit(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

    assert np.allclose(unit(A), unit(B), atol=1e-5)


# -----------------------------
# 空入力 / エラーハンドリング
# -----------------------------
def test_empty_input_numpy() -> None:
    X = np.empty((0, 8), dtype=np.float32)
    out = cosine_fps_downsample(X, ratio=0.3, return_numpy=True)
    assert isinstance(out, np.ndarray)
    assert out.shape == (0, 8)


def test_empty_input_torch() -> None:
    X = torch.empty((0, 5), dtype=torch.float32)
    out = cosine_fps_downsample(X, ratio=0.3, return_numpy=False)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (0, 5)


def test_invalid_dim_raises() -> None:
    with pytest.raises(ValueError):
        cosine_fps_downsample(np.zeros((3, 4, 5), dtype=np.float32))


def test_init_index_out_of_range_raises() -> None:
    X = _make_unit_sphere_data(N=10, C=3, seed=0)
    with pytest.raises(ValueError):
        cosine_fps_downsample(X, ratio=0.2, init_index=999)


# -----------------------------
# 返り値: indices の検証
# -----------------------------
@pytest.mark.parametrize("as_numpy", [True, False])
def test_return_indices_matches_rows(as_numpy: bool) -> None:
    X = _make_unit_sphere_data(N=64, C=9, seed=5)
    subset, indices = cosine_fps_downsample(
        X,
        ratio=0.25,
        seed=2024,
        return_numpy=as_numpy,
        return_indices=True,
    )

    if as_numpy:
        assert isinstance(subset, np.ndarray)
        assert isinstance(indices, np.ndarray)
        assert np.allclose(subset, X[indices])
    else:
        assert torch.is_tensor(subset)
        assert torch.is_tensor(indices)

        # NumPy input + return_numpy=False follows implementation device policy:
        # returned tensors live on the internal compute device (cuda if available else cpu).
        X_t = torch.as_tensor(X, device=subset.device, dtype=subset.dtype)
        idx_t = indices.to(device=subset.device)
        assert torch.allclose(subset, X_t.index_select(0, idx_t))


# -----------------------------
# dtype / device の往復（torch入力）
# -----------------------------
def test_dtype_device_roundtrip_torch() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.as_tensor(
        _make_unit_sphere_data(N=50, C=7, seed=77),
        dtype=torch.float32,
        device=dev,
    )

    out_t = cosine_fps_downsample(X, ratio=0.2, seed=99, return_numpy=False)
    assert torch.is_tensor(out_t)
    assert out_t.device == X.device
    assert out_t.dtype == X.dtype

    out_np = cosine_fps_downsample(X, ratio=0.2, seed=99, return_numpy=True)
    assert isinstance(out_np, np.ndarray)
    assert out_np.shape[1] == X.shape[1]