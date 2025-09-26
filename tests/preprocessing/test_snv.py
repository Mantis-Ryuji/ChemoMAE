import math
import numpy as np
import pytest

from chemomae.preprocessing.snv import snv, SNVScaler

# torch は任意依存。なければ該当テストだけ skip
try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


def _rowwise_mean_std(x: np.ndarray):
    """確認用に行ごとの (mean, std[ddof=1]) を返す"""
    if x.ndim == 1:
        mu = x.mean()
        sd = np.std(x, ddof=1) if x.size >= 2 else np.std(x, ddof=0)
        return mu, sd
    else:
        mu = x.mean(axis=1, keepdims=True)
        ddof = 1 if x.shape[1] >= 2 else 0
        sd = np.std(x, axis=1, ddof=ddof, keepdims=True)
        return mu, sd


# -------------------------
# snv (関数) のテスト
# -------------------------

def test_snv_numpy_1d_zero_mean_unit_std_unbiased():
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    y = snv(x)
    # 出力は float32
    assert y.dtype == np.float32
    # 平均 ≈ 0、標準偏差（ddof=1） ≈ 1
    assert math.isclose(float(y.mean()), 0.0, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(float(np.std(y, ddof=1)), 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_snv_numpy_2d_rowwise_standardization():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 7)).astype(np.float32)
    y = snv(x)
    # 各行で平均≈0, 不偏標準偏差≈1
    mu = y.mean(axis=1, keepdims=True)
    sd = np.std(y, axis=1, ddof=1, keepdims=True)
    assert np.allclose(mu, 0.0, atol=1e-6)
    assert np.allclose(sd, 1.0, atol=1e-6)


def test_snv_length1_no_nan():
    # 要素数1のときは ddof=0 にフォールバックし、NaN を出さない
    x = np.array([5.0], dtype=np.float32)
    y = snv(x)
    assert np.isfinite(y).all()


def test_snv_raises_on_3d_input():
    x = np.zeros((2, 3, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        _ = snv(x)


@pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")
def test_snv_torch_preserves_type_and_device():
    device = torch.device("cpu")
    x = torch.tensor([[1.0, 2.0], [3.0, 5.0]], dtype=torch.float64, device=device)
    y = snv(x)
    # 入力の dtype / device を維持
    assert isinstance(y, torch.Tensor)
    assert y.dtype == x.dtype
    assert y.device == x.device
    # 行ごとに標準化（ddof=1）
    mu = y.mean(dim=1, keepdim=True)
    sd = y.std(dim=1, unbiased=True, keepdim=True)
    assert torch.allclose(mu, torch.zeros_like(mu), atol=1e-6)
    assert torch.allclose(sd, torch.ones_like(sd), atol=1e-6)


# -------------------------
# SNVScaler のテスト
# -------------------------

def test_snv_scaler_transform_equals_functional_numpy():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 6)).astype(np.float32)
    y_func = snv(x)
    scaler = SNVScaler()
    y_scale = scaler.transform(x)
    assert isinstance(y_scale, np.ndarray)
    assert y_scale.dtype == np.float32
    assert np.allclose(y_scale, y_func, atol=1e-7)


@pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")
def test_snv_scaler_transform_equals_functional_torch():
    x = torch.tensor([[1.0, 4.0, 2.0],
                      [3.0, 1.0, 2.0]], dtype=torch.float32)
    y_func = snv(x)
    scaler = SNVScaler()
    y_scale = scaler.transform(x)
    assert isinstance(y_scale, torch.Tensor)
    # dtype / device 維持（snv本体の仕様に追従）
    assert y_scale.dtype == x.dtype
    assert y_scale.device == x.device
    assert torch.allclose(y_scale, y_func, atol=1e-7)


def test_snv_scaler_transform_stats_and_inverse_numpy():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(3, 5)).astype(np.float32)

    scaler = SNVScaler(transform_stats=True)
    y, mu, sd = scaler.transform(x)

    # y は標準化済み
    mu_y = y.mean(axis=1, keepdims=True)
    sd_y = np.std(y, axis=1, ddof=1, keepdims=True)
    assert np.allclose(mu_y, 0.0, atol=1e-6)
    assert np.allclose(sd_y, 1.0, atol=1e-6)

    # inverse_transform で元に戻る（数値誤差レベル）
    x_rec = scaler.inverse_transform(y, mu=mu, sd=sd)
    assert np.allclose(x_rec, x, atol=1e-6)


@pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")
def test_snv_scaler_transform_stats_and_inverse_torch():
    x = torch.tensor([[0.1, 0.2, 0.5, 0.6],
                      [2.0, 1.0, 4.0, 3.0]], dtype=torch.float32)

    scaler = SNVScaler(transform_stats=True)
    y, mu, sd = scaler.transform(x)

    # y は torch、mu/sd は numpy（仕様通り）
    assert isinstance(y, torch.Tensor)
    assert isinstance(mu, np.ndarray)
    assert isinstance(sd, np.ndarray)

    # 標準化確認
    mu_y = y.mean(dim=1, keepdim=True)
    sd_y = y.std(dim=1, unbiased=True, keepdim=True)
    assert torch.allclose(mu_y, torch.zeros_like(mu_y), atol=1e-6)
    assert torch.allclose(sd_y, torch.ones_like(sd_y), atol=1e-6)

    # 逆変換で元に戻る
    x_rec = scaler.inverse_transform(y, mu=mu, sd=sd)
    # 逆変換の戻りも torch
    assert isinstance(x_rec, torch.Tensor)
    assert torch.allclose(x_rec, x, atol=1e-6)
