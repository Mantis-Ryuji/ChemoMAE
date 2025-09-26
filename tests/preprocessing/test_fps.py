# test_fps_downsample.py
import math
import os
import sys
import numpy as np
import torch
import pytest

# --- import fps_downsample  ---
from chemomae.preprocessing.downsampling import fps_downsample 


def _make_unit_sphere_data(N=500, C=64, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, C)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X


def _scale_rows(X, seed=1):
    rng = np.random.default_rng(seed)
    scales = rng.lognormal(mean=0.0, sigma=0.5, size=(X.shape[0], 1)).astype(np.float32)
    return X * scales


# -----------------------------
# 形状・型・比率境界の基本動作
# -----------------------------
@pytest.mark.parametrize("ratio", [0.0, 0.1, 0.25, 1.0])
@pytest.mark.parametrize("return_numpy", [True, False])
def test_shape_and_type_numpy_input(ratio, return_numpy):
    X = _make_unit_sphere_data(N=123, C=7, seed=7)  # 非10の倍数で round の確認
    k = max(1, int(round(X.shape[0] * ratio)))
    out = fps_downsample(X, ratio=ratio, return_numpy=return_numpy, seed=42)

    if return_numpy:
        assert isinstance(out, np.ndarray)
    else:
        assert isinstance(out, torch.Tensor)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        assert str(out.device).startswith(dev)

    assert out.shape == (k, X.shape[1])


@pytest.mark.parametrize("ratio", [0.0, 0.1, 0.25, 1.0])
@pytest.mark.parametrize("return_numpy", [True, False])
def test_shape_and_type_torch_input(ratio, return_numpy):
    X = torch.as_tensor(_make_unit_sphere_data(N=200, C=5, seed=0))
    k = max(1, int(round(X.shape[0] * ratio)))
    out = fps_downsample(X, ratio=ratio, return_numpy=return_numpy, seed=123)

    if return_numpy:
        assert isinstance(out, np.ndarray)
    else:
        assert isinstance(out, torch.Tensor)
        # torch入力→同じデバイスで返る仕様
        assert out.device == X.device

    assert out.shape == (k, X.shape[1])


# -----------------------------
# 再現性（seed / init_index）
# -----------------------------
def test_reproducibility_with_seed():
    X = _make_unit_sphere_data(N=300, C=16, seed=9)
    A = fps_downsample(X, ratio=0.2, seed=111, return_numpy=True)
    B = fps_downsample(X, ratio=0.2, seed=111, return_numpy=True)
    assert np.allclose(A, B)


def test_init_index_controls_first_choice():
    # 等間隔の単位円（C=2）で、init_index=0なら次は反対側（index 4）を選びやすいケース
    t = np.linspace(0, 2*np.pi, 8, endpoint=False)
    X = np.stack([np.cos(t), np.sin(t)], axis=1).astype(np.float32)
    # ratio=0.25 → k=2
    out = fps_downsample(X, ratio=0.25, init_index=0, ensure_unit_sphere=True, return_numpy=True)
    # 期待する2点は X[0], X[4]
    expected = X[[0, 4]]
    # 順序も保持される（init_index=0 → 次は反対側が最遠）
    assert np.allclose(out, expected, atol=1e-6)


# -----------------------------
# 単位球正規化の影響
# -----------------------------
def test_invariance_to_row_scaling_when_ensure_unit_sphere_true():
    X = _make_unit_sphere_data(N=256, C=8, seed=123)
    X_scaled = _scale_rows(X, seed=456)  # 行ごとのスケーリングを導入

    A = fps_downsample(X, ratio=0.1, ensure_unit_sphere=True, seed=7, return_numpy=True)
    B = fps_downsample(X_scaled, ratio=0.1, ensure_unit_sphere=True, seed=7, return_numpy=True)

    # 行スケーリングは cos 幾何では不変 → 選抜結果（元スケールの値）は比例するが、
    # X と X_scaled は行方向に定数倍なので、単位方向は一致。
    # ここでは方向一致を評価：正規化して比較
    def unit(v): return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    assert np.allclose(unit(A), unit(B), atol=1e-5)


def test_no_difference_when_already_unit_and_ensure_unit_sphere_toggled():
    # すでに単位球 → ensure_unit_sphere のON/OFFで結果は同じになるはず
    X = _make_unit_sphere_data(N=200, C=6, seed=11)
    A = fps_downsample(X, ratio=0.15, ensure_unit_sphere=True, seed=77, return_numpy=True)
    B = fps_downsample(X, ratio=0.15, ensure_unit_sphere=False, seed=77, return_numpy=True)
    assert np.allclose(A, B, atol=1e-6)


# -----------------------------
# 空入力 / エラーハンドリング
# -----------------------------
def test_empty_input_numpy():
    X = np.empty((0, 8), dtype=np.float32)
    out = fps_downsample(X, ratio=0.3, return_numpy=True)
    assert isinstance(out, np.ndarray)
    assert out.shape == (0, 8)

def test_empty_input_torch():
    X = torch.empty((0, 5), dtype=torch.float32)
    out = fps_downsample(X, ratio=0.3, return_numpy=False)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (0, 5)

def test_invalid_dim_raises():
    with pytest.raises(ValueError):
        fps_downsample(np.zeros((3, 4, 5), dtype=np.float32))

def test_init_index_out_of_range_raises():
    X = _make_unit_sphere_data(N=10, C=3, seed=0)
    with pytest.raises(ValueError):
        fps_downsample(X, ratio=0.2, init_index=999)
