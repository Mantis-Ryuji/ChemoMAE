import numpy as np
import pytest
import torch
import matplotlib
matplotlib.use("Agg")  # ヘッドレス環境向け
import matplotlib.pyplot as plt

from chemomae.clustering.ops import (
    l2_normalize_rows, cosine_similarity, cosine_dissimilarity,
    find_elbow_curvature, plot_elbow_ckm,
)


def test_l2_normalize_rows_shapes_and_norms():
    X = torch.randn(7, 5, dtype=torch.float32)
    Y = l2_normalize_rows(X)
    assert Y.shape == X.shape
    norms = Y.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_cosine_similarity_and_dissimilarity_bounds():
    A = l2_normalize_rows(torch.randn(10, 6))
    B = l2_normalize_rows(torch.randn(12, 6))
    S = cosine_similarity(A, B)
    D = cosine_dissimilarity(A, B)
    assert S.shape == (10, 12) and D.shape == (10, 12)
    # cos ∈ [-1,1]、1-cos ∈ [0,2]
    assert torch.all(S <= 1.0 + 1e-6) and torch.all(S >= -1.0 - 1e-6)
    assert torch.all(D <= 2.0 + 1e-6) and torch.all(D >= -1e-6)


def test_find_elbow_curvature_and_plot_elbow(tmp_path):
    # 非単調な列でも内部で単調化して曲率法が動く
    k_list = [1, 2, 3, 4, 5, 6]
    inertias = [0.9, 0.7, 0.75, 0.6, 0.59, 0.58]
    K, idx, kappa = find_elbow_curvature(k_list, inertias)
    assert 1 <= K <= 6 and 0 <= idx < len(k_list)
    assert isinstance(kappa, float)

    # 描画ヘルパのスモーク（保存まで）
    plot_elbow_ckm(k_list, inertias, K, idx)
    out = tmp_path / "elbow.png"
    plt.gcf().savefig(out)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.parametrize("length", [5, 6, 8])
def test_smoothing_window_is_bounded_for_even_and_odd_curves(length: int) -> None:
    k = list(range(1, length + 1))
    values = [1 / value for value in k]
    chosen, index, curvature = find_elbow_curvature(k, values, window_length=99)
    assert chosen == k[index]
    assert 0 < index < length - 1
    assert np.isfinite(curvature)


def test_nonuniform_spacing_uses_coordinate_aware_gradients() -> None:
    k = [1, 2, 4, 7, 11, 16]
    values = [0.9, 0.7, 0.5, 0.4, 0.35, 0.34]
    assert find_elbow_curvature(k, values, smooth=True) == find_elbow_curvature(k, values, smooth=False)


@pytest.mark.parametrize("k, values", [
    ([1, 2, 3], [0.9, 0.8]), ([1, 1, 3], [0.9, 0.8, 0.7]),
    ([3, 2, 1], [0.9, 0.8, 0.7]), ([1, 2, 3], [0.9, np.nan, 0.7]),
    ([1, 2.5, 3], [0.9, 0.8, 0.7]), ([0, 1, 2], [0.9, 0.8, 0.7]),
])
def test_invalid_elbow_curves_are_rejected(k: list[float], values: list[float]) -> None:
    with pytest.raises(ValueError):
        find_elbow_curvature(k, values)


@pytest.mark.parametrize("settings", [{"window_length": 4}, {"window_length": 0}, {"polyorder": 1}, {"polyorder": 5}])
def test_invalid_smoothing_settings_are_rejected(settings: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        find_elbow_curvature([1, 2, 3, 4, 5], [0.9, 0.8, 0.7, 0.6, 0.5], **settings)
