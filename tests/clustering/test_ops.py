import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # ヘッドレス環境向け
import matplotlib.pyplot as plt

from wavemae.clustering.ops import (
    l2_normalize_rows, cosine_similarity, cosine_dissimilarity,
    find_elbow_curvature, plot_elbow,
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
    assert isinstance(kappa, np.ndarray) and kappa.shape[0] == len(k_list)

    # 描画ヘルパのスモーク（保存まで）
    plot_elbow(k_list, inertias, K, idx)
    out = tmp_path / "elbow.png"
    plt.gcf().savefig(out)
    assert out.exists() and out.stat().st_size > 0
