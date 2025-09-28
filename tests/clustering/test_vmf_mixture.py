from __future__ import annotations
import math
import os
import numpy as np
import torch
import pytest

# matplotlib はヘッドレス環境でも動くように 'Agg'
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from chemomae.clustering.vmf_mixture import (
        VMFMixture, elbow_vmf, _sample_vmf
    )
from chemomae.clustering.ops import plot_elbow_vmf


@torch.no_grad()
def _make_vmf_blobs(n_per=40, d=16, K=3, kappa=40.0, seed=0, device="cpu"):
    """
    vMF 混合から合成データを生成（各成分 n_per 点）。
    戻り値: X (N,d), true_mus (K,d)
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device(device)
    mus = torch.randn(K, d, generator=g).to(dev)
    mus = torch.nn.functional.normalize(mus, dim=1)
    kappas = torch.full((K,), float(kappa), device=dev)
    X = torch.empty(K * n_per, d, device=dev)
    for k in range(K):
        X[k * n_per : (k + 1) * n_per] = _sample_vmf(mus[k], kappas[k], n_per)
    X = torch.nn.functional.normalize(X, dim=1)
    return X, mus


def test_fit_predict_basic_properties_cpu():
    X, _ = _make_vmf_blobs(n_per=30, d=16, K=3, kappa=35.0, seed=1, device="cpu")
    model = VMFMixture(n_components=3, d=None, device="cpu", random_state=42, tol=1e-4, max_iter=100)
    model.fit(X)
    assert model._fitted is True
    assert model.mus.shape == (3, X.shape[1])
    # μ は単位ベクトル
    norms = model.mus.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    # κ は正
    assert float(model.kappas.min().item()) > 0.0
    # 予測
    labels = model.predict(X)
    probs = model.predict_proba(X)
    assert labels.shape == (X.shape[0],)
    assert probs.shape == (X.shape[0], 3)
    row_sum = probs.sum(dim=1)
    assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-4)


def test_predict_before_fit_raises_and_dim_mismatch():
    X, _ = _make_vmf_blobs(n_per=20, d=8, K=3, seed=0)
    m = VMFMixture(n_components=3, d=None, device="cpu")
    with pytest.raises(RuntimeError):
        _ = m.predict(X)

    m.fit(X)
    with pytest.raises(ValueError):
        _ = m.predict(torch.randn(X.size(0), X.size(1) + 1))


def test_save_and_load_consistency(tmp_path):
    X, _ = _make_vmf_blobs(n_per=25, d=12, K=3, seed=7)
    m1 = VMFMixture(n_components=3, d=None, device="cpu", random_state=0, max_iter=100).fit(X)
    ll1 = m1.loglik(X, average=True)
    bic1 = m1.bic(X)
    path = tmp_path / "vmf_model.pt"
    m1.save(str(path))

    m2 = VMFMixture.load(str(path), map_location="cpu")
    ll2 = m2.loglik(X, average=True)
    bic2 = m2.bic(X)

    # 誤差はごく小さいはず
    assert math.isfinite(ll1) and math.isfinite(ll2)
    assert math.isfinite(bic1) and math.isfinite(bic2)
    assert abs(ll1 - ll2) <= max(1e-5, 1e-6 * max(1.0, abs(ll1)))
    assert abs(bic1 - bic2) <= max(1e-5, 1e-6 * max(1.0, abs(bic1)))


def test_elbow_vmf_smoke_cpu():
    X, _ = _make_vmf_blobs(n_per=20, d=10, K=3, seed=3)
    # BIC を基準に 1..6 を走査（CPU）
    k_list, scores, K, idx, kappa = elbow_vmf(
        VMFMixture, X, device="cpu", k_max=6, chunk=None, verbose=False, random_state=0, criterion="bic"
    )
    assert isinstance(k_list, list) and isinstance(scores, list)
    assert len(k_list) == len(scores) == 6
    assert 1 <= K <= 6 and 0 <= idx < 6
    assert isinstance(kappa, float) or np.isscalar(kappa) or hasattr(kappa, "__float__")


def test_plot_elbow_vmf_smoke(tmp_path):
    ks = list(range(1, 7))
    # 適当な減少列（BIC の体裁）
    scores = [500.0, 410.0, 360.0, 340.0, 335.0, 334.0]
    K = 3
    idx = ks.index(K)
    out = tmp_path / "elbow_vmf_bic.png"
    plot_elbow_vmf(ks, scores, K, idx, criterion="bic")
    plt.savefig(out, dpi=120)
    assert os.path.exists(out)
