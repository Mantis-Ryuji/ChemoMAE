from __future__ import annotations
import math
import os
from decimal import Decimal, localcontext
from functools import lru_cache
import numpy as np
from scipy import integrate, special
import torch
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chemomae.clustering.vmf_mixture import (
    VMFMixture, elbow_vmf, vmf_logC, vmf_bessel_ratio
)
from chemomae.clustering.ops import plot_elbow_vmf


@torch.no_grad()
def _sample_vmf_like(mu: torch.Tensor, kappa: float, n: int, *, generator=None):
    """
    Very simple vMF-like sampler for testing.
    Samples Gaussian noise around mu and projects to the unit sphere.
    """
    d = mu.numel()
    if generator is None:
        eps = torch.randn(n, d, device=mu.device)
    else:
        eps = torch.randn(n, d, device=mu.device, generator=generator)

    x = mu.unsqueeze(0) + eps / max(float(kappa), 1e-6)
    return torch.nn.functional.normalize(x, dim=1)


@torch.no_grad()
def _make_vmf_blobs(n_per=40, d=16, K=3, kappa=40.0, seed=0, device="cpu"):
    """
    vMF 風の混合分布から合成データを生成（各成分 n_per 点）。
    戻り値: X (N,d), true_mus (K,d)
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device(device)

    mus = torch.randn(K, d, generator=g).to(dev)
    mus = torch.nn.functional.normalize(mus, dim=1)

    X = torch.empty(K * n_per, d, device=dev)
    for k in range(K):
        X[k * n_per : (k + 1) * n_per] = _sample_vmf_like(
            mus[k], kappa, n_per, generator=g
        )

    X = torch.nn.functional.normalize(X, dim=1)
    return X, mus


# ============================================================
# Tests
# ============================================================
def test_fit_predict_basic_properties_cpu():
    X, _ = _make_vmf_blobs(n_per=30, d=16, K=3, kappa=35.0, seed=1, device="cpu")
    model = VMFMixture(
        n_components=3,
        d=None,
        device="cpu",
        random_state=42,
        tol=1e-4,
        max_iter=100,
    )
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
    m1 = VMFMixture(
        n_components=3,
        d=None,
        device="cpu",
        random_state=0,
        max_iter=100,
    ).fit(X)

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
    for name in ("mus", "kappas", "logpi", "_logC"):
        torch.testing.assert_close(getattr(m1, name), getattr(m2, name), rtol=0, atol=0)
    torch.testing.assert_close(m1.predict_proba(X), m2.predict_proba(X), rtol=0, atol=0)
    assert torch.equal(m1._g.get_state(), m2._g.get_state())
    assert m1.lower_bound_ == m2.lower_bound_
    assert m1.converged_ == m2.converged_
    assert m1.stop_reason_ == m2.stop_reason_


def test_elbow_vmf_smoke_cpu():
    X, _ = _make_vmf_blobs(n_per=20, d=10, K=3, seed=3)

    # BIC を基準に 1..6 を走査（CPU）
    k_list, scores, K, idx, kappa = elbow_vmf(
        VMFMixture,
        X,
        device="cpu",
        k_max=6,
        chunk=None,
        verbose=False,
        random_state=0,
        criterion="bic",
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


@lru_cache(maxsize=None)
def _decimal_reference(d: int, kappa: float) -> tuple[float, float]:
    """Independent 80-digit defining-series reference, including ive underflow.

    Only used through kappa=1000. Test data are synthetic, not fitted vMF draws.
    """
    with localcontext() as ctx:
        ctx.prec = 80
        nu = Decimal(d) / 2 - 1
        x = Decimal(str(kappa))

        def series(order: Decimal) -> Decimal:
            term = total = Decimal(1)
            for m in range(1, 5000):
                term *= (x * x / 4) / (Decimal(m) * (order + m))
                total += term
                if term < total * Decimal("1e-70"):
                    return total
            raise AssertionError("Decimal reference did not converge")

        total = series(nu)
        uniform = math.lgamma(d / 2) - math.log(2) - (d / 2) * math.log(math.pi)
        logc = float(Decimal(str(uniform)) - total.ln())
        ratio = float(x / (2 * (nu + 1)) * series(nu + 1) / total)
        return logc, ratio


@pytest.mark.parametrize("d", [2, 3, 16, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_numerical_helpers_against_decimal_series(d: int, dtype: torch.dtype) -> None:
    values = torch.tensor(
        [0, 1e-12, 1e-6, 0.01, 0.9999, 1, 1.0001, 1.9999, 2, 2.0001,
         11.9999, 12, 12.0001, 100, 1000],
        dtype=dtype,
    )
    reference = [_decimal_reference(d, float(x)) for x in values]
    logc = vmf_logC(d, values)
    ratio = vmf_bessel_ratio(torch.tensor(d / 2 - 1, dtype=dtype), values)
    rtol, atol = (2e-7, 3e-5) if dtype == torch.float32 else (2e-12, 2e-10)
    torch.testing.assert_close(logc, torch.tensor([r[0] for r in reference], dtype=dtype), rtol=rtol, atol=atol)
    torch.testing.assert_close(ratio, torch.tensor([r[1] for r in reference], dtype=dtype), rtol=rtol, atol=1e-14)
    assert logc.shape == ratio.shape == values.shape
    assert logc.dtype == ratio.dtype == dtype
    assert ratio[0] == 0
    assert torch.isfinite(logc).all() and torch.isfinite(ratio).all()


@pytest.mark.parametrize("d", [16, 256])
def test_large_concentrations_against_scaled_bessel(d: int) -> None:
    k = torch.tensor([1e4, 1e6, 2e8], dtype=torch.float64)
    x = k.numpy()
    nu = d / 2 - 1
    iv0, iv1 = special.ive(nu, x), special.ive(nu + 1, x)
    assert np.all(np.isfinite(iv0) & (iv0 > 0) & np.isfinite(iv1) & (iv1 > 0))
    reference = nu * np.log(x) - d / 2 * math.log(2 * math.pi) - np.log(iv0) - x
    np.testing.assert_allclose(vmf_logC(d, k).numpy(), reference, rtol=2e-14, atol=1e-7)
    np.testing.assert_allclose(vmf_bessel_ratio(torch.tensor(nu), k).numpy(), iv1 / iv0, rtol=2e-12)


def test_three_dimensional_closed_form_and_small_ratio_coefficient() -> None:
    k = torch.tensor([0.01, 2, 12, 100, 1e4, 1e6, 2e8], dtype=torch.float64)
    # C_3(k) = k / (4*pi*sinh(k)); use a scaled form without exp(k).
    reference = torch.log(k) - math.log(2 * math.pi) - k - torch.log(-torch.expm1(-2 * k))
    torch.testing.assert_close(vmf_logC(3, k), reference, rtol=2e-14, atol=1e-7)
    ratio = 1 / torch.tanh(k) - 1 / k
    torch.testing.assert_close(vmf_bessel_ratio(torch.tensor(0.5), k), ratio, rtol=2e-10, atol=1e-14)
    small = torch.tensor(0.01, dtype=torch.float64)
    a, b = 16.0, 18.0
    cubic = small / a * (1 - small.square() / (a * b))
    torch.testing.assert_close(vmf_bessel_ratio(torch.tensor(7.0), small), cubic, rtol=1e-12, atol=0)


@pytest.mark.parametrize("d", [3, 16, 256])
@pytest.mark.parametrize("kappa", [0.0, 2.0, 12.0, 100.0])
def test_density_integrates_to_one(d: int, kappa: float) -> None:
    logc = vmf_logC(d, torch.tensor(kappa, dtype=torch.float64)).item()
    log_area = math.log(2) + (d - 1) / 2 * math.log(math.pi) - math.lgamma((d - 1) / 2)

    def integrand(t: float) -> float:
        return math.exp(logc + kappa * t + log_area + (d - 3) / 2 * math.log1p(-t * t))

    mass, error = integrate.quad(integrand, -1, 1, epsabs=1e-10, epsrel=1e-10)
    assert error < 1e-8
    assert mass == pytest.approx(1.0, abs=2e-9)


def test_bessel_helper_broadcasting_and_invalid_inputs() -> None:
    nu = torch.tensor([[0.0], [7.0], [127.0]], dtype=torch.float64)
    k = torch.tensor([0.0, 0.01, 12.0], dtype=torch.float64)
    result = vmf_bessel_ratio(nu, k)
    assert result.shape == (3, 3)
    for row, d in enumerate([2, 16, 256]):
        expected = torch.tensor([_decimal_reference(d, float(x))[1] for x in k], dtype=k.dtype)
        torch.testing.assert_close(result[row], expected, rtol=2e-12, atol=1e-14)
    for invalid in [-1.0, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="kappa"):
            vmf_logC(16, torch.tensor(invalid))
        with pytest.raises(ValueError, match="kappa"):
            vmf_bessel_ratio(torch.tensor(7.0), torch.tensor(invalid))
        with pytest.raises(ValueError, match="nu"):
            vmf_bessel_ratio(torch.tensor(invalid), torch.tensor(1.0))
    with pytest.raises(ValueError, match="d must"):
        vmf_logC(1, torch.tensor(1.0))


def test_numerical_helpers_return_detached_tensors() -> None:
    k = torch.tensor([0.01, 12.0], dtype=torch.float64, requires_grad=True)
    nu = torch.tensor(7.0, dtype=torch.float64, requires_grad=True)
    assert not vmf_logC(16, k).requires_grad
    assert not vmf_bessel_ratio(nu, k).requires_grad


def _fixed_model(d: int = 16, dtype: torch.dtype = torch.float64) -> VMFMixture:
    model = VMFMixture(3, d=d, device="cpu", dtype=dtype)
    model._allocate_buffers()
    model.mus.copy_(torch.eye(d, dtype=dtype)[:3])
    model.kappas.copy_(torch.tensor([2, 12, 80], dtype=dtype))
    model.logpi.copy_(torch.tensor([0.2, 0.3, 0.5], dtype=dtype).log())
    model._refresh_logC()
    model._fitted = True
    return model


@pytest.mark.parametrize("d", [16, 256])
def test_responsibilities_likelihood_and_component_permutation(d: int) -> None:
    model = _fixed_model(d)
    X = torch.randn(17, d, dtype=torch.float64, generator=torch.Generator().manual_seed(8))
    X = torch.nn.functional.normalize(X, dim=1)
    logc = torch.tensor([_decimal_reference(d, float(k))[0] for k in model.kappas], dtype=X.dtype)
    logpost = X @ model.mus.T * model.kappas + logc + model.logpi
    expected = torch.softmax(logpost, dim=1)
    actual = model.predict_proba(X)
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=1e-13)
    torch.testing.assert_close(actual.sum(1), torch.ones(X.size(0), dtype=X.dtype))
    assert model.loglik(X) == pytest.approx(torch.logsumexp(logpost, 1).sum().item(), abs=1e-9)
    permutation = torch.tensor([2, 0, 1])
    for name in ("mus", "kappas", "logpi"):
        getattr(model, name).copy_(getattr(model, name)[permutation])
    model._refresh_logC()
    torch.testing.assert_close(model.predict_proba(X), expected[:, permutation], rtol=2e-10, atol=1e-13)
    model.kappas.fill_(12)
    model.logpi.zero_()
    model._refresh_logC()
    assert torch.equal(model.predict(X), (X @ model.mus.T).argmax(1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_chunked_statistics_and_predictions_with_fixed_parameters(dtype: torch.dtype) -> None:
    model = _fixed_model(dtype=dtype)
    X = torch.randn(37, 16, dtype=dtype, generator=torch.Generator().manual_seed(3))
    gamma, lb, Nk, Sk = model._e_step_chunk(X, None)
    rtol, atol = (3e-5, 3e-5) if dtype == torch.float32 else (1e-11, 1e-11)
    for chunk in [1, 7, 100]:
        chunk_gamma, chunk_lb, chunk_Nk, chunk_Sk = model._e_step_chunk(X, chunk)
        for actual, expected in [(chunk_gamma, gamma), (chunk_Nk, Nk), (chunk_Sk, Sk)]:
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        assert chunk_lb == pytest.approx(lb, rel=rtol, abs=atol)
        assert model.loglik(X, chunk=chunk) == chunk_lb
        torch.testing.assert_close(model.predict_proba(X, chunk=chunk), gamma, rtol=rtol, atol=atol)


@pytest.mark.parametrize("chunk", [None, 7])
@pytest.mark.parametrize("max_iter", [1, 5])
def test_final_likelihood_matches_final_parameters(chunk: int | None, max_iter: int) -> None:
    X, _ = _make_vmf_blobs(n_per=8, K=2, seed=4)
    model = VMFMixture(2, device="cpu", max_iter=max_iter, tol=0).fit(X, chunk=chunk)
    assert 1 <= model.n_iter_ <= max_iter
    assert model.lower_bound_ == model.loglik(X, chunk=chunk)
    assert model.stop_reason_ in {"tol", "likelihood_decreased", "max_iter"}
    assert model.converged_ == (model.stop_reason_ == "tol")
    assert model.num_params() == 2 * X.size(1) + 1


@pytest.mark.parametrize(
    "likelihoods, reason, converged, iterations",
    [([10.0, 10.0000001], "tol", True, 1),
     ([10.0, 9.9999999], "likelihood_decreased", False, 1),
     ([10.0, 11.0, 12.0], "max_iter", False, 2)],
)
def test_stop_reasons_distinguish_small_decrease(
    monkeypatch: pytest.MonkeyPatch, likelihoods: list[float], reason: str,
    converged: bool, iterations: int,
) -> None:
    model = VMFMixture(1, device="cpu", max_iter=2, tol=1e-4)
    values = iter(likelihoods)

    def e_step(
        X: torch.Tensor, chunk: int | None, *, return_gamma: bool = True,
    ) -> tuple[None, float, torch.Tensor, torch.Tensor]:
        return None, next(values), torch.ones(1), torch.tensor([[1.0, 0.0]])

    monkeypatch.setattr(model, "_e_step_chunk", e_step)
    model.fit(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert model.stop_reason_ == reason
    assert model.converged_ is converged
    assert model.n_iter_ == iterations
    assert model.lower_bound_ == likelihoods[-1]


def test_antipodal_fit_and_empty_component_preserve_valid_directions() -> None:
    X = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64)
    model = VMFMixture(1, device="cpu", dtype=X.dtype, max_iter=3).fit(X)
    torch.testing.assert_close(model.mus.norm(dim=1), torch.ones(1, dtype=X.dtype))
    assert model.kappas.item() == model.kappa_min
    assert model.lower_bound_ == model.loglik(X)

    model = _fixed_model()
    directions, kappas = model.mus.clone(), model.kappas.clone()
    Nk = torch.tensor([2.0, 0.0, 1e-12], dtype=torch.float64)
    Sk = torch.zeros(3, 16, dtype=torch.float64)
    Sk[2, 2] = 5e-13
    model._m_step_from_stats(Nk, Sk)
    torch.testing.assert_close(model.mus, directions, rtol=0, atol=0)
    assert model.kappas[0] == model.kappa_min
    assert model.kappas[1] == kappas[1]
    assert model.kappas[2] == pytest.approx(0.5 * (16 - 0.25) / (0.75 + 1e-8))
    assert torch.isfinite(model._logC).all()
    weights = model.logpi.softmax(0)
    assert weights[1] > 0 and weights[2] > weights[1]
    assert weights.sum().item() == pytest.approx(1)


@pytest.mark.parametrize("d", [16, 256])
def test_closed_form_concentration_residual_against_reference(d: int) -> None:
    resultant = torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99, 0.999], dtype=torch.float64)
    model = VMFMixture(len(resultant), d=d, device="cpu", dtype=resultant.dtype)
    model._allocate_buffers()
    model.mus.zero_()
    model.mus[:, 0] = 1
    model.kappas.fill_(10)
    stats = torch.zeros(len(resultant), d, dtype=resultant.dtype)
    stats[:, 0] = resultant
    model._m_step_from_stats(torch.ones_like(resultant), stats)
    k = model.kappas.numpy()
    nu = d / 2 - 1
    actual = special.ive(nu + 1, k) / special.ive(nu, k)
    assert np.isfinite(actual).all()
    # The retained update is approximate; this is a residual bound, not exact EM.
    np.testing.assert_allclose(actual, resultant.numpy(), rtol=0, atol=5e-3)


def test_nonzero_resultant_does_not_underflow_to_zero() -> None:
    model = _fixed_model()
    Nk = torch.tensor([1.0, 1.0, 1e-200], dtype=torch.float64)
    Sk = model.mus.clone() * Nk[:, None] * 0.5
    model._m_step_from_stats(Nk, Sk)
    torch.testing.assert_close(model.kappas, model.kappas[0].expand(3))
    torch.testing.assert_close(model.mus, torch.eye(16, dtype=torch.float64)[:3])


@pytest.mark.parametrize("bad", [
    torch.empty(0, 16), torch.ones(2, 1), torch.ones(16), torch.zeros(2, 16),
    torch.full((2, 16), float("nan")), torch.full((2, 16), float("inf")),
])
def test_invalid_inputs_rejected_for_fit_and_inference(bad: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        VMFMixture(2, device="cpu").fit(bad, chunk=1)
    model = _fixed_model()
    for method in (model.predict, model.predict_proba, model.loglik, model.bic):
        with pytest.raises(ValueError):
            method(bad, chunk=1)


@pytest.mark.parametrize("chunk", [0, -1, 1.5])
def test_invalid_chunk_rejected_before_initialization(chunk: int | float) -> None:
    model = VMFMixture(2, device="cpu")
    state = model._g.get_state().clone()
    with pytest.raises(ValueError, match="chunk"):
        model.fit(torch.eye(3), chunk=chunk)
    assert torch.equal(model._g.get_state(), state)
    assert model.mus.numel() == 0


@pytest.mark.parametrize("kwargs", [
    {"n_components": 0}, {"n_components": 1.5}, {"d": 1}, {"max_iter": 0},
    {"max_iter": 1.5}, {"tol": -1}, {"tol": float("nan")},
    {"kappa_min": 0}, {"kappa_min": 1e-50}, {"kappa_min": float("inf")}, {"kappa_init": float("inf")},
    {"kappa_init": 1e-8}, {"dtype": torch.float16},
])
def test_invalid_constructor_arguments(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        VMFMixture(**{"n_components": 2, "device": "cpu", **kwargs})


def test_normalization_handles_large_and_tiny_nonzero_rows() -> None:
    model = _fixed_model(dtype=torch.float32)
    X = torch.eye(16, dtype=torch.float32)[:3]
    expected = model.predict_proba(X)
    for scale in [1e-30, 1e30]:
        torch.testing.assert_close(model.predict_proba(X * scale), expected)


@pytest.mark.parametrize("location", ["cpu", torch.device("cpu")])
def test_cpu_load_overrides_saved_device(tmp_path, location: str | torch.device) -> None:
    model = _fixed_model()
    state = model.state_dict_vmf()
    state["device"] = "cuda:0"
    path = tmp_path / "cuda_metadata.pt"
    torch.save(state, path)
    restored = VMFMixture.load(path, map_location=location)
    assert restored.device == torch.device("cpu")
    for name in ("mus", "kappas", "logpi", "_logC"):
        assert getattr(restored, name).device.type == "cpu"
    assert torch.equal(model._g.get_state(), restored._g.get_state())
    torch.testing.assert_close(model.predict_proba(torch.eye(16)), restored.predict_proba(torch.eye(16)))


def test_legacy_load_refreshes_normalizer_and_invalidates_old_likelihood(tmp_path) -> None:
    model = _fixed_model()
    state = model.state_dict_vmf()
    for name in ("numerics_version", "converged_", "stop_reason_"):
        state.pop(name)
    state["_logC"].fill_(12345)
    state["lower_bound_"] = 12345.0
    path = tmp_path / "legacy.pt"
    torch.save(state, path)
    restored = VMFMixture.load(path, map_location="cpu")
    torch.testing.assert_close(restored._logC, model._logC)
    assert math.isnan(restored.lower_bound_)
    assert restored.converged_ is False and restored.stop_reason_ is None
    assert math.isfinite(restored.loglik(torch.eye(16)))


def test_invalid_legacy_direction_is_not_silently_loaded(tmp_path) -> None:
    state = _fixed_model().state_dict_vmf()
    state.pop("numerics_version")
    state["mus"][0].zero_()
    path = tmp_path / "invalid_legacy.pt"
    torch.save(state, path)
    with pytest.raises(ValueError, match="unit vectors"):
        VMFMixture.load(path, map_location="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("chunk", [None, 7])
def test_cuda_kmeans_initialization_fit_and_cpu_restore(tmp_path, chunk: int | None) -> None:
    # Generate on CPU so the fixture does not mix a CPU RNG with CUDA tensors.
    X = torch.randn(24, 16, generator=torch.Generator().manual_seed(9))
    model = VMFMixture(3, device="cuda", random_state=42, max_iter=2)
    repeat = VMFMixture(3, device="cuda", random_state=42, max_iter=2)
    model._init_params(X, chunk=chunk)
    repeat._init_params(X, chunk=chunk)
    torch.testing.assert_close(model.mus, repeat.mus, rtol=0, atol=0)
    assert torch.equal(model._g.get_state(), repeat._g.get_state())
    model.fit(X, chunk=chunk)
    assert model.lower_bound_ == model.loglik(X, chunk=chunk)
    assert torch.isfinite(model.predict_proba(X)).all()
    path = tmp_path / "cuda_fit.pt"
    model.save(path)
    restored = VMFMixture.load(path, map_location="cpu")
    torch.testing.assert_close(restored.mus, model.mus.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(restored.kappas, model.kappas.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(restored.logpi, model.logpi.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(restored.predict_proba(X), model.predict_proba(X).cpu(), rtol=3e-5, atol=3e-6)
    assert torch.equal(restored._g.get_state(), model._g.get_state())
