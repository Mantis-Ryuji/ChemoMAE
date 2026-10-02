import numpy as np
import pytest
import torch

from chemomae.preprocessing.snv import SNVScaler, snv


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("shape", [(5,), (2, 5)])
def test_numpy_matches_independent_sample_standardization(dtype: type, shape: tuple[int, ...]) -> None:
    x = np.array([0.25, 1.0, -0.5, 2.0, 4.0] * (1 if len(shape) == 1 else 2), dtype=dtype).reshape(shape)
    original = x.copy()
    output = snv(x, eps=1e-5)
    reference = x.astype(np.float64)
    reference = (reference - reference.mean(axis=-1, keepdims=True)) / (
        reference.std(axis=-1, ddof=1, keepdims=True) + 1e-5
    )
    expected_dtype = np.float64 if dtype == np.float64 else np.float32
    assert output.dtype == expected_dtype
    tolerance = 1e-12 if dtype == np.float64 else 2e-6
    np.testing.assert_allclose(output, reference, rtol=tolerance, atol=tolerance)
    np.testing.assert_array_equal(x, original)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_torch_and_numpy_share_statistics_and_precision_contract(dtype: torch.dtype) -> None:
    x = torch.tensor([[0.25, 1.0, -0.5, 2.0], [4.0, 3.0, 0.0, -2.0]], dtype=dtype)
    scaler = SNVScaler(eps=0.125, transform_stats=True)
    normalized, mean, scale = scaler.transform(x)
    # Use the actual represented input values, including any half quantization.
    numpy_input = x.to(dtype=torch.float64 if dtype == torch.float64 else torch.float32).numpy()
    expected, expected_mean, expected_scale = scaler.transform(numpy_input)
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    for actual, reference in ((normalized, expected), (mean, expected_mean), (scale, expected_scale)):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected_dtype
        assert actual.device == x.device
        torch.testing.assert_close(actual, torch.from_numpy(reference), rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(snv(x, eps=0.125), normalized)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("shape", [(1,), (3, 1), (7,), (2, 7)])
def test_constant_and_length_one_spectra_have_zero_output_and_positive_scale(
    backend: str, shape: tuple[int, ...]
) -> None:
    x = np.full(shape, 0.1, dtype=np.float32)
    scaler = SNVScaler(transform_stats=True)
    if backend == "torch":
        x = torch.from_numpy(x)
    normalized, mean, scale = scaler.transform(x)
    expected_stats_shape = () if len(shape) == 1 else (shape[0], 1)
    assert mean.shape == expected_stats_shape
    assert scale.shape == expected_stats_shape
    reconstructed = scaler.inverse_transform(normalized, mu=mean, sd=scale)
    if backend == "numpy":
        np.testing.assert_array_equal(normalized, np.zeros(shape, dtype=np.float32))
        np.testing.assert_allclose(scale, 1e-12, rtol=1e-6, atol=0)
        np.testing.assert_array_equal(reconstructed, x)
    else:
        torch.testing.assert_close(normalized, torch.zeros_like(x), rtol=0, atol=0)
        torch.testing.assert_close(scale, torch.full_like(scale, 1e-12), rtol=0, atol=0)
        torch.testing.assert_close(reconstructed, x, rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("copy", [False, True])
def test_noncontiguous_input_round_trips_with_effective_scale(backend: str, copy: bool) -> None:
    x = np.array([[1.0, 2.0, 4.0, 8.0], [-2.0, 1.0, 3.0, 9.0]], dtype=np.float64)[:, ::2]
    if backend == "torch":
        x = torch.tensor([[1.0, 2.0, 4.0, 8.0], [-2.0, 1.0, 3.0, 9.0]], dtype=torch.float64)[:, ::2]
        original = x.clone()
    else:
        original = x.copy()
    scaler = SNVScaler(eps=0.25, copy=copy, transform_stats=True)
    normalized, mean, scale = scaler.transform(x)
    reconstructed = scaler.inverse_transform(normalized, mu=mean, sd=scale)
    if backend == "numpy":
        np.testing.assert_allclose(reconstructed, original, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(x, original)
        np.testing.assert_allclose(scale, original.std(axis=1, ddof=1, keepdims=True) + 0.25)
    else:
        torch.testing.assert_close(reconstructed, original, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(x, original, rtol=0, atol=0)
        torch.testing.assert_close(scale, original.std(dim=1, correction=1, keepdim=True) + 0.25)


def test_float64_precision_is_not_reduced_to_float32() -> None:
    x = np.array([1e8 + 0.125, 1e8 + 0.25, 1e8 + 0.5], dtype=np.float64)
    numpy_output = snv(x)
    torch_output = snv(torch.from_numpy(x))
    assert numpy_output.dtype == np.float64
    assert torch_output.dtype == torch.float64
    np.testing.assert_allclose(numpy_output.std(ddof=1), 1.0, rtol=1e-10)
    torch.testing.assert_close(torch_output, torch.from_numpy(numpy_output), rtol=1e-12, atol=1e-12)


def test_torch_transform_never_detaches_or_moves_data_to_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    x = torch.tensor([[0.25, 1.0, 2.0, 4.0]], dtype=torch.float64, requires_grad=True)

    def forbidden_conversion(*args: object, **kwargs: object) -> None:
        raise AssertionError("Torch SNV must not detach, call cpu(), or convert to NumPy.")

    with monkeypatch.context() as context:
        for name in ("detach", "cpu", "numpy"):
            context.setattr(torch.Tensor, name, forbidden_conversion)
        scaler = SNVScaler(transform_stats=True)
        normalized, mean, scale = scaler.transform(x)
        reconstructed = scaler.inverse_transform(normalized, mu=mean, sd=scale)
    assert normalized.requires_grad and mean.requires_grad and scale.requires_grad
    torch.testing.assert_close(reconstructed, x, rtol=1e-12, atol=1e-12)
    assert torch.autograd.gradcheck(snv, (x,))


def test_constant_torch_spectra_have_finite_gradients() -> None:
    x = torch.full((2, 3), 0.1, dtype=torch.float64, requires_grad=True)
    snv(x)[0, 0].backward()
    assert x.grad is not None and bool(torch.isfinite(x.grad).all())


def test_torch_rejects_sparse_and_meta_inputs() -> None:
    sparse = torch.sparse_coo_tensor([[0], [1]], [1.0], size=(2, 3))
    for value in (sparse, torch.empty((2, 3), device="meta")):
        with pytest.raises(TypeError, match="dense tensor with actual data"):
            snv(value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_transform_and_statistics_stay_on_input_device() -> None:
    x = torch.tensor([[1.0, 3.0, 2.0], [4.0, 0.0, -2.0]], device="cuda", requires_grad=True)
    scaler = SNVScaler(transform_stats=True)
    normalized, mean, scale = scaler.transform(x)
    for output in (normalized, mean, scale):
        assert output.device == x.device
    reconstructed = scaler.inverse_transform(normalized, mu=mean, sd=scale)
    torch.testing.assert_close(reconstructed, x)
    with pytest.raises(ValueError, match="must be on"):
        scaler.inverse_transform(normalized, mu=mean.cpu(), sd=scale)
    (normalized * torch.tensor([[1.0, 0.0, -1.0]], device=x.device)).sum().backward()
    assert x.grad is not None and bool(torch.isfinite(x.grad).all())


@pytest.mark.parametrize("shape", [(), (0,), (0, 3), (3, 0), (2, 3, 4)])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_invalid_shapes_are_rejected(shape: tuple[int, ...], backend: str) -> None:
    x = np.zeros(shape, dtype=np.float32)
    if backend == "torch":
        x = torch.from_numpy(x)
    with pytest.raises(ValueError, match="shape|nonempty"):
        snv(x)


@pytest.mark.parametrize("dtype", [np.int32, np.bool_, np.complex64])
def test_numpy_rejects_nonfloating_input(dtype: type) -> None:
    with pytest.raises(TypeError, match="requires float"):
        snv(np.ones((2, 3), dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.int32, torch.bool, torch.complex64])
def test_torch_rejects_nonfloating_input(dtype: torch.dtype) -> None:
    with pytest.raises(TypeError, match="requires float"):
        snv(torch.ones((2, 3), dtype=dtype))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_nonfinite_input_is_rejected(value: float, backend: str) -> None:
    x = np.array([1.0, value], dtype=np.float32)
    if backend == "torch":
        x = torch.from_numpy(x)
    with pytest.raises(ValueError, match="NaN or infinity"):
        snv(x)


@pytest.mark.parametrize("eps", [0.0, -1e-6, np.nan, np.inf])
def test_invalid_eps_is_rejected(eps: float) -> None:
    with pytest.raises(ValueError, match="eps"):
        SNVScaler(eps=eps)
    with pytest.raises(ValueError, match="eps"):
        snv(np.array([1.0, 2.0], dtype=np.float32), eps=eps)


@pytest.mark.parametrize("eps", [True, "small"])
def test_nonreal_eps_is_rejected(eps: object) -> None:
    with pytest.raises(TypeError, match="eps"):
        SNVScaler(eps=eps)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_underflowed_eps_is_rejected_for_constant_float32_input(backend: str) -> None:
    x = np.array([1.0, 1.0], dtype=np.float32)
    if backend == "torch":
        x = torch.from_numpy(x)
    with pytest.raises(ValueError, match="positive scale"):
        snv(x, eps=1e-100)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("scale", [0.0, -1.0, np.nan, np.inf])
def test_inverse_rejects_invalid_scale(backend: str, scale: float) -> None:
    normalized = np.ones((2, 3), dtype=np.float32)
    if backend == "torch":
        normalized = torch.from_numpy(normalized)
    with pytest.raises(ValueError, match="statistics"):
        SNVScaler().inverse_transform(normalized, mu=0.0, sd=scale)


def test_inverse_rejects_ambiguous_statistics_shape_and_backend_mixing() -> None:
    numpy_input = np.ones((3, 3), dtype=np.float32)
    torch_input = torch.from_numpy(numpy_input)
    scaler = SNVScaler()
    with pytest.raises(ValueError, match="mu must have shape"):
        scaler.inverse_transform(numpy_input, mu=np.ones(3, dtype=np.float32), sd=1.0)
    with pytest.raises(ValueError, match="mu must have shape"):
        scaler.inverse_transform(torch_input, mu=torch.ones(3), sd=1.0)
    with pytest.raises(TypeError, match="Torch tensor"):
        scaler.inverse_transform(torch_input, mu=np.zeros((3, 1), dtype=np.float32), sd=1.0)
    with pytest.raises(TypeError, match="NumPy array"):
        scaler.inverse_transform(numpy_input, mu=torch.zeros((3, 1)), sd=1.0)


def test_stateless_fit_transform_and_parameter_contract() -> None:
    x = np.array([[1.0, 3.0, 4.0], [-1.0, 0.0, 2.0]], dtype=np.float32)
    scaler = SNVScaler(copy=False)
    assert scaler.fit(x) is scaler
    np.testing.assert_array_equal(scaler.fit_transform(x), scaler.transform(x))
    assert scaler.get_params(deep=False) == {"eps": 1e-12, "copy": False, "transform_stats": False}
    assert scaler.set_params(eps=1e-5, transform_stats=True) is scaler
    assert isinstance(scaler.fit_transform(x), tuple)
    parameters = scaler.get_params()
    with pytest.raises(ValueError, match="Unknown"):
        scaler.set_params(unknown=True)
    with pytest.raises(ValueError, match="eps"):
        scaler.set_params(eps=-1.0, copy=True)
    assert scaler.get_params() == parameters


def test_scaler_parameters_and_input_framework_are_validated() -> None:
    with pytest.raises(TypeError, match="copy"):
        SNVScaler(copy="yes")
    with pytest.raises(TypeError, match="transform_stats"):
        SNVScaler(transform_stats=1)
    with pytest.raises(TypeError, match="NumPy array or a Torch tensor"):
        snv([1.0, 2.0, 3.0])
