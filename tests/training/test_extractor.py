from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from chemomae.models.chemo_mae import ChemoMAE
from chemomae.training.augmenter import SpectraAugmenter, SpectraAugmenterConfig
from chemomae.training.extractor import Extractor, ExtractorConfig


def _make_tiny_model(dropout: float = 0.0) -> ChemoMAE:
    return ChemoMAE(
        seq_len=16,
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=dropout,
        latent_dim=8,
        n_patches=4,
        n_mask=1,
    )


class _SpyAugmenter(SpectraAugmenter):
    """Record activation and apply a deterministic perturbation."""

    def __init__(self, delta: float = 0.01) -> None:
        super().__init__(SpectraAugmenterConfig(shift_prob=0.0, noise_prob=0.0))
        self.delta = delta
        self.seen_training: list[bool] = []
        self.seen_grad: list[bool] = []
        self.seen_inference: list[bool] = []
        self.seen_cpu_autocast: list[bool] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen_training.append(self.training)
        self.seen_grad.append(torch.is_grad_enabled())
        self.seen_inference.append(torch.is_inference_mode_enabled())
        self.seen_cpu_autocast.append(torch.is_autocast_cpu_enabled())
        return x + self.delta if self.training else x


@pytest.mark.parametrize("representation", ["latent", "raw_latent", "normalized_latent", "cls"])
def test_stream_matches_aggregate_and_public_encode_in_loader_order(representation: str) -> None:
    torch.manual_seed(0)
    model = _make_tiny_model()
    x = torch.randn(5, 16, dtype=torch.float64)
    order = torch.tensor([3, 0, 4, 1, 2])
    loader = DataLoader(TensorDataset(x[order], order), batch_size=2, shuffle=False)
    extractor = Extractor(model, ExtractorConfig(representation=representation, output_dtype=torch.float64))

    streamed = list(extractor.iter_transform(loader))
    aggregate = extractor.transform(loader)

    assert [batch.shape[0] for batch in streamed] == [2, 2, 1]
    assert isinstance(aggregate, torch.Tensor)
    assert aggregate.dtype == torch.float64
    assert aggregate.device.type == "cpu"
    torch.testing.assert_close(torch.cat(streamed), aggregate, rtol=0, atol=0)
    model.eval()
    with torch.no_grad():
        expected = torch.cat([model.encode(batch[0].float(), representation=representation) for batch in loader])
    torch.testing.assert_close(aggregate, expected.double(), rtol=0, atol=0)


def test_iterator_is_lazy_and_restores_modes_before_each_yield_and_early_close() -> None:
    model = _make_tiny_model(dropout=0.5)
    model.train()
    model.encoder.eval()
    augmenter = _SpyAugmenter()
    augmenter.eval()
    flags = {module: module.training for module in model.modules()}
    consumed: list[int] = []

    def batches():
        for index in range(3):
            consumed.append(index)
            yield torch.ones(2, 16) * index

    stream = Extractor(model, augmenter=augmenter).iter_transform(batches())
    assert consumed == []
    assert torch.is_grad_enabled()
    first = next(stream)

    assert consumed == [0]
    assert {module: module.training for module in model.modules()} == flags
    assert augmenter.training is False
    assert augmenter.seen_training == [True]
    assert augmenter.seen_grad == [False]
    assert augmenter.seen_inference == [False]
    assert torch.is_grad_enabled()
    assert not torch.is_inference_mode_enabled()
    assert not first.requires_grad
    assert not first.is_inference()
    stream.close()
    assert consumed == [0]
    assert {module: module.training for module in model.modules()} == flags

    # Streaming outputs can be consumed by a trainable downstream head without
    # cloning an inference tensor or enabling gradients through the encoder.
    head = torch.nn.Linear(8, 2)
    head(first).square().mean().backward()
    assert head.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.parameters())


def test_outer_inference_and_autocast_are_restored_without_affecting_features() -> None:
    model = _make_tiny_model()
    augmenter = _SpyAugmenter()
    extractor = Extractor(model, augmenter=augmenter)
    x = torch.randn(2, 16)
    expected = extractor([x])
    augmenter.seen_cpu_autocast.clear()

    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        stream = extractor.iter_transform([x])
        actual = next(stream)
        assert torch.is_inference_mode_enabled()
        assert torch.is_autocast_cpu_enabled()
        assert not actual.is_inference()
        stream.close()

    assert augmenter.seen_cpu_autocast == [False]
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.is_inference_mode_enabled()
    assert torch.is_grad_enabled()


def test_dropout_is_disabled_without_changing_mixed_training_modes() -> None:
    torch.manual_seed(1)
    model = _make_tiny_model(dropout=0.5)
    model.train()
    model.decoder.eval()
    flags = {module: module.training for module in model.modules()}
    extractor = Extractor(model)
    x = torch.randn(3, 16)
    rng_state = torch.random.get_rng_state().clone()

    first = extractor([x])
    second = extractor([x])

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert {module: module.training for module in model.modules()} == flags
    assert torch.equal(torch.random.get_rng_state(), rng_state)


def test_closing_an_early_iterator_releases_its_progress_bar(monkeypatch) -> None:
    closed: list[bool] = []

    class Progress:
        def __init__(self, loader, **kwargs) -> None:
            self.batches = iter(loader)

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.batches)

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr("chemomae.training.extractor.tqdm", Progress)
    extractor = Extractor(_make_tiny_model(), ExtractorConfig(progress=True))
    stream = extractor.iter_transform([torch.ones(2, 16), torch.ones(2, 16)])
    next(stream)
    assert closed == []
    stream.close()
    assert closed == [True]


def test_exception_restores_model_and_augmenter_modes() -> None:
    class FailingAugmenter(_SpyAugmenter):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("augmentation failed")

    model = _make_tiny_model()
    model.train()
    model.encoder.eval()
    augmenter = FailingAugmenter()
    augmenter.eval()
    flags = {module: module.training for module in model.modules()}
    stream = Extractor(model, augmenter=augmenter).iter_transform([torch.ones(2, 16)])

    with pytest.raises(RuntimeError, match="augmentation failed"):
        next(stream)

    assert {module: module.training for module in model.modules()} == flags
    assert augmenter.training is False
    assert torch.is_grad_enabled()
    assert not torch.is_inference_mode_enabled()


def test_explicit_augmentation_changes_features_and_preserves_mode() -> None:
    model = _make_tiny_model()
    x = torch.randn(3, 16)
    augmenter = _SpyAugmenter(delta=0.3)
    augmenter.eval()
    actual = Extractor(model, augmenter=augmenter)([x[:2], x[2:]])
    model.eval()
    with torch.no_grad():
        expected = torch.cat([model.encode(x[:2] + 0.3), model.encode(x[2:] + 0.3)])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert augmenter.seen_training == [True, True]
    assert augmenter.training is False


@pytest.mark.parametrize("representation,dimension", [("latent", 8), ("raw_latent", 8), ("normalized_latent", 8), ("cls", 16)])
@pytest.mark.parametrize("output_type", ["tensor", "numpy"])
def test_empty_loaders_and_batches_have_requested_shape_and_dtype(
    representation: str, dimension: int, output_type: str
) -> None:
    extractor = Extractor(
        _make_tiny_model(),
        ExtractorConfig(representation=representation, output_type=output_type, output_dtype=torch.float64),
    )

    assert list(extractor.iter_transform([])) == []
    empty = extractor.transform([])
    empty_batch = list(extractor.iter_transform([torch.empty(0, 16)]))[0]

    assert empty.shape == empty_batch.shape == (0, dimension)
    if output_type == "numpy":
        assert isinstance(empty, np.ndarray)
        assert empty.dtype == empty_batch.dtype == np.float64
    else:
        assert isinstance(empty, torch.Tensor)
        assert empty.dtype == empty_batch.dtype == torch.float64
        assert empty.device.type == empty_batch.device.type == "cpu"


def test_numpy_stream_and_aggregate_saving_are_separate(tmp_path: Path) -> None:
    x = torch.randn(5, 16)
    model = _make_tiny_model()
    path = tmp_path / "features" / "latent.npy"
    extractor = Extractor(
        model,
        ExtractorConfig(output_type="numpy", output_dtype=torch.float64, save_path=path),
    )

    streamed = list(extractor.iter_transform([x[:2], x[2:]]))
    assert not path.exists()
    aggregate = extractor.extract([x[:2], x[2:]])
    assert isinstance(aggregate, np.ndarray)
    assert aggregate.dtype == np.float64
    np.testing.assert_array_equal(np.concatenate(streamed), aggregate)
    np.testing.assert_array_equal(np.load(path), aggregate)

    tensor_path = tmp_path / "latent.pt"
    tensor_extractor = Extractor(model, ExtractorConfig(save_path=tensor_path))
    tensor_result = tensor_extractor([x])
    saved = torch.load(tensor_path, weights_only=True)
    assert saved.device.type == "cpu"
    torch.testing.assert_close(saved, tensor_result)


@pytest.mark.parametrize(
    "batch,error,match",
    [
        ([], ValueError, "first item"),
        ("spectra", TypeError, "torch.Tensor"),
        (torch.ones(16), ValueError, "shape"),
        (torch.ones(2, 15), ValueError, "shape"),
        (torch.ones(2, 16, dtype=torch.int64), TypeError, "floating-point"),
    ],
)
def test_invalid_batches_fail_clearly(batch: object, error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        Extractor(_make_tiny_model())([batch])


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"representation": "tokens"}, "representation"),
        ({"output_type": "list"}, "output_type"),
        ({"amp_dtype": "fp32"}, "amp_dtype"),
        ({"output_dtype": torch.int64}, "output_dtype"),
        ({"output_type": "numpy", "output_dtype": torch.bfloat16}, "bfloat16"),
        ({"save_path": "features.npy", "output_dtype": torch.bfloat16}, "bfloat16"),
        ({"output_type": "numpy", "output_device": "cuda"}, "requires"),
    ],
)
def test_invalid_output_configuration_fails_clearly(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ExtractorConfig(**kwargs)


def test_unsupported_device_and_cpu_amp_fail_clearly() -> None:
    model = _make_tiny_model()
    with pytest.raises(ValueError, match="CPU and CUDA"):
        Extractor(model, ExtractorConfig(device="meta"))
    with pytest.raises(ValueError, match="requires a CUDA"):
        Extractor(model, ExtractorConfig(device="cpu", amp=True))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_input_is_rejected_before_encoding(value: float) -> None:
    spectra = torch.ones(2, 16)
    spectra[0, 3] = value
    with pytest.raises(ValueError, match="finite values"):
        Extractor(_make_tiny_model())([spectra])


def test_finite_input_that_overflows_model_dtype_is_rejected() -> None:
    spectra = torch.full((2, 16), 1e100, dtype=torch.float64)
    with pytest.raises(ValueError, match="model dtype"):
        Extractor(_make_tiny_model())([spectra])


def test_nonfinite_augmentation_restores_modes_and_does_not_save(tmp_path: Path) -> None:
    model = _make_tiny_model()
    model.train()
    model.encoder.eval()
    augmenter = _SpyAugmenter(delta=float("inf"))
    augmenter.eval()
    flags = {module: module.training for module in model.modules()}
    destination = tmp_path / "features.npy"
    with pytest.raises(ValueError, match="augmenter must return finite"):
        Extractor(
            model, ExtractorConfig(save_path=destination), augmenter=augmenter,
        )([torch.ones(2, 16)])
    assert {module: module.training for module in model.modules()} == flags
    assert not augmenter.training
    assert not destination.exists()


def test_feature_failure_restores_modes_and_prevents_saving(monkeypatch, tmp_path: Path) -> None:
    model = _make_tiny_model()
    model.train()
    model.encoder.eval()
    flags = {module: module.training for module in model.modules()}

    def nonfinite_encode(x: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.full((len(x), 8), float("nan"), device=x.device)

    monkeypatch.setattr(model, "encode", nonfinite_encode)
    destination = tmp_path / "features.npy"
    with pytest.raises(ValueError, match="nonfinite features"):
        Extractor(model, ExtractorConfig(save_path=destination))([torch.ones(2, 16)])
    assert {module: module.training for module in model.modules()} == flags
    assert not destination.exists()


def test_finite_features_that_overflow_output_dtype_are_rejected(monkeypatch) -> None:
    model = _make_tiny_model()

    def large_encode(x: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.full((len(x), 8), 1e10, device=x.device)

    monkeypatch.setattr(model, "encode", large_encode)
    with pytest.raises(ValueError, match="output_dtype"):
        Extractor(model, ExtractorConfig(output_dtype=torch.float16))([torch.ones(2, 16)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_tensor_outputs_can_remain_on_cuda_or_move_to_cpu() -> None:
    model = _make_tiny_model().cuda()
    x = torch.randn(3, 16)
    default_extractor = Extractor(model)
    gpu_batches = list(default_extractor.iter_transform([x[:2], x[2:]]))
    assert all(batch.device.type == "cuda" for batch in gpu_batches)

    cpu_extractor = Extractor(model, ExtractorConfig(output_device="cpu"))
    cpu_result = cpu_extractor.transform([x[:2], x[2:]])
    assert cpu_result.device.type == "cpu"
    torch.testing.assert_close(cpu_result, torch.cat(gpu_batches).cpu(), rtol=0, atol=0)

    numpy_result = Extractor(model, ExtractorConfig(output_type="numpy"))([x])
    assert isinstance(numpy_result, np.ndarray)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_device_moves_inside_outer_inference_mode_keep_model_parameters_ordinary() -> None:
    model = _make_tiny_model()
    extractor = Extractor(model, ExtractorConfig(device="cuda"))
    with torch.inference_mode():
        result = extractor([torch.ones(2, 16)])
    assert not result.is_inference()
    assert all(not parameter.is_inference() for parameter in model.parameters())
