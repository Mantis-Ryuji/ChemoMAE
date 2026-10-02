from __future__ import annotations

import pytest
import torch

from chemomae.training.augmenter import SpectraAugmenter, SpectraAugmenterConfig


def _snv_like_batch(
    batch_size: int = 8,
    seq_len: int = 16,
    seed: int = 0,
) -> torch.Tensor:
    """Build a zero-mean, unit-row-norm batch without drawing from global RNG."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, seq_len, generator=g, dtype=torch.float32)
    x = x - x.mean(dim=1, keepdim=True)
    x = x / torch.linalg.norm(x, dim=1, keepdim=True).clamp_min(1.0e-12)
    return x


def _row_cosine(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute row-wise cosine similarity."""
    denom = (
        torch.linalg.norm(x, dim=1) * torch.linalg.norm(y, dim=1)
    ).clamp_min(1.0e-12)
    return torch.sum(x * y, dim=1) / denom


def _row_angle_deg(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute row-wise angles in degrees."""
    cos = _row_cosine(x, y).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos))


def test_augmenter_eval_mode_returns_input_unchanged() -> None:
    x = _snv_like_batch()
    aug = SpectraAugmenter(
        SpectraAugmenterConfig(
            shift_prob=1.0,
            shift_delta_range=(-4.0, 4.0),
            noise_prob=1.0,
            noise_angle_deg_range=(0.5, 3.0),
            shuffle_order_per_batch=False,
        )
    )
    aug.eval()

    x_aug = aug(x)

    torch.testing.assert_close(x_aug, x)


def test_augmenter_train_mode_preserves_shape() -> None:
    x = _snv_like_batch()
    aug = SpectraAugmenter(
        SpectraAugmenterConfig(
            shift_prob=1.0,
            shift_delta_range=(-4.0, 4.0),
            noise_prob=1.0,
            noise_angle_deg_range=(0.5, 3.0),
            shuffle_order_per_batch=False,
        )
    )
    aug.train()

    torch.manual_seed(0)
    x_aug = aug(x)

    assert x_aug.shape == x.shape


def test_noise_only_preserves_row_norm() -> None:
    x = _snv_like_batch()
    aug = SpectraAugmenter(
        SpectraAugmenterConfig(
            shift_prob=0.0,
            shift_delta_range=(0.0, 0.0),
            noise_prob=1.0,
            noise_angle_deg_range=(2.0, 2.0),
            shuffle_order_per_batch=False,
            recenter_after_each_op=True,
            renorm_to_input_norm=True,
        )
    )
    aug.train()

    torch.manual_seed(0)
    x_aug = aug(x)

    x_norm = torch.linalg.norm(x, dim=1)
    x_aug_norm = torch.linalg.norm(x_aug, dim=1)
    torch.testing.assert_close(x_norm, x_aug_norm, rtol=1.0e-5, atol=1.0e-6)


def test_noise_only_preserves_row_mean_when_recenter_enabled() -> None:
    x = _snv_like_batch()
    aug = SpectraAugmenter(
        SpectraAugmenterConfig(
            shift_prob=0.0,
            noise_prob=1.0,
            noise_angle_deg_range=(2.0, 2.0),
            recenter_after_each_op=True,
            renorm_to_input_norm=True,
        )
    )
    aug.train()

    torch.manual_seed(0)
    x_aug = aug(x)

    x_aug_mean = x_aug.mean(dim=1)
    torch.testing.assert_close(
        x_aug_mean,
        torch.zeros_like(x_aug_mean),
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_shift_only_preserves_row_norm() -> None:
    x = _snv_like_batch()
    aug = SpectraAugmenter(
        SpectraAugmenterConfig(
            shift_prob=1.0,
            shift_delta_range=(2.0, 2.0),
            noise_prob=0.0,
            noise_angle_deg_range=(0.0, 0.0),
            shuffle_order_per_batch=False,
            recenter_after_each_op=True,
            renorm_to_input_norm=True,
        )
    )
    aug.train()

    torch.manual_seed(0)
    x_aug = aug(x)

    x_norm = torch.linalg.norm(x, dim=1)
    x_aug_norm = torch.linalg.norm(x_aug, dim=1)
    torch.testing.assert_close(x_norm, x_aug_norm, rtol=1.0e-5, atol=1.0e-6)


def test_shift_only_preserves_row_mean_when_recenter_enabled() -> None:
    x = _snv_like_batch()
    aug = SpectraAugmenter(
        SpectraAugmenterConfig(
            shift_prob=1.0,
            shift_delta_range=(2.0, 2.0),
            noise_prob=0.0,
            recenter_after_each_op=True,
            renorm_to_input_norm=True,
        )
    )
    aug.train()

    torch.manual_seed(0)
    x_aug = aug(x)

    x_aug_mean = x_aug.mean(dim=1)
    torch.testing.assert_close(
        x_aug_mean,
        torch.zeros_like(x_aug_mean),
        rtol=1.0e-5,
        atol=1.0e-6,
    )


def test_noise_and_shift_together_preserve_row_norm() -> None:
    x = _snv_like_batch()
    aug = SpectraAugmenter(
        SpectraAugmenterConfig(
            shift_prob=1.0,
            shift_delta_range=(2.0, 2.0),
            noise_prob=1.0,
            noise_angle_deg_range=(1.0, 1.0),
            shuffle_order_per_batch=False,
            recenter_after_each_op=True,
            renorm_to_input_norm=True,
        )
    )
    aug.train()

    torch.manual_seed(0)
    x_aug = aug(x)

    x_norm = torch.linalg.norm(x, dim=1)
    x_aug_norm = torch.linalg.norm(x_aug, dim=1)
    torch.testing.assert_close(x_norm, x_aug_norm, rtol=1.0e-5, atol=1.0e-6)


def test_zero_probability_keeps_input_unchanged_even_in_train_mode() -> None:
    x = _snv_like_batch()
    aug = SpectraAugmenter(
        SpectraAugmenterConfig(
            shift_prob=0.0,
            shift_delta_range=(-4.0, 4.0),
            noise_prob=0.0,
            noise_angle_deg_range=(0.5, 3.0),
        )
    )
    aug.train()

    torch.manual_seed(0)
    x_aug = aug(x)

    torch.testing.assert_close(x_aug, x)


def test_noise_changes_input_when_enabled() -> None:
    x = _snv_like_batch()
    aug = SpectraAugmenter(
        SpectraAugmenterConfig(
            shift_prob=0.0,
            noise_prob=1.0,
            noise_angle_deg_range=(3.0, 3.0),
            recenter_after_each_op=True,
            renorm_to_input_norm=True,
        )
    )
    aug.train()

    torch.manual_seed(0)
    x_aug = aug(x)

    assert not torch.allclose(x_aug, x)


def test_shift_changes_input_when_enabled() -> None:
    x = _snv_like_batch()
    aug = SpectraAugmenter(
        SpectraAugmenterConfig(
            shift_prob=1.0,
            shift_delta_range=(2.0, 2.0),
            noise_prob=0.0,
            recenter_after_each_op=True,
            renorm_to_input_norm=True,
        )
    )
    aug.train()

    torch.manual_seed(0)
    x_aug = aug(x)

    assert not torch.allclose(x_aug, x)


@pytest.mark.parametrize(
    (
        "shift_prob",
        "shift_delta_range",
        "noise_prob",
        "noise_angle_deg_range",
    ),
    [
        (-0.1, (-4.0, 4.0), 0.0, (0.5, 3.0)),
        (1.1, (-4.0, 4.0), 0.0, (0.5, 3.0)),
        (0.0, (4.0, -4.0), 0.0, (0.5, 3.0)),
        (0.0, (-4.0, 4.0), -0.1, (0.5, 3.0)),
        (0.0, (-4.0, 4.0), 1.1, (0.5, 3.0)),
        (0.0, (-4.0, 4.0), 0.0, (-0.5, 3.0)),
        (0.0, (-4.0, 4.0), 0.0, (3.0, 0.5)),
        (0.0, (-4.0, 4.0), 0.0, (0.5, 181.0)),
    ],
)
def test_invalid_config_raises_value_error(
    shift_prob: float,
    shift_delta_range: tuple[float, float],
    noise_prob: float,
    noise_angle_deg_range: tuple[float, float],
) -> None:
    with pytest.raises(ValueError):
        _ = SpectraAugmenterConfig(
            shift_prob=shift_prob,
            shift_delta_range=shift_delta_range,
            noise_prob=noise_prob,
            noise_angle_deg_range=noise_angle_deg_range,
        )


def test_non_positive_eps_raises_value_error() -> None:
    with pytest.raises(ValueError):
        _ = SpectraAugmenterConfig(eps=0.0)


@pytest.mark.parametrize("name", ["shift_delta_range", "noise_angle_deg_range"])
@pytest.mark.parametrize("endpoint", [0, 1])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_range_endpoints_fail_at_configuration(
    name: str, endpoint: int, value: float
) -> None:
    limits = [0.0, 1.0]
    limits[endpoint] = value
    with pytest.raises(ValueError, match="endpoints must be finite"):
        SpectraAugmenterConfig(**{name: tuple(limits)})


@pytest.mark.parametrize("eps", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_eps_fails_at_configuration(eps: float) -> None:
    with pytest.raises(ValueError, match="eps must be finite and positive"):
        SpectraAugmenterConfig(eps=eps)


@pytest.mark.parametrize(
    "dtype,seq_len",
    [(torch.bfloat16, 600), (torch.float16, 4096), (torch.float32, 256), (torch.float64, 256)],
)
@pytest.mark.parametrize("delta", [0.0, 0.5])
def test_shift_preserves_channel_coordinates_and_fractional_weights(
    dtype: torch.dtype, seq_len: int, delta: float
) -> None:
    # Alternating values are exactly representable even when the channel index
    # itself is not. A half-channel shift has a closed-form interpolation oracle.
    x = (torch.arange(seq_len) % 2).to(dtype=dtype).unsqueeze(0).requires_grad_()
    augmenter = SpectraAugmenter(
        SpectraAugmenterConfig(
            shift_prob=1.0,
            shift_delta_range=(delta, delta),
            noise_prob=0.0,
            shuffle_order_per_batch=False,
            recenter_after_each_op=False,
            renorm_to_input_norm=False,
        ),
        generator=torch.Generator().manual_seed(31),
    )

    actual = augmenter(x)
    expected = x.detach().clone()
    if delta == 0.5:
        expected[:, 1:] = 0.5
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == x.dtype and actual.device == x.device
    actual.sum().backward()
    assert x.grad is not None and bool(torch.isfinite(x.grad).all())


def test_forward_rejects_non_2d_input() -> None:
    x = torch.randn(8, 16, 1, dtype=torch.float32)
    aug = SpectraAugmenter(SpectraAugmenterConfig())
    aug.train()

    with pytest.raises(ValueError):
        _ = aug(x)


def test_forward_rejects_non_floating_input() -> None:
    x = torch.randint(0, 10, (8, 16), dtype=torch.int64)
    aug = SpectraAugmenter(SpectraAugmenterConfig())
    aug.train()

    with pytest.raises(TypeError):
        _ = aug(x)


def test_num_features_less_than_two_raises_value_error() -> None:
    x = _snv_like_batch(batch_size=4, seq_len=1)
    aug = SpectraAugmenter(
        SpectraAugmenterConfig(
            shift_prob=1.0,
            shift_delta_range=(0.0, 0.0),
            noise_prob=1.0,
            noise_angle_deg_range=(1.0, 1.0),
        )
    )
    aug.train()

    with pytest.raises(ValueError):
        _ = aug(x)


def test_noise_only_angle_is_close_to_configured_value() -> None:
    x = _snv_like_batch()
    aug = SpectraAugmenter(
        SpectraAugmenterConfig(
            shift_prob=0.0,
            noise_prob=1.0,
            noise_angle_deg_range=(2.0, 2.0),
            recenter_after_each_op=False,
            renorm_to_input_norm=False,
        )
    )
    aug.train()

    torch.manual_seed(0)
    x_aug = aug(x)

    angle_deg = _row_angle_deg(x, x_aug)
    torch.testing.assert_close(
        angle_deg,
        torch.full_like(angle_deg, 2.0),
        rtol=1.0e-4,
        atol=1.0e-4,
    )


def test_cosine_similarity_stays_within_valid_range_for_weak_aug() -> None:
    x = _snv_like_batch()
    aug = SpectraAugmenter(
        SpectraAugmenterConfig(
            shift_prob=1.0,
            shift_delta_range=(2.0, 2.0),
            noise_prob=1.0,
            noise_angle_deg_range=(1.0, 1.0),
            shuffle_order_per_batch=False,
            recenter_after_each_op=True,
            renorm_to_input_norm=True,
        )
    )
    aug.train()

    torch.manual_seed(0)
    x_aug = aug(x)

    cos = _row_cosine(x, x_aug)

    assert torch.all(cos <= 1.0 + 1.0e-6)
    assert torch.all(cos >= -1.0 - 1.0e-6)


@pytest.mark.parametrize(
    "shift_prob,noise_prob",
    [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5)],
)
def test_explicit_generators_reproduce_augmentation_without_global_rng_changes(
    shift_prob: float, noise_prob: float
) -> None:
    x = _snv_like_batch(batch_size=32)
    config = SpectraAugmenterConfig(
        shift_prob=shift_prob,
        noise_prob=noise_prob,
        shuffle_order_per_batch=True,
    )
    default_stream = torch.Generator().manual_seed(17)
    call_stream = torch.Generator().manual_seed(17)
    augmenter = SpectraAugmenter(config, generator=default_stream)
    global_state = torch.random.get_rng_state().clone()

    default_output = augmenter(x)
    assert torch.equal(torch.random.get_rng_state(), global_state)
    override_output = augmenter(x, generator=call_stream)
    assert torch.equal(torch.random.get_rng_state(), global_state)
    torch.testing.assert_close(default_output, override_output, rtol=0, atol=0)
    assert torch.equal(default_stream.get_state(), call_stream.get_state())


def test_per_call_stream_is_independent_of_default_stream() -> None:
    x = _snv_like_batch()
    config = SpectraAugmenterConfig(shift_prob=1.0, noise_prob=1.0)
    default_stream = torch.Generator().manual_seed(4)
    reference_stream = torch.Generator().manual_seed(4)
    override_stream = torch.Generator().manual_seed(29)
    augmenter = SpectraAugmenter(config, generator=default_stream)
    reference = SpectraAugmenter(config, generator=reference_stream)

    first = augmenter(x)
    expected_first = reference(x)
    state_before_override = default_stream.get_state().clone()
    override_state = override_stream.get_state().clone()
    augmenter(x, generator=override_stream)
    assert torch.equal(default_stream.get_state(), state_before_override)
    assert not torch.equal(override_stream.get_state(), override_state)
    second = augmenter(x)
    expected_second = reference(x)

    torch.testing.assert_close(first, expected_first, rtol=0, atol=0)
    torch.testing.assert_close(second, expected_second, rtol=0, atol=0)
    assert augmenter.generator is default_stream


def test_generator_state_restores_the_next_augmentation() -> None:
    x = _snv_like_batch()
    stream = torch.Generator().manual_seed(11)
    augmenter = SpectraAugmenter(
        SpectraAugmenterConfig(shift_prob=1.0, noise_prob=1.0),
        generator=stream,
    )
    augmenter(x)
    saved_state = stream.get_state().clone()
    expected = augmenter(x)
    stream.set_state(saved_state)
    actual = augmenter(x)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert augmenter.state_dict() == {}


def test_eval_identity_does_not_advance_default_or_override_streams() -> None:
    x = _snv_like_batch()
    default_stream = torch.Generator().manual_seed(3)
    override_stream = torch.Generator().manual_seed(7)
    augmenter = SpectraAugmenter(SpectraAugmenterConfig(), generator=default_stream).eval()
    default_state = default_stream.get_state().clone()
    override_state = override_stream.get_state().clone()
    global_state = torch.random.get_rng_state().clone()

    assert augmenter(x) is x
    assert augmenter(x, generator=override_stream) is x
    assert torch.equal(default_stream.get_state(), default_state)
    assert torch.equal(override_stream.get_state(), override_state)
    assert torch.equal(torch.random.get_rng_state(), global_state)


def test_invalid_generator_type_fails_clearly() -> None:
    with pytest.raises(TypeError, match="torch.Generator"):
        SpectraAugmenter(SpectraAugmenterConfig(), generator=17)
    augmenter = SpectraAugmenter(SpectraAugmenterConfig())
    with pytest.raises(TypeError, match="torch.Generator"):
        augmenter(_snv_like_batch(), generator=17)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_generator_device_mismatch_is_reported_without_moving_the_stream() -> None:
    stream = torch.Generator(device="cpu").manual_seed(9)
    augmenter = SpectraAugmenter(SpectraAugmenterConfig(), generator=stream).to("cuda")
    state = stream.get_state().clone()
    with pytest.raises(ValueError, match="must match input device"):
        augmenter(_snv_like_batch().cuda())
    assert augmenter.generator is stream
    assert stream.device.type == "cpu"
    assert torch.equal(stream.get_state(), state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_generator_controls_every_augmentation_draw() -> None:
    x = _snv_like_batch().cuda()
    config = SpectraAugmenterConfig(shift_prob=1.0, noise_prob=1.0)
    first_stream = torch.Generator(device=x.device).manual_seed(31)
    second_stream = torch.Generator(device=x.device).manual_seed(31)
    augmenter = SpectraAugmenter(config, generator=first_stream)
    global_state = torch.cuda.get_rng_state(x.device).clone()

    first = augmenter(x)
    second = augmenter(x, generator=second_stream)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.equal(torch.cuda.get_rng_state(x.device), global_state)
