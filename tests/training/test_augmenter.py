import math

import pytest
import torch

from chemomae.training.augmenter import SpectraAugmenter, SpectraAugmenterConfig


def _normalized_batch(
    batch_size: int = 8,
    seq_len: int = 16,
    seed: int = 0,
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, seq_len, generator=g, dtype=torch.float32)
    x = x / (torch.linalg.norm(x, dim=1, keepdim=True) + 1e-12)
    return x


def test_augmenter_eval_mode_returns_input_unchanged() -> None:
    x = _normalized_batch()
    aug = SpectraAugmenter(
        SpectraAugmenterConfig(
            noise_prob=1.0,
            noise_cos_range=(0.99, 0.999),
            tilt_prob=1.0,
            tilt_cos_range=(0.99, 0.999),
        )
    )
    aug.eval()

    y = aug(x)

    torch.testing.assert_close(y, x)


def test_augmenter_train_mode_preserves_shape() -> None:
    x = _normalized_batch()
    aug = SpectraAugmenter(
        SpectraAugmenterConfig(
            noise_prob=1.0,
            noise_cos_range=(0.99, 0.999),
            tilt_prob=1.0,
            tilt_cos_range=(0.99, 0.999),
        )
    )
    aug.train()

    y = aug(x)

    assert y.shape == x.shape


def test_noise_only_preserves_row_norm() -> None:
    x = _normalized_batch()
    aug = SpectraAugmenter(
        SpectraAugmenterConfig(
            noise_prob=1.0,
            noise_cos_range=(0.995, 0.995),
            tilt_prob=0.0,
            tilt_cos_range=(1.0, 1.0),
        )
    )
    aug.train()

    y = aug(x)

    x_norm = torch.linalg.norm(x, dim=1)
    y_norm = torch.linalg.norm(y, dim=1)
    torch.testing.assert_close(x_norm, y_norm, rtol=1e-5, atol=1e-6)


def test_tilt_only_preserves_row_norm() -> None:
    x = _normalized_batch()
    aug = SpectraAugmenter(
        SpectraAugmenterConfig(
            noise_prob=0.0,
            noise_cos_range=(1.0, 1.0),
            tilt_prob=1.0,
            tilt_cos_range=(0.995, 0.995),
        )
    )
    aug.train()

    y = aug(x)

    x_norm = torch.linalg.norm(x, dim=1)
    y_norm = torch.linalg.norm(y, dim=1)
    torch.testing.assert_close(x_norm, y_norm, rtol=1e-5, atol=1e-6)


def test_noise_and_tilt_together_preserve_row_norm() -> None:
    x = _normalized_batch()
    aug = SpectraAugmenter(
        SpectraAugmenterConfig(
            noise_prob=1.0,
            noise_cos_range=(0.995, 0.999),
            tilt_prob=1.0,
            tilt_cos_range=(0.995, 0.999),
        )
    )
    aug.train()

    y = aug(x)

    x_norm = torch.linalg.norm(x, dim=1)
    y_norm = torch.linalg.norm(y, dim=1)
    torch.testing.assert_close(x_norm, y_norm, rtol=1e-5, atol=1e-6)


def test_zero_probability_keeps_input_unchanged_even_in_train_mode() -> None:
    x = _normalized_batch()
    aug = SpectraAugmenter(
        SpectraAugmenterConfig(
            noise_prob=0.0,
            noise_cos_range=(0.99, 0.999),
            tilt_prob=0.0,
            tilt_cos_range=(0.99, 0.999),
        )
    )
    aug.train()

    y = aug(x)

    torch.testing.assert_close(y, x)


def test_noise_changes_input_when_enabled() -> None:
    x = _normalized_batch()
    aug = SpectraAugmenter(
        SpectraAugmenterConfig(
            noise_prob=1.0,
            noise_cos_range=(0.99, 0.99),
            tilt_prob=0.0,
            tilt_cos_range=(1.0, 1.0),
        )
    )
    aug.train()

    y = aug(x)

    assert not torch.allclose(y, x)


def test_tilt_changes_input_when_enabled() -> None:
    x = _normalized_batch()
    aug = SpectraAugmenter(
        SpectraAugmenterConfig(
            noise_prob=0.0,
            noise_cos_range=(1.0, 1.0),
            tilt_prob=1.0,
            tilt_cos_range=(0.99, 0.99),
        )
    )
    aug.train()

    y = aug(x)

    assert not torch.allclose(y, x)


@pytest.mark.parametrize(
    ("noise_prob", "noise_cos_range", "tilt_prob", "tilt_cos_range"),
    [
        (-0.1, (1.0, 1.0), 0.0, (1.0, 1.0)),
        (1.1, (1.0, 1.0), 0.0, (1.0, 1.0)),
        (0.0, (0.0, 1.0), 0.0, (1.0, 1.0)),
        (0.0, (1.0, 1.1), 0.0, (1.0, 1.0)),
        (0.0, (0.999, 0.5), 0.0, (1.0, 1.0)),
        (0.0, (1.0, 1.0), -0.1, (1.0, 1.0)),
        (0.0, (1.0, 1.0), 1.1, (1.0, 1.0)),
        (0.0, (1.0, 1.0), 0.0, (0.0, 1.0)),
        (0.0, (1.0, 1.0), 0.0, (1.0, 1.1)),
        (0.0, (1.0, 1.0), 0.0, (0.999, 0.5)),
    ],
)
def test_invalid_config_raises_value_error(
    noise_prob: float,
    noise_cos_range: tuple[float, float],
    tilt_prob: float,
    tilt_cos_range: tuple[float, float],
) -> None:
    with pytest.raises(ValueError):
        SpectraAugmenter(
            SpectraAugmenterConfig(
                noise_prob=noise_prob,
                noise_cos_range=noise_cos_range,
                tilt_prob=tilt_prob,
                tilt_cos_range=tilt_cos_range,
            )
        )


def test_non_positive_eps_raises_value_error() -> None:
    with pytest.raises(ValueError):
        SpectraAugmenter(
            SpectraAugmenterConfig(
                noise_prob=0.0,
                noise_cos_range=(1.0, 1.0),
                tilt_prob=0.0,
                tilt_cos_range=(1.0, 1.0),
                eps=0.0,
            )
        )


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


def test_num_features_less_than_two_raises_for_noise_fallback() -> None:
    x = _normalized_batch(batch_size=4, seq_len=1)
    aug = SpectraAugmenter(
        SpectraAugmenterConfig(
            noise_prob=1.0,
            noise_cos_range=(0.99, 0.99),
            tilt_prob=0.0,
            tilt_cos_range=(1.0, 1.0),
        )
    )
    aug.train()

    with pytest.raises(ValueError):
        _ = aug(x)


def test_num_features_less_than_two_raises_for_tilt() -> None:
    x = _normalized_batch(batch_size=4, seq_len=1)
    aug = SpectraAugmenter(
        SpectraAugmenterConfig(
            noise_prob=0.0,
            noise_cos_range=(1.0, 1.0),
            tilt_prob=1.0,
            tilt_cos_range=(0.99, 0.99),
        )
    )
    aug.train()

    with pytest.raises(ValueError):
        _ = aug(x)


def test_cosine_similarity_stays_within_valid_range_for_weak_aug() -> None:
    x = _normalized_batch()
    aug = SpectraAugmenter(
        SpectraAugmenterConfig(
            noise_prob=1.0,
            noise_cos_range=(0.999, 0.999),
            tilt_prob=1.0,
            tilt_cos_range=(0.999, 0.999),
        )
    )
    aug.train()

    y = aug(x)

    cos = torch.sum(x * y, dim=1) / (
        torch.linalg.norm(x, dim=1) * torch.linalg.norm(y, dim=1) + 1e-12
    )

    assert torch.all(cos <= 1.0 + 1e-6)
    assert torch.all(cos >= -1.0 - 1e-6)