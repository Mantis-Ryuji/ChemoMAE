from __future__ import annotations

import pytest
import torch

from chemomae.training.augmenter import SpectraAugmenter, SpectraAugmenterConfig


def _snv_like_batch(
    batch_size: int = 8,
    seq_len: int = 16,
    seed: int = 0,
) -> torch.Tensor:
    """平均0・行ノルム1のSNV風バッチを作成する。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, seq_len, generator=g, dtype=torch.float32)
    x = x - x.mean(dim=1, keepdim=True)
    x = x / torch.linalg.norm(x, dim=1, keepdim=True).clamp_min(1.0e-12)
    return x


def _row_cosine(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """行ごとの cosine similarity を計算する。"""
    denom = (
        torch.linalg.norm(x, dim=1) * torch.linalg.norm(y, dim=1)
    ).clamp_min(1.0e-12)
    return torch.sum(x * y, dim=1) / denom


def _row_angle_deg(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """行ごとの角度を degree 単位で計算する。"""
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