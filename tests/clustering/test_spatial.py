"""Exact directed-pair oracles for occupancy-corrected spatial LLA."""

from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import pytest
import torch

from chemomae.clustering import LLAResult, LLAWindowResult, local_label_agreement
from chemomae.clustering.spatial import _finalize_scores


def _pair_oracle(
    labels: np.ndarray, mask: np.ndarray, width: int
) -> tuple[int, int, int, float, float]:
    """Enumerate distinct pixel pairs, without convolutions or offset slices."""
    coordinates = [tuple(int(value) for value in point) for point in np.argwhere(mask)]
    pairs = [
        (center, neighbor) for center in coordinates for neighbor in coordinates
        if center != neighbor and max(
            abs(center[0] - neighbor[0]), abs(center[1] - neighbor[1])
        ) <= width // 2
    ]
    matching = sum(int(labels[center] == labels[neighbor]) for center, neighbor in pairs)
    coverage = len({center for center, _ in pairs})
    counts = [sum(int(labels[point] == label) for point in coordinates)
              for label in sorted({int(labels[point]) for point in coordinates})]
    n = len(coordinates)
    chance = Fraction(sum(count * (count - 1) for count in counts), n * (n - 1)) if n > 1 else None
    raw = Fraction(matching, len(pairs)) if pairs else None
    score = (raw - chance) / (1 - chance) if raw is not None and chance is not None and chance < 1 else None
    return matching, len(pairs), coverage, float(raw) if raw is not None else math.nan, \
        float(score) if score is not None else math.nan


def _assert_float(actual: float, expected: float) -> None:
    if math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert actual == pytest.approx(expected, rel=1e-15, abs=1e-15)


def _assert_window_equal(actual: LLAWindowResult, expected: LLAWindowResult) -> None:
    assert actual.window == expected.window
    assert (actual.matching_pairs, actual.valid_pairs, actual.pixels_with_neighbors) == (
        expected.matching_pairs, expected.valid_pairs, expected.pixels_with_neighbors
    )
    _assert_float(actual.score, expected.score)
    _assert_float(actual.raw_agreement, expected.raw_agreement)
    assert actual.neighbor_pixel_fraction == expected.neighbor_pixel_fraction
    assert actual.raw_undefined_reason == expected.raw_undefined_reason
    assert actual.undefined_reasons == expected.undefined_reasons


def test_hand_counted_reference_uses_finite_sample_and_pair_weighting() -> None:
    # The WoodDegradationMap reference's [1, 1, 1, 2] map, relabeled to include 0.
    labels = np.array([[0, 0, 0, 8]], dtype=np.int32)
    result = local_label_agreement(labels, np.ones_like(labels, dtype=bool))
    assert isinstance(result, LLAResult)
    assert result.class_labels == (0, 8) and result.class_counts == (3, 1)
    assert result.valid_pixels == 4 and result.occupancy == (0.75, 0.25)
    assert result.used_classes == 2 and result.maximum_occupancy == 0.75
    assert result.chance_agreement == 0.5  # sum(p_k**2) would be 0.625.
    for row, (width, matching, pairs, corrected) in zip(result.windows, (
        (3, 4, 6, Fraction(1, 3)), (5, 6, 10, Fraction(1, 5)), (9, 6, 12, Fraction(0)),
    )):
        assert (row.window, row.matching_pairs, row.valid_pairs) == (width, matching, pairs)
        assert row.score == float(corrected)
        assert row.raw_agreement == matching / pairs
        assert row.pixels_with_neighbors == 4 and row.neighbor_pixel_fraction == 1.0
        assert row.raw_undefined_reason is None and row.undefined_reasons == ()
    assert result.windows[0].raw_agreement != pytest.approx((1 + 1 + 0.5 + 0) / 4)


@pytest.mark.parametrize("shape", [(1, 1), (1, 8), (2, 3), (6, 10), (9, 9)])
@pytest.mark.parametrize("class_chunk", [1, 2, 16, None])
def test_exact_counts_match_pixel_pair_oracle(
    shape: tuple[int, int], class_chunk: int | None
) -> None:
    random = np.random.default_rng(78)
    mask = random.random(shape) > 0.3
    mask[0, 0] = True
    labels = random.choice([-10, 0, 8, 30], size=shape)
    result = local_label_agreement(labels, mask, windows=(1, 3, 5, 9), class_chunk=class_chunk)
    for row in result.windows:
        matching, pairs, coverage, raw, score = _pair_oracle(labels, mask, row.window)
        assert (row.matching_pairs, row.valid_pairs, row.pixels_with_neighbors) == (
            matching, pairs, coverage
        )
        _assert_float(row.raw_agreement, raw)
        _assert_float(row.score, score)
        assert row.neighbor_pixel_fraction == coverage / result.valid_pixels


def test_negative_corrected_score_is_retained() -> None:
    labels = np.array([[-4, 0, -4]])
    result = local_label_agreement(labels, np.ones_like(labels), windows=(3, 5, 9))
    assert result.chance_agreement == 1 / 3
    assert result.windows[0].matching_pairs == 0 and result.windows[0].valid_pairs == 4
    assert result.windows[0].raw_agreement == 0.0
    assert result.windows[0].score == -0.5
    assert result.windows[1].score == result.windows[2].score == 0.0


def test_holes_diagonals_self_exclusion_and_single_class_reason() -> None:
    labels = np.array([[0, -20], [-20, 0]])
    mask = np.array([[True, False], [False, True]])
    result = local_label_agreement(labels, mask)
    assert result.class_labels == (0,) and result.class_counts == (2,)
    for row in result.windows:
        assert row.matching_pairs == row.valid_pairs == 2
        assert row.raw_agreement == 1.0 and math.isnan(row.score)
        assert row.undefined_reasons == ("single_class",)
    assert result.chance_agreement == 1.0


def test_width_is_not_radius_and_image_edges_do_not_wrap() -> None:
    labels = np.array([[8, 99, 99, 0, 99, 99, 8]])
    mask = np.array([[1, 0, 0, 1, 0, 0, 1]])
    result = local_label_agreement(labels, mask)
    for row in result.windows[:2]:
        assert row.valid_pairs == row.matching_pairs == 0
        assert math.isnan(row.score) and math.isnan(row.raw_agreement)
        assert row.raw_undefined_reason == "no_valid_neighbor_pairs"
        assert row.undefined_reasons == ("no_valid_neighbor_pairs",)
    assert result.windows[2].valid_pairs == 4 and result.windows[2].score == -0.5
    endpoints = np.array([[1, 0, 0, 0, 0, 1]])
    assert all(row.valid_pairs == 0 for row in local_label_agreement(
        endpoints, endpoints > 0, windows=(3, 5, 9)
    ).windows)


def test_isolated_valid_pixels_remain_in_occupancy() -> None:
    labels = np.array([[0, -4, 8, 8, 8, 8, 8, 8, 8, 0]])
    mask = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 1]])
    result = local_label_agreement(labels, mask)
    assert result.class_labels == (-4, 0) and result.class_counts == (1, 2)
    assert result.chance_agreement == 1 / 3
    for row in result.windows:
        assert row.valid_pairs == 2 and row.matching_pairs == 0
        assert row.pixels_with_neighbors == 2 and row.neighbor_pixel_fraction == 2 / 3
        assert row.score == -0.5


def test_one_valid_pixel_has_all_applicable_undefined_reasons() -> None:
    result = local_label_agreement(np.array([[0]]), np.array([[True]]))
    assert math.isnan(result.chance_agreement)
    for row in result.windows:
        assert math.isnan(row.score) and math.isnan(row.raw_agreement)
        assert row.matching_pairs == row.valid_pairs == row.pixels_with_neighbors == 0
        assert row.undefined_reasons == (
            "fewer_than_two_valid_pixels", "no_valid_neighbor_pairs", "single_class"
        )


def test_chunking_relabeling_padding_transpose_and_read_only_inputs() -> None:
    labels = np.array([[0, 8, -4, 0], [-4, 0, 8, 0], [8, -4, 0, 8]])
    mask = labels != 8
    before = labels.copy()
    labels.flags.writeable = False
    mask.flags.writeable = False
    original = local_label_agreement(labels, mask, class_chunk=None)
    relabeled = np.where(labels == 0, 700, np.where(labels == -4, 300, 900))
    variants = [
        local_label_agreement(labels, mask, class_chunk=1),
        local_label_agreement(labels.T, mask.T, class_chunk=2),
        local_label_agreement(np.pad(labels, 5), np.pad(mask, 5)),
        local_label_agreement(relabeled, mask),
        local_label_agreement(torch.tensor(before), torch.tensor(mask.copy())),
        local_label_agreement(labels[:, ::-1], mask[:, ::-1]),
    ]
    for result in variants:
        assert result.chance_agreement == original.chance_agreement
        for actual, expected in zip(result.windows, original.windows):
            _assert_window_equal(actual, expected)
    np.testing.assert_array_equal(labels, before)


def test_final_integer_correction_handles_near_one_and_large_products() -> None:
    # No huge image allocation: even float64 rounds chance agreement to one.
    n = 2**60
    counts = (n - 1, 1)
    assert sum(count * (count - 1) for count in counts) / (n * (n - 1)) == 1.0
    raw, score, reasons = _finalize_scores(100, 100, counts)
    assert raw == score == 1.0 and reasons == ()
    raw, score, reasons = _finalize_scores(0, 2, counts)
    assert raw == 0.0 and score == -float(Fraction(n - 2, 2)) and reasons == ()


def test_default_device_and_mixed_numpy_torch_inputs_preserve_inputs() -> None:
    labels = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
    mask = np.ones((2, 2), dtype=bool)
    before = labels.clone()
    result = local_label_agreement(labels, mask)
    mixed = local_label_agreement(labels.numpy(), torch.tensor(mask))
    assert result == mixed
    assert torch.equal(labels, before)


def test_large_signed_labels_and_default_float_dtype_do_not_change_counts() -> None:
    labels = np.array([[np.iinfo(np.int64).min, np.iinfo(np.int64).max]], dtype=np.int64)
    mask = np.array([[True, True]])
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            result = local_label_agreement(labels, mask)
    finally:
        torch.set_default_dtype(original_dtype)
    assert result.class_labels == (int(labels[0, 0]), int(labels[0, 1]))
    assert result.class_counts == (1, 1)
    assert all(row.matching_pairs == 0 and row.valid_pairs == 2 for row in result.windows)


@pytest.mark.parametrize("labels, mask, message", [
    (np.zeros((2, 2), dtype=int), np.zeros((2, 2), dtype=bool), "at least one"),
    (np.zeros((0, 2), dtype=int), np.zeros((0, 2), dtype=bool), "at least one"),
    (np.array([[1.0]]), np.array([[True]]), "integer"),
    (np.array([[np.nan]]), np.array([[True]]), "integer"),
    (np.array([[True]]), np.array([[True]]), "integer"),
    (np.array([1]), np.array([True]), "two-dimensional"),
    (np.array([[1]]), np.ones((2, 2), dtype=bool), "matching"),
    (np.array([[1]]), np.array([[1.0]]), "binary"),
    (np.array([[1]]), np.array([[2]]), "binary"),
    (np.array([[1]]), np.array([[-1]]), "binary"),
    (np.array([[2**64 - 1]], dtype=np.uint64), np.array([[True]]), "64-bit"),
    (torch.tensor([[1.0]]), torch.tensor([[True]]), "integer"),
    (torch.tensor([[1]]), torch.tensor([[2]]), "zero and one"),
    (torch.tensor([[1]]), torch.tensor([[1.0]]), "binary"),
])
def test_invalid_maps_masks_and_dtypes_raise(
    labels: np.ndarray | torch.Tensor, mask: np.ndarray | torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        local_label_agreement(labels, mask)


@pytest.mark.parametrize("windows", [(), (0,), (2,), (True,), (3.0,), (3, 3), (4097,)])
def test_invalid_widths_raise(windows: tuple[object, ...]) -> None:
    with pytest.raises(ValueError, match="window"):
        local_label_agreement(np.array([[0, 1]]), np.array([[True, True]]), windows=windows)


@pytest.mark.parametrize("chunk", [0, -1, True, 1.5])
def test_invalid_chunks_raise(chunk: object) -> None:
    with pytest.raises(ValueError, match="class_chunk"):
        local_label_agreement(np.array([[0, 1]]), np.array([[True, True]]), class_chunk=chunk)


def test_unsupported_device_and_input_types_raise() -> None:
    with pytest.raises(ValueError, match="CPU or CUDA"):
        local_label_agreement(np.array([[0]]), np.array([[True]]), device="meta")
    with pytest.raises(TypeError, match="NumPy array or Torch tensor"):
        local_label_agreement([[0]], np.array([[True]]))
    with pytest.raises(TypeError, match="NumPy array or Torch tensor"):
        local_label_agreement(np.array([[0]]), [[True]])


def test_unavailable_cuda_reports_cpu_alternative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA is not available"):
        local_label_agreement(np.array([[0]]), np.array([[True]]), device="cuda")


def test_noninteger_convolution_counts_are_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    from chemomae.clustering import spatial

    original = spatial.F.conv2d

    def noninteger_conv(*args: object, **kwargs: object) -> torch.Tensor:
        return original(*args, **kwargs) + 0.25

    monkeypatch.setattr(spatial.F, "conv2d", noninteger_conv)
    with pytest.raises(RuntimeError, match="noninteger"):
        local_label_agreement(np.array([[0, 1]]), np.array([[True, True]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_exact_cpu_counts_and_works_outside_amp() -> None:
    random = np.random.default_rng(37)
    labels = random.integers(-2, 3, size=(8, 12))
    mask = random.random(labels.shape) > 0.25
    cpu = local_label_agreement(labels, mask, device="cpu", class_chunk=None)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        cuda = local_label_agreement(
            torch.tensor(labels, device="cuda"), torch.tensor(mask, device="cuda"),
            class_chunk=1,
        )
    assert cuda == cpu
