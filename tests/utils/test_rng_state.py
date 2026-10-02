import random

import numpy as np
import pytest
import torch

from chemomae.utils import capture_rng_state, restore_rng_state


def test_global_streams_round_trip_and_owned_generator_is_untouched() -> None:
    original = capture_rng_state()
    try:
        random.seed(12)
        np.random.seed(34)
        torch.manual_seed(56)
        owned = torch.Generator().manual_seed(78)
        owned_state = owned.get_state().clone()
        snapshot = capture_rng_state()
        expected = (random.random(), np.random.randn(7), torch.randn(9))
        random.random()
        np.random.randn(11)
        torch.randn(13)
        restore_rng_state(snapshot, restore_cuda=False)
        actual = (random.random(), np.random.randn(7), torch.randn(9))
        assert expected[0] == actual[0]
        np.testing.assert_array_equal(expected[1], actual[1])
        torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)
        assert torch.equal(owned.get_state(), owned_state)
    finally:
        restore_rng_state(original, restore_cuda=False)


@pytest.mark.parametrize("corruption", ["schema", "cpu", "numpy", "python"])
def test_invalid_rng_state_fails_without_changing_global_streams(corruption: str) -> None:
    snapshot = capture_rng_state()
    broken = capture_rng_state()
    if corruption == "schema":
        broken["format_version"] = 2
    elif corruption == "cpu":
        broken["torch_cpu"] = torch.zeros(3, dtype=torch.uint8)
    elif corruption == "numpy":
        broken["numpy"]["position"] = -1
    else:
        broken["python"] = None
    with pytest.raises(ValueError):
        restore_rng_state(broken, restore_cuda=False)
    actual = capture_rng_state()
    assert snapshot["python"] == actual["python"]
    assert torch.equal(snapshot["torch_cpu"], actual["torch_cpu"])
    assert snapshot["numpy"]["position"] == actual["numpy"]["position"]
    assert torch.equal(snapshot["numpy"]["keys"], actual["numpy"]["keys"])
