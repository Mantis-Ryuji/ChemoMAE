from copy import deepcopy
from pathlib import Path

import pytest
import torch

from chemomae.models import ChemoMAE, ChemoMAEConfig


def _model(**overrides: object) -> ChemoMAE:
    config = {
        "seq_len": 16, "n_patches": 4, "n_mask": 2, "d_model": 12,
        "nhead": 3, "num_layers": 1, "dim_feedforward": 19, "dropout": 0.2,
        "latent_dim": 5, "latent_normalize": False, "decoder_num_layers": 3,
    }
    return ChemoMAE(**{**config, **overrides})


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_artifact_restores_nonrecoverable_config_dtype_and_predictions(
    tmp_path: Path, dtype: torch.dtype,
) -> None:
    model = _model().to(dtype=dtype).eval()
    model.n_mask = 1
    model.encoder.latent_normalize = True
    x = torch.arange(48, dtype=dtype).reshape(3, 16) / 48
    visible = torch.ones_like(x, dtype=torch.bool)
    with torch.no_grad():
        expected = model(x, visible_mask=visible)
    saved = {key: value.clone() for key, value in model.state_dict().items()}
    path = tmp_path / "nested" / "model.pt"
    model.save(path)
    restored = ChemoMAE.load(path)
    assert restored.get_config() == model.get_config()
    assert not restored.training
    assert next(restored.parameters()).dtype == dtype
    with torch.no_grad():
        actual = restored(x, visible_mask=visible)
    for before, after in zip(expected, actual):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    for key in saved:
        torch.testing.assert_close(saved[key], model.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(saved[key], restored.state_dict()[key], rtol=0, atol=0)


def test_explicit_weight_snapshot_does_not_replace_live_weights_or_mode(tmp_path: Path) -> None:
    model = _model().train()
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    selected = {key: value + 0.125 for key, value in state.items()}
    model.save(tmp_path / "selected.pt", state_dict=selected)
    restored = ChemoMAE.load(tmp_path / "selected.pt")
    assert model.training
    for key, value in state.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(restored.state_dict()[key], selected[key], rtol=0, atol=0)


@pytest.mark.parametrize("corruption", ["schema", "kind", "config", "shape", "dtype", "nonfinite", "missing_weight"])
def test_corrupt_artifacts_fail_clearly(tmp_path: Path, corruption: str) -> None:
    path = tmp_path / "model.pt"
    _model().save(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    key = next(iter(payload["state_dict"]))
    if corruption == "schema":
        payload["format_version"] = 99
    elif corruption == "kind":
        payload["artifact"] = "chemomae.training"
    elif corruption == "config":
        del payload["config"]["nhead"]
    elif corruption == "shape":
        payload["state_dict"][key] = torch.zeros(1)
    elif corruption == "dtype":
        payload["state_dict"][key] = payload["state_dict"][key].double()
    elif corruption == "nonfinite":
        payload["state_dict"][key].fill_(float("nan"))
    else:
        del payload["state_dict"][key]
    torch.save(payload, path)
    with pytest.raises((ValueError, TypeError)):
        ChemoMAE.load(path)


@pytest.mark.parametrize("change", [
    {"n_patches": 0}, {"seq_len": 15}, {"nhead": 5}, {"n_mask": 5},
    {"dropout": float("nan")}, {"latent_normalize": 1}, {"num_layers": 0},
])
def test_invalid_constructor_config_is_rejected(change: dict[str, object]) -> None:
    with pytest.raises((ValueError, TypeError)):
        _model(**change)


def test_config_is_complete_and_missing_fields_are_not_inferred() -> None:
    config = ChemoMAEConfig().to_dict()
    assert ChemoMAEConfig.from_dict(config).to_dict() == config
    broken = deepcopy(config)
    del broken["dropout"]
    with pytest.raises(ValueError, match="exactly"):
        ChemoMAEConfig.from_dict(broken)


def test_invalid_snapshot_does_not_create_output(tmp_path: Path) -> None:
    model = _model()
    path = tmp_path / "missing_parent" / "model.pt"
    with pytest.raises(ValueError, match="keys"):
        model.save(path, state_dict={})
    assert not path.parent.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_gpu_save_load_is_portable_to_cpu(tmp_path: Path) -> None:
    model = _model().cuda().eval()
    path = tmp_path / "cuda.pt"
    model.save(path)
    restored = ChemoMAE.load(path, device="cpu")
    assert next(restored.parameters()).device.type == "cpu"
    assert restored.get_config() == model.get_config()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value.cpu(), restored.state_dict()[name], rtol=0, atol=0)
