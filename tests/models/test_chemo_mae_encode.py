import pytest
import torch
import torch.nn.functional as F

from chemomae.models import ChemoMAE


def _model(*, normalize: bool = True) -> ChemoMAE:
    return ChemoMAE(
        seq_len=12, n_patches=3, n_mask=2, d_model=8, nhead=2,
        num_layers=1, dim_feedforward=16, latent_dim=4,
        latent_normalize=normalize, dropout=0.0,
    )


@pytest.mark.parametrize("normalize", [True, False])
def test_encode_representations_and_all_visible_forward(normalize: bool) -> None:
    model = _model(normalize=normalize).eval()
    x = torch.randn(3, 12)
    with torch.no_grad():
        cls = model.encode(x, representation="cls")
        raw = model.encode(x, representation="raw_latent")
        normalized = model.encode(x, representation="normalized_latent")
        latent = model.encode(x)
        _, forward_latent, visible = model(x, n_mask=0)
    assert cls.shape == (3, 8)
    assert raw.shape == (3, 4)
    torch.testing.assert_close(raw, model.encoder.to_latent(cls))
    torch.testing.assert_close(normalized, F.normalize(raw, dim=1))
    torch.testing.assert_close(latent, normalized if normalize else raw)
    torch.testing.assert_close(latent, forward_latent)
    assert visible.all()


def test_encode_preserves_rng_mode_and_gradient_flow() -> None:
    model = _model().train()
    x = torch.randn(2, 12, requires_grad=True)
    rng_before = torch.get_rng_state().clone()
    cls = model.encode(x, representation="cls")
    assert model.training
    assert torch.equal(rng_before, torch.get_rng_state())
    cls.square().sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert model.encoder.patch_proj.weight.grad is not None
    assert model.decoder.net[0].weight.grad is None
    assert model.encoder.to_latent.weight.grad is None


def test_encode_bypasses_decoder_and_mask_generator(monkeypatch) -> None:
    model = _model().eval()

    def unexpected_call(*args, **kwargs):
        raise AssertionError("encoding must not generate a mask or reconstruct")

    monkeypatch.setattr(model, "make_visible", unexpected_call)
    monkeypatch.setattr(model.decoder, "forward", unexpected_call)
    assert model.encode(torch.randn(2, 12)).shape == (2, 4)


@pytest.mark.parametrize("shape", [(0, 12), (2, 11), (12,), (2, 3, 4)])
def test_encode_rejects_invalid_shapes(shape) -> None:
    with pytest.raises(ValueError):
        _model().encode(torch.randn(shape))


def test_encode_rejects_integer_input_and_unknown_representation() -> None:
    with pytest.raises(TypeError):
        _model().encode(torch.zeros(2, 12, dtype=torch.int64))
    with pytest.raises(ValueError, match="representation"):
        _model().encode(torch.randn(2, 12), representation="unknown")


def test_normalized_latent_keeps_zero_projection_finite() -> None:
    model = _model(normalize=False).eval()
    with torch.no_grad():
        model.encoder.to_latent.weight.zero_()
        model.encoder.to_latent.bias.zero_()
        z = model.encode(torch.randn(2, 12), representation="normalized_latent")
    assert torch.equal(z, torch.zeros_like(z))
