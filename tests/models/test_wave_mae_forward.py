import torch

from wavemae.models.wave_mae import WaveMAE
from wavemae.models.losses import masked_sse


def test_wave_mae_forward_shapes_and_types():
    B, L = 4, 32
    model = WaveMAE(
        seq_len=L,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        use_learnable_pos=True,
        latent_dim=16,
        dec_hidden=64,
        dec_dropout=0.0,
        n_blocks=8,
        n_mask=3,
    )

    x = torch.randn(B, L)
    x_recon, z, mask = model(x)

    assert x_recon.shape == (B, L)
    assert z.shape == (B, 16)
    assert mask.shape == (B, L) and mask.dtype == torch.bool

    # マスク総数は n_mask * block_size のはず
    block_size = L // model.n_blocks
    assert int(mask[0].sum().item()) == model.n_mask * block_size


def test_wave_mae_encode_and_reconstruct_api():
    B, L = 2, 24
    model = WaveMAE(seq_len=L, d_model=32, nhead=4, num_layers=1, dim_feedforward=64,
                    latent_dim=8, dec_hidden=32, n_blocks=6, n_mask=2)

    x = torch.randn(B, L)
    # reconstruct: mask 省略可
    xr = model.reconstruct(x)
    assert xr.shape == (B, L)

    # encode: 可視マスク（True=可視）を与える
    mask = model.make_mask(B)
    visible = ~mask
    z = model.encode(x, visible)
    assert z.shape == (B, 8)


def test_wave_mae_loss_and_backward_on_masked_sse():
    B, L = 3, 48
    model = WaveMAE(seq_len=L, d_model=64, nhead=4, num_layers=2,
                    dim_feedforward=128, latent_dim=12, dec_hidden=64,
                    n_blocks=12, n_mask=4)

    x = torch.randn(B, L, requires_grad=True)
    x_recon, z, mask = model(x)

    loss = masked_sse(x_recon, x, mask, reduction="batch_mean")
    loss.backward()

    # エンコーダ・デコーダ側のパラメータに勾配が出ているか（どれか一つで確認）
    any_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters())
    assert any_grad, "No gradients flowed through model parameters"
