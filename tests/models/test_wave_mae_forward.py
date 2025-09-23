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
    x_recon, z, visible = model(x)  # 第3戻り値は visible(True=使う)

    assert x_recon.shape == (B, L)
    assert z.shape == (B, 16)
    assert visible.shape == (B, L) and visible.dtype == torch.bool

    # マスク総数は n_mask * block_size のはず（masked = ~visible）
    block_size = L // model.n_blocks
    masked = ~visible
    assert int(masked[0].sum().item()) == model.n_mask * block_size


def test_wave_mae_encode_and_reconstruct_api():
    B, L = 2, 24
    model = WaveMAE(seq_len=L, d_model=32, nhead=4, num_layers=1, dim_feedforward=64,
                    latent_dim=8, dec_hidden=32, n_blocks=6, n_mask=2)

    x = torch.randn(B, L)

    # reconstruct: visible 省略可（内部で生成）
    xr, _, v = model(x)
    assert xr.shape == (B, L)
    assert v.dtype == torch.bool and v.shape == (B, L)

    # encode: 可視マスク（True=可視）を与える
    visible = torch.ones(B, L, dtype=torch.bool)
    z = model.encode(x, visible)
    assert z.shape == (B, 8)


def test_wave_mae_loss_and_backward_on_masked_sse():
    B, L = 3, 48
    model = WaveMAE(seq_len=L, d_model=64, nhead=4, num_layers=2,
                    dim_feedforward=128, latent_dim=12, dec_hidden=64,
                    n_blocks=12, n_mask=4)

    x = torch.randn(B, L, requires_grad=True)
    x_recon, z, visible = model(x)

    loss = masked_sse(x_recon, x, ~visible, reduction="batch_mean")  # ← visible を反転
    loss.backward()

    any_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters())
    assert any_grad, "No gradients flowed through model parameters"
