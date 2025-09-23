import torch

from wavemae.models.wave_mae import make_block_mask, WaveMAE


def test_make_block_mask_counts_and_block_structure():
    B, L = 5, 64
    n_blocks, n_mask = 16, 5
    mask = make_block_mask(B, L, n_blocks, n_mask)  # True=masked(隠す)

    assert mask.shape == (B, L) and mask.dtype is torch.bool

    block_size = L // n_blocks
    # 各バッチで True 数 = n_mask * block_size
    true_counts = mask.sum(dim=1)
    assert torch.all(true_counts == n_mask * block_size)

    # 連続ブロックの簡易性質（元テストの方針を踏襲）
    for b in range(B):
        idx = torch.where(mask[b])[0]
        if idx.numel() <= 1:
            continue
        diffs = torch.diff(idx)
        assert torch.all((diffs == 1) | (diffs > 1))


def test_encoder_accepts_visible_mask_from_make_block_mask():
    B, L = 3, 48
    model = WaveMAE(seq_len=L, d_model=32, nhead=4, num_layers=1,
                    dim_feedforward=64, latent_dim=10, n_blocks=12, n_mask=3)

    x = torch.randn(B, L)
    masked = model.make_mask(B)     # True=masked（ユーティリティ）
    visible = ~masked               # True=visible（エンコーダに渡す）
    z = model.encode(x, visible)

    assert z.shape == (B, 10)
    norms = z.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_make_block_mask_extreme_cases_zero_or_full_mask():
    B, L = 2, 32
    n_blocks = 8

    # マスク無し
    mask0 = make_block_mask(B, L, n_blocks, 0)
    assert mask0.sum().item() == 0

    # 全ブロックマスク
    mask_all = make_block_mask(B, L, n_blocks, n_blocks)
    assert mask_all.sum().item() == B * L
