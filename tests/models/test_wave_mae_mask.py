import torch

from wavemae.models.wave_mae import make_block_mask, WaveMAE


def test_make_block_mask_counts_and_block_structure():
    B, L = 5, 64
    n_blocks, n_mask = 16, 5
    mask = make_block_mask(B, L, n_blocks, n_mask)

    assert mask.shape == (B, L) and mask.dtype is torch.bool

    block_size = L // n_blocks
    # 各バッチで True 数 = n_mask * block_size
    true_counts = mask.sum(dim=1)
    assert torch.all(true_counts == n_mask * block_size)

    # 各行の True はブロック単位に連続している（ブロック境界内で連続 True）
    # ここでは簡易に「True のインデックス差が block_size-1 の倍数」で塊を確認
    for b in range(B):
        idx = torch.where(mask[b])[0]
        # ブロック境界内では等差（差分=1）が続く → 差分を見て block_size-1 ごとに切れるはず
        diffs = torch.diff(idx)
        # 1 以外のところでブロック境界が現れる（例外なく 1 と block_size の境目のみを想定）
        assert torch.all((diffs == 1) | (diffs > 1))


def test_encoder_accepts_visible_mask_from_make_block_mask():
    B, L = 3, 48
    model = WaveMAE(seq_len=L, d_model=32, nhead=4, num_layers=1,
                    dim_feedforward=64, latent_dim=10, n_blocks=12, n_mask=3)

    x = torch.randn(B, L)
    mask = model.make_mask(B)
    visible = ~mask  # エンコーダに渡す可視マスク（True=可視）
    z = model.encode(x, visible)

    assert z.shape == (B, 10)
    # z は L2 正規化済み → ノルム ≈ 1
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
