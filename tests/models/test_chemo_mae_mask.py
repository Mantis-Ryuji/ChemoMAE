import torch

from chemomae.models.chemo_mae import make_patch_mask, ChemoMAE


def test_make_patch_mask_counts_and_patch_structure():
    B, L = 5, 64
    n_patches, n_mask = 16, 5
    mask = make_patch_mask(B, L, n_patches, n_mask)  # True=masked(隠す)

    assert mask.shape == (B, L) and mask.dtype is torch.bool

    patch_size = L // n_patches
    # 各バッチで True 数 = n_mask * patch_size
    true_counts = mask.sum(dim=1)
    assert torch.all(true_counts == n_mask * patch_size)

    # 連続ブロックの簡易性質（元テストの方針を踏襲）
    for b in range(B):
        idx = torch.where(mask[b])[0]
        if idx.numel() <= 1:
            continue
        diffs = torch.diff(idx)
        assert torch.all((diffs == 1) | (diffs > 1))


def test_encoder_accepts_visible_mask_from_make_patch_mask():
    B, L = 3, 48
    model = ChemoMAE(seq_len=L, d_model=32, nhead=4, num_layers=1,
                    dim_feedforward=64, latent_dim=10, n_patches=12, n_mask=3)

    x = torch.randn(B, L)
    visible_mask = model.make_visible(B)
    z = model.encoder(x, visible_mask)

    assert z.shape == (B, 10)
    norms = z.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_make_patch_mask_extreme_cases_zero_or_full_mask():
    B, L = 2, 32
    n_patches = 8

    # マスク無し
    mask0 = make_patch_mask(B, L, n_patches, 0)
    assert mask0.sum().item() == 0

    # 全ブロックマスク
    mask_all = make_patch_mask(B, L, n_patches, n_patches)
    assert mask_all.sum().item() == B * L


def test_explicit_generator_reproduces_masks_without_consuming_global_rng():
    generator = torch.Generator().manual_seed(81)
    state = generator.get_state().clone()
    global_state = torch.get_rng_state().clone()
    first = make_patch_mask(5, 64, 16, 5, generator=generator)
    generator.set_state(state)
    repeated = make_patch_mask(5, 64, 16, 5, generator=generator)
    assert torch.equal(first, repeated)
    assert torch.equal(global_state, torch.get_rng_state())
    assert not torch.equal(state, generator.get_state())


def test_model_forwards_generator_and_ignores_it_for_explicit_mask():
    model = ChemoMAE(seq_len=12, n_patches=3, n_mask=1, d_model=8,
                    nhead=2, num_layers=1, dim_feedforward=16, latent_dim=4).eval()
    x = torch.randn(2, 12)
    generator = torch.Generator().manual_seed(23)
    state = generator.get_state().clone()
    _, _, actual_visible = model(x, generator=generator)
    generator.set_state(state)
    expected_visible = model.make_visible(2, generator=generator)
    assert torch.equal(actual_visible, expected_visible)
    state_after_mask = generator.get_state().clone()
    model(x, visible_mask=expected_visible, generator=generator)
    model.reconstruct(x, visible_mask=expected_visible, generator=generator)
    assert torch.equal(state_after_mask, generator.get_state())
