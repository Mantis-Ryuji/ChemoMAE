import torch
import math

from wavemae.models.losses import masked_sse, masked_mse


def test_masked_losses_sum_mean_batchmean_relationship():
    B, L = 3, 8
    x = torch.arange(B * L, dtype=torch.float32).view(B, L) / 10.0
    x_recon = x + 0.5  # 一様に誤差を入れる
    # 先頭半分をマスク
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, : L // 2] = True
    num_masked = int(mask.sum().item())

    sse = masked_sse(x_recon, x, mask, reduction="sum")
    mse = masked_mse(x_recon, x, mask, reduction="mean")
    bmean_mse = masked_mse(x_recon, x, mask, reduction="batch_mean")
    bmean_sse = masked_sse(x_recon, x, mask, reduction="batch_mean")

    # mean * (masked_count) = sum
    assert math.isclose(mse.item() * num_masked, sse.item(), rel_tol=1e-6, abs_tol=1e-6)
    # batch_mean は SSE / B
    assert math.isclose(bmean_mse.item(), sse.item() / B, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(bmean_sse.item(), sse.item() / B, rel_tol=1e-6, abs_tol=1e-6)


def test_masked_losses_empty_mask_edgecase():
    # 空マスク：SSE は 0、MSE(mean) は 0（実装でゼロテンソル返却）
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    x_recon = x + 1.0
    mask = torch.zeros_like(x, dtype=torch.bool)

    sse = masked_sse(x_recon, x, mask, reduction="sum")
    mse = masked_mse(x_recon, x, mask, reduction="mean")
    bmean = masked_mse(x_recon, x, mask, reduction="batch_mean")

    assert sse.item() == 0.0
    assert mse.item() == 0.0
    # batch_mean も 0
    assert bmean.item() == 0.0


def test_masked_losses_backward_works():
    B, L = 2, 5
    x = torch.randn(B, L, requires_grad=True)
    x_recon = (x + torch.randn_like(x)).detach().clone().requires_grad_(True)
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, ::2] = True  # 偶数番目だけマスク

    loss = masked_sse(x_recon, x, mask, reduction="mean")
    loss.backward()

    # 勾配が流れる（マスクされた位置に対応する recon 側に勾配が乗る）
    assert x_recon.grad is not None and x_recon.grad.abs().sum().item() > 0
    # x も計算グラフ上にあるので勾配が出る
    assert x.grad is not None and x.grad.abs().sum().item() > 0
