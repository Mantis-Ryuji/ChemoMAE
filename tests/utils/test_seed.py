import os
import random
import numpy as np
import pytest

from chemomae.utils.seed import set_global_seed, enable_deterministic

try:
    import torch
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


def test_set_global_seed_python_numpy_reproducible(monkeypatch):
    # まず異なる乱数列
    random.seed(None); np.random.seed(None)
    a1 = [random.random() for _ in range(3)]
    b1 = np.random.RandomState().randn(3)

    # 固定後は再現
    set_global_seed(123, fix_cudnn=True)
    a2 = [random.random() for _ in range(3)]
    b2 = np.random.randn(3)

    # 再度同じシードで一致
    set_global_seed(123, fix_cudnn=True)
    a3 = [random.random() for _ in range(3)]
    b3 = np.random.randn(3)

    assert a2 == a3
    assert np.allclose(b2, b3)
    assert os.environ.get("PYTHONHASHSEED") == "123"


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_set_global_seed_torch_and_cudnn_flags():
    set_global_seed(7, fix_cudnn=True)
    # トーチの再現性（簡易チェック）
    r1 = torch.rand(3)
    set_global_seed(7, fix_cudnn=True)
    r2 = torch.rand(3)
    assert torch.allclose(r1, r2)

    # CUDNN フラグ
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
def test_enable_deterministic_toggles_flags_only():
    enable_deterministic(True)   # :contentReference[oaicite:3]{index=3}
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

    enable_deterministic(False)
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True
