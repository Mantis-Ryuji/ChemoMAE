from __future__ import annotations
import os
import random
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def set_global_seed(seed: int = 42, *, fix_cudnn: bool = True) -> None:
    """
    Python/NumPy(/PyTorch) の乱数をまとめて固定。
    fix_cudnn=True のとき、CUDNN の決定論モードも設定（速度低下の可能性あり）。
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if _HAS_TORCH:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if fix_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def enable_deterministic(enable: bool = True) -> None:
    """
    決定論性フラグだけを切り替え（CUDNN）。Seed の固定は set_global_seed() で行う。
    """
    if not _HAS_TORCH:
        return
    import torch
    torch.backends.cudnn.deterministic = bool(enable)
    torch.backends.cudnn.benchmark = not bool(enable)
