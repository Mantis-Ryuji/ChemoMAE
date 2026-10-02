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
    Seed Python, NumPy, and available Torch global random streams.

    Parameters
    ----------
    seed : int, default=42
        Value converted to int before seeding the global generators.
    fix_cudnn : bool, default=True
        Set cuDNN deterministic mode and disable cuDNN benchmarking.

    Notes
    -----
    Caller-owned Torch Generators are not seeded. PYTHONHASHSEED is set in the
    environment for subsequently started interpreters; this does not change
    hash randomization in the current interpreter. The cuDNN flags can affect
    performance and do not guarantee deterministic behavior for all operators.
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
    Toggle cuDNN deterministic and benchmark flags without changing seeds.

    Parameters
    ----------
    enable : bool, default=True
        Enable cuDNN deterministic mode and disable benchmarking when true;
        apply the opposite flags when false. Does nothing when Torch is absent.

    Notes
    -----
    Use :func:`set_global_seed` to seed global random streams. This helper does
    not call ``torch.use_deterministic_algorithms`` or control every source of
    nondeterminism.
    """
    if not _HAS_TORCH:
        return
    import torch
    torch.backends.cudnn.deterministic = bool(enable)
    torch.backends.cudnn.benchmark = not bool(enable)
