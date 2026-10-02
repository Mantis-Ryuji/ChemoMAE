from __future__ import annotations
import os
import random
from typing import TypedDict, cast
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


class NumPyRNGState(TypedDict):
    """Primitive/tensor representation of NumPy's legacy global MT19937 stream."""

    algorithm: str
    keys: torch.Tensor
    position: int
    has_gauss: int
    cached_gaussian: float


class RNGState(TypedDict):
    """Versioned standard global streams; independent generators are excluded."""

    format_version: int
    python: tuple[int, tuple[int, ...], float | None]
    numpy: NumPyRNGState
    torch_cpu: torch.Tensor
    torch_cuda: list[torch.Tensor]


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


def capture_rng_state() -> RNGState:
    """Capture Python, NumPy global, Torch CPU, and initialized CUDA streams.

    The result contains only primitives and tensors. CUDA is not initialized
    solely to capture state. Independent NumPy/Torch generators, DataLoader
    workers, and external services remain the caller's responsibility.
    """
    import torch

    numpy_state = np.random.get_state()
    return {
        "format_version": 1,
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.tensor(numpy_state[1].astype(np.int64), dtype=torch.int64),
            "position": int(numpy_state[2]), "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [value.cpu().clone() for value in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_initialized() else [],
    }


def _validate_rng_state(state: object, *, restore_cuda: bool) -> RNGState:
    """Check streams with isolated generators before changing any global state."""
    import torch

    if not isinstance(state, dict) or type(state.get("format_version")) is not int or state["format_version"] != 1:
        raise ValueError("Unsupported or missing RNG format_version; expected 1.")
    numpy_state = state.get("numpy")
    if not isinstance(numpy_state, dict) or numpy_state.get("algorithm") != "MT19937":
        raise ValueError("RNG state must include NumPy's global MT19937 stream.")
    keys = numpy_state.get("keys")
    if not isinstance(keys, torch.Tensor) or keys.dtype != torch.int64 or keys.device.type != "cpu" or keys.shape != (624,) or keys.layout != torch.strided:
        raise ValueError("NumPy RNG keys must be a CPU int64 tensor of length 624.")
    if bool(((keys < 0) | (keys > 2**32 - 1)).any()):
        raise ValueError("NumPy RNG keys must fit in uint32.")
    position = numpy_state.get("position")
    has_gauss = numpy_state.get("has_gauss")
    cached = numpy_state.get("cached_gaussian")
    if type(position) is not int or not 0 <= position <= 624 or type(has_gauss) is not int or has_gauss not in {0, 1}:
        raise ValueError("NumPy RNG position/cache flag is invalid.")
    if not isinstance(cached, float) or not np.isfinite(cached):
        raise ValueError("NumPy RNG cached Gaussian must be finite.")
    python_state = state.get("python")
    if not isinstance(python_state, tuple) or len(python_state) != 3:
        raise ValueError("Python RNG state must be a complete three-part tuple.")
    python_cache = python_state[2]
    if python_cache is not None and (not isinstance(python_cache, float) or not np.isfinite(python_cache)):
        raise ValueError("Python RNG cached Gaussian must be finite or None.")
    cpu_state = state.get("torch_cpu")
    cuda_states = state.get("torch_cuda")
    if not isinstance(cuda_states, list):
        raise ValueError("Torch CUDA RNG state must be a list.")
    for value in [cpu_state, *cuda_states]:
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1 or value.device.type != "cpu" or value.layout != torch.strided:
            raise ValueError("Torch RNG states must be CPU byte vectors.")
    try:
        random.Random(0).setstate(python_state)
        torch.Generator(device="cpu").set_state(cpu_state)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("Python/Torch CPU RNG state is invalid.") from error
    if restore_cuda and cuda_states:
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise ValueError("CUDA RNG restoration requires the saved number of CUDA devices.")
        try:
            for index, value in enumerate(cuda_states):
                torch.Generator(device=f"cuda:{index}").set_state(value)
        except RuntimeError as error:
            raise ValueError("Torch CUDA RNG state is invalid.") from error
    return cast(RNGState, state)


def restore_rng_state(state: object, *, restore_cuda: bool = True) -> None:
    """Restore a validated global RNG snapshot without touching owned generators.

    CUDA restoration requires matching device count and compatible software.
    Set restore_cuda=False explicitly for CPU transfer; CUDA streams are then
    left unchanged. This cannot guarantee cross-device/version reproducibility.
    """
    import torch

    checked = _validate_rng_state(state, restore_cuda=restore_cuda)
    numpy_state = checked["numpy"]
    random.setstate(checked["python"])
    np.random.set_state((
        numpy_state["algorithm"], numpy_state["keys"].numpy().astype(np.uint32),
        numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"],
    ))
    torch.set_rng_state(checked["torch_cpu"])
    if restore_cuda and checked["torch_cuda"]:
        torch.cuda.set_rng_state_all(checked["torch_cuda"])
