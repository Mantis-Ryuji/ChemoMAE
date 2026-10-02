from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn

__all__ = []

class EMACallback:
    r"""
    Track an exponential moving average of floating model state.

    Floating parameters and buffers are cloned from ``model.state_dict()``.
    Each update combines the previous shadow with the current model state.
    ``apply_to`` copies the shadow into a model; it does not restore raw weights
    afterward, so callers must manage any temporary application themselves.

    Parameters
    ----------
    model : nn.Module
        Model supplying floating state entries to track.
    decay : float, default=0.999
        Weight assigned to the previous shadow in each update.

    Attributes
    ----------
    decay : float
        Weight assigned to the previous shadow.
    shadow : dict[str, torch.Tensor]
        Cloned floating parameters and buffers being tracked.

    Methods
    -------
    register(model: nn.Module)
        Initialize the shadow from the model's current floating state.
    update(model: nn.Module)
        Update the shadow using the model's current floating state.
    apply_to(model: nn.Module)
        Copy tracked floating state into the model; preserve nonfloating state.
    state_dict() -> dict
        Return decay and cloned shadow entries for checkpoint storage.
    load_state_dict(state: dict)
        Restore decay and cloned shadow entries; devices are not changed.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self.register(model)

    @torch.no_grad()
    def register(self, model: nn.Module):
        self.shadow = {k: p.detach().clone() for k, p in model.state_dict().items() if p.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, p in model.state_dict().items():
            if p.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        model.load_state_dict({**model.state_dict(), **self.shadow}, strict=False)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"decay": self.decay, "shadow": {k: v.detach().clone() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: Dict[str, torch.Tensor]):
        self.decay = float(state.get("decay", self.decay))
        sh = state.get("shadow", {})
        self.shadow = {k: v.detach().clone() for k, v in sh.items()}
