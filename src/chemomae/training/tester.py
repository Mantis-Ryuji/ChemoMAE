from __future__ import annotations

import json
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Literal, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from .augmenter import SpectraAugmenter


@dataclass
class TesterConfig:
    r"""
    Configure dataset-wide reconstruction evaluation.

    Parameters
    ----------
    out_dir : str or pathlib.Path, default="runs"
        History directory, created only when logging is enabled.
    device : str or torch.device, optional
        Evaluation device. None follows model parameters, or CPU for models
        without parameters. Supported devices are CPU and CUDA.
    amp : bool, default=False
        Enable explicit CUDA autocast. CPU AMP is rejected.
    amp_dtype : {"bf16", "fp16"}, default="bf16"
        CUDA autocast dtype; bf16 requires device support.
    loss_type : {"sse", "mse"}, default="mse"
        Squared-error loss name. The reduction determines its scaling.
    loss_region : {"masked", "all"}, default="masked"
        Evaluate hidden positions or the full spectrum.
    reduction : {"sum", "mean", "batch_mean"}, default="mean"
        Dataset total SSE, SSE per selected element, or SSE per spectrum.
        Aggregation is independent of batch sizes and variable mask counts.
    fixed_visible : torch.Tensor, optional
        Boolean visible mask with shape (L,), (1, L), or current (B, L).
    log_history : bool, default=True
        Append successful evaluations to JSON history.
    history_filename : str, default="test_history.json"
        History filename below out_dir.
    progress : bool, default=True
        Display evaluation progress.
    """

    out_dir: str | Path = "runs"
    device: str | torch.device | None = None
    amp: bool = False
    amp_dtype: Literal["bf16", "fp16"] = "bf16"

    loss_type: Literal["sse", "mse"] = "mse"
    loss_region: Literal["masked", "all"] = "masked"
    reduction: Literal["sum", "mean", "batch_mean"] = "mean"
    fixed_visible: Optional[torch.Tensor] = None

    log_history: bool = True
    history_filename: str = "test_history.json"
    progress: bool = True


class Tester:
    r"""
    Evaluate reconstruction of clean targets from optional augmented inputs.

    Parameters
    ----------
    model : torch.nn.Module
        Model returning (reconstruction, latent, visible_mask).
        Fixed-mask evaluation additionally requires encoder and decoder.
    cfg : TesterConfig, optional
        Evaluation settings; defaults to TesterConfig().
    augmenter : SpectraAugmenter, optional
        Explicit opt-in to stochastic input perturbations. The reconstruction
        target remains the unaugmented spectrum.

    Notes
    -----
    The model is moved to the selected device, without changing its mode at
    construction. Evaluation temporarily sets the model to eval and a supplied
    augmenter to train. All original submodule modes are restored on success
    or failure. Input tensors are cast to the model parameter dtype.

    Masked evaluation with no selected elements and loaders without spectra
    raise ValueError rather than reporting a successful zero loss. Use
    loss_region="all" with n_mask=0 for full-spectrum evaluation.
    No checkpoint selection or fitting occurs.
    """

    __test__ = False

    def __init__(
        self,
        model: nn.Module,
        cfg: TesterConfig | None = None,
        *,
        augmenter: SpectraAugmenter | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg if cfg is not None else TesterConfig()
        reference = next(self.model.parameters(), None)
        self.device = torch.device(self.cfg.device) if self.cfg.device is not None else (
            reference.device if reference is not None else torch.device("cpu")
        )
        self.input_dtype = reference.dtype if reference is not None else torch.float32
        if self.cfg.loss_region not in {"masked", "all"}:
            raise ValueError("loss_region must be 'masked' or 'all'")
        if self.cfg.loss_type not in {"sse", "mse"}:
            raise ValueError("loss_type must be 'sse' or 'mse'")
        if self.cfg.reduction not in {"sum", "mean", "batch_mean"}:
            raise ValueError("reduction must be 'sum', 'mean', or 'batch_mean'")
        if self.cfg.amp_dtype not in {"bf16", "fp16"}:
            raise ValueError("amp_dtype must be 'bf16' or 'fp16'")
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Tester supports CPU and CUDA devices")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if self.cfg.amp and self.device.type != "cuda":
            raise ValueError("AMP evaluation requires a CUDA device")
        if self.cfg.amp and self.cfg.amp_dtype == "bf16":
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError("bf16 AMP is unsupported on the selected CUDA device")
        with torch.inference_mode(False), torch.no_grad():
            self.model.to(self.device)
        self.augmenter = augmenter

        self.out_dir = Path(self.cfg.out_dir)
        if self.cfg.log_history:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.out_dir / self.cfg.history_filename

        # Preserve existing records and report malformed histories explicitly.
        self._history: list[Dict[str, Any]] = []
        if self.cfg.log_history and self.history_path.exists():
            data = json.loads(self.history_path.read_text(encoding="utf-8"))
            if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
                raise ValueError(f"evaluation history must contain a list of records: {self.history_path}")
            self._history = data

    def _append_history(self, rec: Dict[str, Any]) -> None:
        """Append one evaluation record using a temporary file and replacement."""
        if not self.cfg.log_history:
            return

        self._history.append(rec)
        tmp = self.history_path.with_suffix(self.history_path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self._history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.history_path)

    @contextmanager
    def _autocast(self) -> Iterator[None]:
        """Apply configured precision explicitly, disabling ambient autocast otherwise."""
        dtype = torch.bfloat16 if self.cfg.amp_dtype == "bf16" else torch.float16
        with torch.autocast("cpu", enabled=False), torch.autocast(
            "cuda", dtype=dtype, enabled=self.cfg.amp,
        ):
            yield

    def _to_x(self, batch: object) -> torch.Tensor:
        """Read a floating (B, L) tensor or the first item of a tuple/list batch."""
        if isinstance(batch, (list, tuple)) and not batch:
            raise ValueError("a tuple/list batch must contain spectra as its first item")
        x = batch[0] if isinstance(batch, (list, tuple)) else batch

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"batch must contain a torch.Tensor, got {type(x)}.")
        if x.ndim != 2:
            raise ValueError(f"x must be 2D (B, L), got shape={tuple(x.shape)}.")
        if not x.is_floating_point():
            raise TypeError("x must be a floating tensor.")
        if not torch.isfinite(x).all():
            raise ValueError("input spectra contain NaN or infinity")

        converted = x.to(self.device, dtype=self.input_dtype, non_blocking=True)
        if not torch.isfinite(converted).all():
            raise ValueError("input spectra cannot be represented finitely in the model dtype")
        return converted

    def _prepare_fixed_visible(
        self,
        fixed_visible: torch.Tensor,
        *,
        batch_size: int,
        num_features: int,
    ) -> torch.Tensor:
        """Broadcast a boolean (L,), (1, L), or current (B, L) visible mask."""
        if not isinstance(fixed_visible, torch.Tensor):
            raise TypeError("fixed_visible must be a boolean torch.Tensor")
        visible_mask = fixed_visible.to(self.device)

        if visible_mask.dtype != torch.bool:
            raise TypeError(
                "fixed_visible must be a bool tensor where True means visible, "
                f"got dtype={visible_mask.dtype}."
            )

        if visible_mask.ndim == 1:
            if visible_mask.shape[0] != num_features:
                raise ValueError(
                    "1D fixed_visible must have shape (L,), "
                    f"got {tuple(visible_mask.shape)} for L={num_features}."
                )
            visible_mask = visible_mask.unsqueeze(0).expand(batch_size, num_features)

        elif visible_mask.ndim == 2:
            if visible_mask.shape[1] != num_features:
                raise ValueError(
                    "2D fixed_visible must have shape (B, L), "
                    f"got {tuple(visible_mask.shape)} for L={num_features}."
                )

            if visible_mask.shape[0] == 1:
                visible_mask = visible_mask.expand(batch_size, num_features)
            elif visible_mask.shape[0] != batch_size:
                raise ValueError(
                    "2D fixed_visible batch dimension must match current batch size "
                    "or be 1 for broadcasting, "
                    f"got {visible_mask.shape[0]} and batch_size={batch_size}."
                )

        else:
            raise ValueError(
                "fixed_visible must have shape (L,) or (B, L), "
                f"got shape={tuple(visible_mask.shape)}."
            )

        return visible_mask

    def __call__(self, data_loader: Iterable) -> float:
        """
        Evaluate an iterable of spectra and return the dataset-wide reduction.

        Sum accumulates selected squared errors; mean divides by the number
        of selected elements; batch_mean divides by the number of spectra.
        These definitions hold for both loss_type names. An empty evaluation
        or a batch with no selected elements raises ValueError and is not logged.
        """
        module_modes = [(module, module.training) for module in self.model.modules()]
        augmenter_modes = [] if self.augmenter is None else [
            (module, module.training) for module in self.augmenter.modules()
        ]
        total = torch.zeros((), device=self.device, dtype=torch.float64)
        elements = torch.zeros((), device=self.device, dtype=torch.int64)
        count = 0
        fixed_visible = self.cfg.fixed_visible

        try:
            self.model.eval()
            if self.augmenter is not None:
                with torch.inference_mode(False), torch.no_grad():
                    self.augmenter.to(self.device)
                self.augmenter.train()
            with torch.inference_mode(), self._autocast():
                for batch in tqdm(data_loader, desc="Testing", unit="batch", disable=not self.cfg.progress):
                    x = self._to_x(batch)
                    if x.size(0) == 0:
                        continue
                    x_input = self.augmenter(x) if self.augmenter is not None else x
                    if not isinstance(x_input, torch.Tensor) or x_input.shape != x.shape:
                        raise ValueError("augmenter must return a tensor with the input shape")
                    batch_size, num_features = x_input.shape

                    if fixed_visible is None:
                        with self._autocast():
                            x_recon, _, visible_mask = self.model(x_input)
                    else:
                        visible_mask = self._prepare_fixed_visible(
                            fixed_visible,
                            batch_size=batch_size,
                            num_features=num_features,
                        )
                        with self._autocast():
                            z = self.model.encoder(x_input, visible_mask)
                            x_recon = self.model.decoder(z)

                    if not isinstance(x_recon, torch.Tensor) or x_recon.shape != x.shape:
                        raise ValueError("model reconstruction must have the input shape")
                    if not x_recon.is_floating_point() or not torch.isfinite(x_recon).all():
                        raise ValueError("model reconstruction must contain finite floating values")
                    if not isinstance(visible_mask, torch.Tensor) or (
                        visible_mask.dtype != torch.bool or visible_mask.shape != x.shape
                    ):
                        raise ValueError("model visible_mask must be boolean with the input shape")
                    selected = ~visible_mask if self.cfg.loss_region == "masked" else torch.ones_like(
                        visible_mask, dtype=torch.bool,
                    )
                    if not selected.any():
                        raise ValueError(
                            "no elements selected for evaluation; use loss_region='all' "
                            "for n_mask=0 or an all-visible mask"
                        )
                    # Sum selected errors once; reduce across the whole loader.
                    # This is invariant to batch sizes and variable mask counts.
                    error = (x_recon.to(torch.float64) - x.to(torch.float64)).square()[selected]
                    if not torch.isfinite(error).all():
                        raise ValueError("selected squared errors are nonfinite")
                    total += error.sum(dtype=torch.float64)
                    elements += selected.sum(dtype=torch.int64)
                    count += batch_size

        finally:
            for module, was_training in module_modes + augmenter_modes:
                module.training = was_training

        if count == 0:
            raise ValueError("evaluation loader contains no spectra")
        if self.cfg.reduction == "sum":
            avg = total.item()
        elif self.cfg.reduction == "mean":
            avg = (total / elements).item()
        else:
            avg = (total / count).item()
        if not torch.isfinite(total) or not math.isfinite(avg):
            raise ValueError("dataset reconstruction loss is nonfinite")
        self._append_history(
            {
                "phase": "test",
                "test_loss": float(avg),
                "loss_type": self.cfg.loss_type,
                "loss_region": self.cfg.loss_region,
                "reduction": self.cfg.reduction,
                "samples": count,
                "selected_elements": int(elements.item()),
                "augmented": self.augmenter is not None,
            }
        )
        return float(avg)
