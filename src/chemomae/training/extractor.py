from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Iterator, Literal

import numpy as np
import torch
from tqdm import tqdm

from ..models.chemo_mae import ChemoMAE
from .augmenter import SpectraAugmenter


Representation = Literal["latent", "raw_latent", "normalized_latent", "cls"]
FeatureBatch = torch.Tensor | np.ndarray


@dataclass(frozen=True)
class ExtractorConfig:
    """Configure all-visible feature extraction and output storage.

    Parameters
    ----------
    device : str or torch.device, optional
        Inference device. None follows the model's device when extraction starts.
        Supported device types are CPU and CUDA.
    amp : bool, default=False
        Enable CUDA autocast. CPU AMP is not supported by this helper.
    amp_dtype : {"bf16", "fp16"}, default="bf16"
        CUDA autocast dtype. BF16 requires a supporting CUDA device.
    representation : {"latent", "raw_latent", "normalized_latent", "cls"}
        Representation selected through ChemoMAE.encode. "latent" follows the
        model's normalization setting; "normalized_latent" always applies L2
        normalization; "raw_latent" precedes normalization; "cls" precedes the
        latent projection.
    output_type : {"tensor", "numpy"}, default="tensor"
        Type returned by both streaming and aggregate methods.
    output_device : str or torch.device, optional
        Tensor storage device. None follows the inference device for tensors and
        resolves to CPU for NumPy. NumPy requires CPU output.
    output_dtype : torch.dtype, default=torch.float32
        Storage dtype, independent of inference precision. Supported values are
        float16, bfloat16, float32, and float64. NumPy cannot store bfloat16.
    save_path : str or pathlib.Path, optional
        Aggregate output destination. ".npy" saves a NumPy array; other suffixes
        save a CPU tensor with torch.save. Streaming never saves implicitly.
    progress : bool, default=False
        Display a tqdm progress bar during extraction.

    Notes
    -----
    Output conversion does not increase the precision of the encoder arithmetic.
    Model inputs are converted to the model's floating-point parameter dtype.
    """

    device: str | torch.device | None = None
    amp: bool = False
    amp_dtype: Literal["bf16", "fp16"] = "bf16"
    representation: Representation = "latent"
    output_type: Literal["tensor", "numpy"] = "tensor"
    output_device: str | torch.device | None = None
    output_dtype: torch.dtype = torch.float32
    save_path: str | Path | None = None
    progress: bool = False

    def __post_init__(self) -> None:
        if self.representation not in {"latent", "raw_latent", "normalized_latent", "cls"}:
            raise ValueError(f"Unsupported representation: {self.representation!r}.")
        if self.output_type not in {"tensor", "numpy"}:
            raise ValueError("output_type must be 'tensor' or 'numpy'.")
        if self.amp_dtype not in {"bf16", "fp16"}:
            raise ValueError("amp_dtype must be 'bf16' or 'fp16'.")
        if self.output_dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
            raise ValueError("output_dtype must be float16, bfloat16, float32, or float64.")
        needs_numpy = self.output_type == "numpy" or (
            self.save_path is not None and Path(self.save_path).suffix.lower() == ".npy"
        )
        if needs_numpy and self.output_dtype == torch.bfloat16:
            raise ValueError("NumPy output and .npy saving do not support bfloat16; choose another output_dtype.")
        if self.output_type == "numpy" and self.output_device is not None:
            if torch.device(self.output_device).type != "cpu":
                raise ValueError("NumPy output requires output_device='cpu' or None.")


class Extractor:
    """Extract features in loader order, either by batch or as one array.

    Parameters
    ----------
    model : ChemoMAE
        Model whose current weights are used. No checkpoint is loaded implicitly.
    cfg : ExtractorConfig, optional
        Extraction and storage settings.
    augmenter : SpectraAugmenter, optional
        Explicitly requested augmentation. It runs in training mode for each
        batch and may make results stochastic. None applies no augmentation.

    Notes
    -----
    Every batch uses all-visible model.encode under evaluation mode and no_grad.
    All model and augmenter training flags, including mixed child modes, are
    restored before yielding or propagating an exception. Grad and autocast
    scopes also end before yielding. Returned tensors are ordinary detached
    tensors, suitable as inputs to trainable downstream heads.

    Requested device moves persist; original devices are not restored. The same
    model must not be used concurrently for training or another extraction call.
    The caller owns loader ordering, worker state, and external I/O resources.
    """

    def __init__(
        self,
        model: ChemoMAE,
        cfg: ExtractorConfig | None = None,
        *,
        augmenter: SpectraAugmenter | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg if cfg is not None else ExtractorConfig()
        self.augmenter = augmenter
        self.device = self._resolve_device()
        self.output_device = self._resolve_output_device()
        self._validate_precision()

    @staticmethod
    def _validate_device(device: torch.device) -> None:
        if device.type not in {"cpu", "cuda"}:
            raise ValueError(f"Extractor supports CPU and CUDA, got {device.type!r}.")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA was requested, but CUDA is not available.")
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise ValueError(f"CUDA device index {device.index} is unavailable.")

    def _resolve_device(self) -> torch.device:
        device = (
            torch.device(self.cfg.device)
            if self.cfg.device is not None
            else next(self.model.parameters()).device
        )
        self._validate_device(device)
        return device

    def _resolve_output_device(self) -> torch.device:
        if self.cfg.output_device is not None:
            device = torch.device(self.cfg.output_device)
        else:
            device = torch.device("cpu") if self.cfg.output_type == "numpy" else self.device
        self._validate_device(device)
        return device

    def _validate_precision(self) -> None:
        if not self.cfg.amp:
            return
        if self.device.type != "cuda":
            raise ValueError("Extractor AMP requires a CUDA inference device; set amp=False for CPU.")
        if self.cfg.amp_dtype == "bf16":
            with torch.cuda.device(self.device):
                if not torch.cuda.is_bf16_supported():
                    raise ValueError("CUDA BF16 is unsupported on the selected device; use fp16 or disable AMP.")

    @contextmanager
    def _autocast(self) -> Iterator[None]:
        dtype = torch.bfloat16 if self.cfg.amp_dtype == "bf16" else torch.float16
        # Explicitly disable ambient autocast rather than inheriting a caller's
        # mixed-precision policy when this helper is configured with amp=False.
        with torch.autocast("cpu", enabled=False), torch.autocast(
            "cuda", enabled=self.cfg.amp, dtype=dtype
        ):
            yield

    @contextmanager
    def _temporary_modes(self) -> Iterator[None]:
        modules = list(self.model.modules())
        if self.augmenter is not None:
            modules.extend(self.augmenter.modules())
        states = {module: module.training for module in modules}
        try:
            self.model.eval()
            if self.augmenter is not None:
                self.augmenter.train()
            yield
        finally:
            # Calling train(root_flag) would destroy intentionally mixed child
            # modes. Restore each saved flag independently instead.
            for module, training in states.items():
                module.training = training

    def _to_x(self, batch: object) -> torch.Tensor:
        if isinstance(batch, (list, tuple)):
            if not batch:
                raise ValueError("A tuple/list batch must contain spectra as its first item.")
            x = batch[0]
        else:
            x = batch
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"batch must contain a torch.Tensor, got {type(x).__name__}.")
        if x.ndim != 2 or x.shape[1] != self.model.seq_len:
            raise ValueError(f"x must have shape (B, {self.model.seq_len}), got {tuple(x.shape)}.")
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor.")
        if x.layout != torch.strided or x.device.type == "meta":
            raise ValueError("x must be a dense tensor with stored values.")
        if not torch.isfinite(x).all():
            raise ValueError("x must contain finite values.")
        converted = x.to(device=self.device, dtype=next(self.model.parameters()).dtype, non_blocking=True)
        if not torch.isfinite(converted).all():
            raise ValueError("x cannot be represented finitely in the model dtype.")
        return converted

    def _empty_features(self) -> torch.Tensor:
        dimension = (
            self.model.encoder.d_model
            if self.cfg.representation == "cls"
            else self.model.encoder.latent_dim
        )
        with torch.inference_mode(False), torch.no_grad():
            return torch.empty((0, dimension), device=self.output_device, dtype=self.cfg.output_dtype)

    def _iter_tensors(self, loader: Iterable[object]) -> Generator[torch.Tensor, None, None]:
        self.device = self._resolve_device()
        self.output_device = self._resolve_output_device()
        self._validate_precision()
        with torch.inference_mode(False), torch.no_grad():
            self.model.to(self.device)
            if self.augmenter is not None:
                self.augmenter.to(self.device)
        batches = tqdm(loader, desc="Extracting", unit="batch", disable=not self.cfg.progress)
        try:
            for batch in batches:
                with torch.inference_mode(False), torch.no_grad():
                    x = self._to_x(batch)
                if x.shape[0] == 0:
                    yield self._empty_features()
                    continue
                with self._temporary_modes(), torch.inference_mode(False), torch.no_grad(), self._autocast():
                    x_input = self.augmenter(x) if self.augmenter is not None else x
                    if not isinstance(x_input, torch.Tensor) or x_input.shape != x.shape:
                        raise ValueError("augmenter must return a tensor with the same shape as its input.")
                    if not x_input.is_floating_point() or not torch.isfinite(x_input).all():
                        raise ValueError("augmenter must return finite floating-point values.")
                    features = self.model.encode(x_input, representation=self.cfg.representation)
                    if not torch.isfinite(features).all():
                        raise ValueError("encoder returned nonfinite features.")
                    features = features.detach().to(device=self.output_device, dtype=self.cfg.output_dtype)
                    if not torch.isfinite(features).all():
                        raise ValueError("features cannot be represented finitely in output_dtype.")
                yield features
        finally:
            batches.close()

    def _as_output(self, features: torch.Tensor) -> FeatureBatch:
        return features.numpy() if self.cfg.output_type == "numpy" else features

    def iter_transform(self, loader: Iterable[object]) -> Generator[FeatureBatch, None, None]:
        """Yield one feature batch per input batch without dataset aggregation.

        Tensor batches or tuples/lists with spectra first are accepted. Spectra
        must be floating-point tensors of shape (B, model.seq_len). Remaining
        tuple/list items are ignored. Empty batches yield correctly shaped empty
        features; an empty loader yields nothing. No masks are sampled.

        save_path is not used. Write each yielded batch to a caller-owned sink or
        consume it on its requested device. If stopping early, call the
        generator's close() (or use contextlib.closing) to release the progress
        bar promptly; all mode scopes already ended before the yield.
        """
        batches = self._iter_tensors(loader)
        try:
            for features in batches:
                yield self._as_output(features)
        finally:
            batches.close()

    def transform(self, loader: Iterable[object]) -> FeatureBatch:
        """Collect all features in loader order, optionally saving the result.

        This method stores every feature batch and allocates the concatenated
        result on output_device. Use iter_transform for bounded feature storage.
        An empty loader returns shape (0, representation_dimension) with the
        requested type, device, and dtype.
        """
        features = list(self._iter_tensors(loader))
        with torch.inference_mode(False), torch.no_grad():
            result = torch.cat(features, dim=0) if features else self._empty_features()
        if self.cfg.save_path is not None:
            path = Path(self.cfg.save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with torch.inference_mode(False), torch.no_grad():
                cpu_result = result.cpu()
                if path.suffix.lower() == ".npy":
                    with path.open("wb") as destination:
                        np.save(destination, cpu_result.numpy())
                else:
                    torch.save(cpu_result, path)
        return self._as_output(result)

    def extract(self, loader: Iterable[object]) -> FeatureBatch:
        """Alias for transform, including optional aggregate saving."""
        return self.transform(loader)

    def __call__(self, loader: Iterable[object]) -> FeatureBatch:
        """Alias for transform, including optional aggregate saving."""
        return self.transform(loader)
