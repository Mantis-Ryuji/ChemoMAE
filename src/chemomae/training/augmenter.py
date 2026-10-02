from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

__all__ = ["SpectraAugmenterConfig", "SpectraAugmenter"]


def _validate_probability(name: str, value: float) -> None:
    """Validate a probability in the closed interval [0, 1]."""
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}.")


def _validate_range(name: str, value: tuple[float, float]) -> None:
    """Validate finite endpoints of a two-element interval with low <= high."""
    if len(value) != 2:
        raise ValueError(f"{name} must have length 2, got {len(value)}.")

    low, high = value
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError(f"{name} endpoints must be finite, got {value}.")
    if low > high:
        raise ValueError(f"{name} must satisfy low <= high, got {value}.")


def _validate_angle_deg_range(name: str, value: tuple[float, float]) -> None:
    """Validate an ordered angle interval within [0, 180] degrees."""
    _validate_range(name=name, value=value)

    low, high = value
    if low < 0.0:
        raise ValueError(f"{name}[0] must be >= 0.0, got {low}.")
    if high > 180.0:
        raise ValueError(f"{name}[1] must be <= 180.0, got {high}.")


@dataclass(frozen=True)
class SpectraAugmenterConfig:
    """Configure fractional shift and tangent Gaussian noise after SNV.

    Parameters
    ----------
    shift_prob : float, default=0.5
        Per-spectrum probability of fractional shift.
    shift_delta_range : tuple of float, default=(-2.0, 2.0)
        Uniform shift interval in channel-index units.
    noise_prob : float, default=0.5
        Per-spectrum probability of tangent Gaussian noise.
    noise_angle_deg_range : tuple of float, default=(0.5, 3.0)
        Uniform geodesic noise-angle interval in degrees.
    shuffle_order_per_batch : bool, default=True
        Randomize the shift/noise order once per batch.
    recenter_after_each_op : bool, default=True
        Restore zero mean after each applied operation.
    renorm_to_input_norm : bool, default=True
        Restore the input row norm after each applied operation.
    eps : float, default=1e-8
        Finite positive threshold for normalization and degenerate directions.

    Notes
    -----
    Shift strength is controlled only by channel displacement. Noise strength
    is controlled by spherical angle. These settings do not define a universal
    experimental recipe; callers choose and record their perturbation protocol.
    """

    shift_prob: float = 0.5
    shift_delta_range: tuple[float, float] = (-2.0, 2.0)

    noise_prob: float = 0.5
    noise_angle_deg_range: tuple[float, float] = (0.5, 3.0)

    shuffle_order_per_batch: bool = True
    recenter_after_each_op: bool = True
    renorm_to_input_norm: bool = True
    eps: float = 1.0e-8

    def __post_init__(self) -> None:
        """Validate probabilities, ranges, angles, and the positive epsilon."""
        _validate_probability("shift_prob", self.shift_prob)
        _validate_probability("noise_prob", self.noise_prob)

        _validate_range("shift_delta_range", self.shift_delta_range)
        _validate_angle_deg_range("noise_angle_deg_range", self.noise_angle_deg_range)

        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError(f"eps must be finite and positive, got {self.eps}.")


class SpectraAugmenter(nn.Module):
    """Apply fractional shift and tangent Gaussian noise to SNV spectra.

    Parameters
    ----------
    config : SpectraAugmenterConfig
        Augmentation strengths, application probabilities, and reprojection.
    generator : torch.Generator, optional
        Caller-owned default stream. It must match the active input device.
        A generator supplied to forward overrides this default for that call.

    Notes
    -----
    Training mode enables augmentation; evaluation mode returns the input
    unchanged without drawing random numbers. Every random operation uses the
    resolved generator. With no explicit/default generator, the input device's
    global Torch stream is used without reseeding or replacing it.

    The generator is an ordinary reference, not a parameter or buffer. Module
    to() does not move it, and state_dict() does not include its state. Callers
    own its seed, device, lifetime, and get_state()/set_state() persistence.
    """

    def __init__(
        self,
        config: SpectraAugmenterConfig,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator or None.")
        self.config = config
        self.generator = generator

    def forward(
        self,
        x: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Apply augmentation using the selected random stream.

        Parameters
        ----------
        x : torch.Tensor, shape (B, L)
            Floating-point spectra. Active augmentation requires B >= 1 and L >= 2.
        generator : torch.Generator, optional
            Per-call stream; None uses the module's default stream. Active
            augmentation requires the generator and input devices to match.

        Returns
        -------
        torch.Tensor, shape (B, L)
            Augmented spectra with input dtype/device, or x itself in eval mode.

        Raises
        ------
        TypeError
            Input is not a floating tensor or the active generator has an invalid type.
        ValueError
            Input shape, active batch size, feature count, or generator device is invalid.

        Notes
        -----
        Reproducible sequences require matching inputs, config, modes, batch
        partitioning, call order, generator state, dtype, device, and software.
        Equal seeds do not promise equal CPU/CUDA or cross-version outputs.
        Shift coordinates and interpolation use at least float32 precision;
        the returned spectra retain the input dtype.
        """
        self._validate_input(x)

        if not self.training:
            return x

        generator = generator if generator is not None else self.generator
        self._validate_generator(generator, x.device)
        batch_size, num_features = x.shape
        if batch_size < 1:
            raise ValueError("x must contain at least one sample.")
        if num_features < 2:
            raise ValueError(f"x must have at least 2 features, got {num_features}.")

        out = x

        for op_name in self._sample_op_order(device=x.device, generator=generator):
            if op_name == "shift":
                out = self._apply_shift(out, generator=generator)
            elif op_name == "noise":
                out = self._apply_noise(out, generator=generator)
            else:
                raise RuntimeError(f"Unknown augmentation op: {op_name}")

        return out

    @staticmethod
    def _validate_generator(generator: torch.Generator | None, device: torch.device) -> None:
        """Reject an active generator whose type or device does not match x."""
        if generator is None:
            return
        if not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator or None.")
        generator_device = torch.device(generator.device)
        if generator_device.type != device.type:
            raise ValueError(f"generator device {generator_device} must match input device {device}.")
        if device.type == "cuda":
            generator_index = generator_device.index
            if generator_index is None:
                generator_index = torch.cuda.current_device()
            input_index = device.index if device.index is not None else torch.cuda.current_device()
            if generator_index != input_index:
                raise ValueError(f"generator device {generator_device} must match input device {device}.")

    @staticmethod
    def _validate_input(x: torch.Tensor) -> None:
        """Validate a floating-point tensor with shape (B, L)."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a floating torch.Tensor.")
        if x.ndim != 2:
            raise ValueError(f"x must be 2D (B, L), got shape={tuple(x.shape)}.")
        if not x.is_floating_point():
            raise TypeError("x must be a floating tensor.")

    def _sample_op_order(
        self, device: torch.device, *, generator: torch.Generator | None = None
    ) -> list[str]:
        """Return fixed or generator-sampled operation order for one batch."""
        ops = ["shift", "noise"]

        if not self.config.shuffle_order_per_batch:
            return ops

        perm = torch.randperm(len(ops), device=device, generator=generator).tolist()
        return [ops[int(i)] for i in perm]

    @staticmethod
    def _sample_apply_mask(
        batch_size: int,
        prob: float,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw a per-spectrum Bernoulli application mask."""
        return torch.rand(batch_size, device=device, generator=generator) < prob

    @staticmethod
    def _sample_uniform(
        batch_size: int,
        value_range: tuple[float, float],
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw uniform values from the configured interval."""
        low, high = value_range
        return torch.empty(batch_size, device=device, dtype=dtype).uniform_(low, high, generator=generator)

    def _sample_angle_rad(
        self,
        batch_size: int,
        angle_deg_range: tuple[float, float],
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw angles in degrees and convert them to radians."""
        angle_deg = self._sample_uniform(
            batch_size=batch_size,
            value_range=angle_deg_range,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        return angle_deg * (math.pi / 180.0)

    @staticmethod
    def _center(x: torch.Tensor) -> torch.Tensor:
        """Subtract each row's mean."""
        return x - x.mean(dim=1, keepdim=True)

    def _renorm_like(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Restore reference row norms, retaining the reference for collapsed rows."""
        ref_norm = torch.linalg.norm(ref, dim=1, keepdim=True).clamp_min(
            self.config.eps
        )
        x_norm = torch.linalg.norm(x, dim=1, keepdim=True)

        bad = x_norm <= self.config.eps
        scaled = x * (ref_norm / x_norm.clamp_min(self.config.eps))
        return torch.where(bad, ref, scaled)

    def _reproject(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Apply configured mean and norm reprojection."""
        out = x
        if self.config.recenter_after_each_op:
            out = self._center(out)
        if self.config.renorm_to_input_norm:
            out = self._renorm_like(out, ref=ref)
        return out

    def _project_to_tangent(
        self,
        base: torch.Tensor,
        direction: torch.Tensor,
    ) -> torch.Tensor:
        """Center a direction and remove its component parallel to the base."""
        direction_centered = self._center(direction)

        denom = torch.sum(base * base, dim=1, keepdim=True).clamp_min(
            self.config.eps
        )
        coeff = torch.sum(direction_centered * base, dim=1, keepdim=True) / denom
        return direction_centered - coeff * base

    def _rotate_along_tangent_angle(
        self,
        base: torch.Tensor,
        direction: torch.Tensor,
        angle_deg_range: tuple[float, float],
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Rotate along a valid tangent direction by a generator-sampled angle."""
        batch_size = base.shape[0]
        device = base.device
        dtype = base.dtype

        tangent = self._project_to_tangent(base=base, direction=direction)
        tangent_norm = torch.linalg.norm(tangent, dim=1, keepdim=True)

        valid = tangent_norm.squeeze(1) > self.config.eps
        if not torch.any(valid):
            return base

        base_norm = torch.linalg.norm(base, dim=1, keepdim=True).clamp_min(
            self.config.eps
        )
        unit_base = base / base_norm
        unit_tangent = tangent / tangent_norm.clamp_min(self.config.eps)

        angle = self._sample_angle_rad(
            batch_size=batch_size,
            angle_deg_range=angle_deg_range,
            device=device,
            dtype=dtype,
            generator=generator,
        ).unsqueeze(1)

        rotated = base_norm * (
            torch.cos(angle) * unit_base + torch.sin(angle) * unit_tangent
        )

        out = base.clone()
        out[valid] = rotated[valid]
        return self._reproject(out, ref=base)

    def _fractional_shift_batch(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """Shift each spectrum by delta channels using clamped linear interpolation."""
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape={tuple(x.shape)}.")
        if delta.ndim != 1:
            raise ValueError(f"delta must be 1D, got shape={tuple(delta.shape)}.")
        if x.shape[0] != delta.shape[0]:
            raise ValueError(
                "delta.shape[0] must match batch size, "
                f"got {delta.shape[0]} and {x.shape[0]}."
            )

        _, num_features = x.shape
        device = x.device
        coordinate_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32

        # Half precision cannot represent fractional positions across an ordinary
        # spectral grid; keep coordinates precise before gathering/rounding output.
        grid = torch.arange(num_features, device=device, dtype=coordinate_dtype).unsqueeze(0)
        src_pos = grid - delta.to(dtype=coordinate_dtype).unsqueeze(1)

        left = torch.floor(src_pos).to(torch.long)
        right = left + 1

        left_clamped = left.clamp(0, num_features - 1)
        right_clamped = right.clamp(0, num_features - 1)

        alpha = (src_pos - torch.floor(src_pos)).clamp(0.0, 1.0)

        x_left = torch.gather(x, dim=1, index=left_clamped)
        x_right = torch.gather(x, dim=1, index=right_clamped)

        shifted = (1.0 - alpha) * x_left + alpha * x_right
        return shifted.to(dtype=x.dtype)

    def _apply_shift(self, x: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        """Apply generator-sampled fractional shifts to the selected rows."""
        cfg = self.config
        batch_size = x.shape[0]
        device = x.device
        coordinate_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32

        apply_mask = self._sample_apply_mask(
            batch_size=batch_size,
            prob=cfg.shift_prob,
            device=device,
            generator=generator,
        )
        if not torch.any(apply_mask):
            return x

        rows = torch.nonzero(apply_mask, as_tuple=False).view(-1)

        deltas = self._sample_uniform(
            batch_size=batch_size,
            value_range=cfg.shift_delta_range,
            device=device,
            dtype=coordinate_dtype,
            generator=generator,
        )

        x_selected = x[rows]
        delta_selected = deltas[rows]

        shifted = self._fractional_shift_batch(
            x=x_selected,
            delta=delta_selected,
        )
        shifted = self._reproject(shifted, ref=x_selected)

        out = x.clone()
        out[rows] = shifted
        return out

    def _apply_noise(self, x: torch.Tensor, *, generator: torch.Generator | None = None) -> torch.Tensor:
        """Apply generator-sampled tangent Gaussian noise to the selected rows."""
        cfg = self.config
        batch_size, num_features = x.shape
        device = x.device
        dtype = x.dtype

        apply_mask = self._sample_apply_mask(
            batch_size=batch_size,
            prob=cfg.noise_prob,
            device=device,
            generator=generator,
        )
        if not torch.any(apply_mask):
            return x

        rows = torch.nonzero(apply_mask, as_tuple=False).view(-1)

        direction = torch.randn(
            (batch_size, num_features),
            device=device,
            dtype=dtype,
            generator=generator,
        )

        out = x.clone()
        out[rows] = self._rotate_along_tangent_angle(
            base=x[rows],
            direction=direction[rows],
            angle_deg_range=cfg.noise_angle_deg_range,
            generator=generator,
        )
        return out
