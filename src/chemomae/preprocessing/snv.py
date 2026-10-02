"""Stateless Standard Normal Variate normalization for NumPy and Torch."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import overload

import numpy as np
import torch

__all__ = ["snv", "SNVScaler"]

SpectrumArray = np.ndarray | torch.Tensor
Statistic = float | np.ndarray | torch.Tensor


def _validate_eps(eps: object) -> float:
    if isinstance(eps, (bool, np.bool_)) or not isinstance(eps, Real):
        raise TypeError("eps must be a real number.")
    value = float(eps)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("eps must be finite and strictly positive.")
    return value


def _validate_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")
    return value


def _validate_input(x: SpectrumArray) -> None:
    if not isinstance(x, (np.ndarray, torch.Tensor)):
        raise TypeError("SNV expects a NumPy array or a Torch tensor.")
    if x.ndim not in (1, 2):
        raise ValueError(f"SNV expects shape (L,) or (N, L), got {tuple(x.shape)}.")
    if any(size == 0 for size in x.shape):
        raise ValueError("SNV requires at least one nonempty spectrum.")

    if isinstance(x, torch.Tensor):
        if x.layout != torch.strided or x.device.type == "meta":
            raise TypeError("SNV requires a dense tensor with actual data.")
        if x.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise TypeError("SNV requires float16, bfloat16, float32, or float64 tensors.")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("SNV input contains NaN or infinity.")
    else:
        if x.dtype not in (np.dtype("float16"), np.dtype("float32"), np.dtype("float64")):
            raise TypeError("SNV requires float16, float32, or float64 NumPy arrays.")
        if not np.isfinite(x).all():
            raise ValueError("SNV input contains NaN or infinity.")


def _numpy_work(x: np.ndarray, *, copy: bool) -> np.ndarray:
    dtype = np.float64 if x.dtype == np.float64 else np.float32
    return x.astype(dtype, copy=copy)


def _torch_work(x: torch.Tensor, *, copy: bool) -> torch.Tensor:
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    work = x.to(dtype=dtype)
    return work.clone() if copy else work


def _numpy_statistics_valid(mu: np.ndarray, scale: np.ndarray) -> None:
    if not np.isfinite(mu).all() or not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError(
            "SNV statistics must be finite with a positive scale; "
            "use float64 for large values or increase eps if it underflows."
        )


def _torch_statistics_valid(mu: torch.Tensor, scale: torch.Tensor) -> None:
    if not bool(torch.isfinite(mu).all() & torch.isfinite(scale).all() & (scale > 0).all()):
        raise ValueError(
            "SNV statistics must be finite with a positive scale; "
            "use float64 for large values or increase eps if it underflows."
        )


def _standardize(
    x: SpectrumArray,
    eps: float,
    *,
    copy: bool,
) -> tuple[SpectrumArray, SpectrumArray, SpectrumArray]:
    _validate_input(x)
    eps = _validate_eps(eps)
    correction = 1 if x.shape[-1] >= 2 else 0
    keepdims = x.ndim == 2

    if isinstance(x, torch.Tensor):
        work_t = _torch_work(x, copy=copy)
        # Reduce after subtracting a row anchor so constant spectra stay exact
        # and a large common offset does not dominate mean accumulation.
        anchor_t = work_t[..., :1] if keepdims else work_t[0]
        shifted_t = work_t - anchor_t
        sd_t, mean_delta_t = torch.std_mean(
            shifted_t, dim=-1, correction=correction, keepdim=keepdims
        )
        mu_t = mean_delta_t + anchor_t
        scale_t = sd_t + eps
        _torch_statistics_valid(mu_t, scale_t)
        normalized_t = (shifted_t - mean_delta_t) / scale_t
        if not bool(torch.isfinite(normalized_t).all()):
            raise ValueError("SNV output is nonfinite; use float64 or a larger eps.")
        return normalized_t, mu_t, scale_t

    work = _numpy_work(x, copy=copy)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            anchor = work[..., :1] if keepdims else work[0]
            shifted = work - anchor
            mean_delta = shifted.mean(axis=-1, keepdims=keepdims)
            mu = np.asarray(mean_delta + anchor, dtype=work.dtype)
            sd = shifted.std(axis=-1, ddof=correction, keepdims=keepdims)
            scale = np.asarray(sd + eps, dtype=work.dtype)
            _numpy_statistics_valid(mu, scale)
            normalized = (shifted - mean_delta) / scale
    except FloatingPointError as error:
        raise ValueError("SNV arithmetic overflowed; use float64 or a larger eps.") from error
    if not np.isfinite(normalized).all():
        raise ValueError("SNV output is nonfinite; use float64 or a larger eps.")
    return normalized, mu, scale


def _validate_statistic_shape(
    shape: tuple[int, ...], reference: SpectrumArray, name: str
) -> None:
    allowed = ((), (1,)) if reference.ndim == 1 else ((), (reference.shape[0], 1))
    if shape not in allowed:
        raise ValueError(f"{name} must have shape {allowed}, got {shape}.")


def _numpy_statistic(value: Statistic, reference: np.ndarray, name: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.dtype.kind != "f":
            raise TypeError(f"{name} must be a floating NumPy array or a real scalar.")
        result = value.astype(reference.dtype, copy=False)
    elif isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
        result = np.asarray(value, dtype=reference.dtype)
    else:
        raise TypeError(f"{name} must be a NumPy array or a real scalar for NumPy input.")
    _validate_statistic_shape(tuple(result.shape), reference, name)
    return result


def _torch_statistic(value: Statistic, reference: torch.Tensor, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.device != reference.device:
            raise ValueError(f"{name} must be on {reference.device}, got {value.device}.")
        if value.layout != torch.strided or not value.is_floating_point():
            raise TypeError(f"{name} must be a dense floating tensor or a real scalar.")
        result = value.to(dtype=reference.dtype)
    elif isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
        result = torch.tensor(float(value), dtype=reference.dtype, device=reference.device)
    else:
        raise TypeError(f"{name} must be a Torch tensor or a real scalar for Torch input.")
    _validate_statistic_shape(tuple(result.shape), reference, name)
    return result


@overload
def snv(x: np.ndarray, eps: float = 1e-12) -> np.ndarray: ...


@overload
def snv(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor: ...


def snv(x: SpectrumArray, eps: float = 1e-12) -> SpectrumArray:
    """Normalize each spectrum by its own mean and sample standard deviation.

    Parameters
    ----------
    x : numpy.ndarray or torch.Tensor
        Finite floating spectra with shape ``(L,)`` or ``(N, L)``. Empty
        spectra and integer or complex inputs are rejected.
    eps : float, default=1e-12
        Finite positive value added to the standard deviation.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Normalized spectra in the input framework and, for Torch, on the
        input device. Float32 and float64 are preserved; float16 and bfloat16
        are promoted to float32 for both computation and output.

    Raises
    ------
    TypeError
        If the input framework/dtype or eps type is unsupported.
    ValueError
        If shape, finite values, positive eps, or computed statistics are invalid.

    Notes
    -----
    The denominator is ``std + eps``, with ``ddof=1`` for at least two channels
    and ``ddof=0`` for one channel. Constant spectra produce zeros. Torch uses
    native differentiable operations; no detach or host transfer is performed.
    Input arrays/tensors are never modified. Validation of CUDA inputs checks
    scalar conditions and may synchronize the device.
    """
    return _standardize(x, eps, copy=False)[0]


@dataclass
class SNVScaler:
    """Stateless per-spectrum SNV with optional inverse-transform statistics.

    Parameters
    ----------
    eps : float, default=1e-12
        Finite positive value added to the standard deviation.
    copy : bool, default=True
        Whether to copy the computation input. Neither setting modifies the
        caller's data; normalization always allocates an output.
    transform_stats : bool, default=False
        Return ``(normalized, mean, scale)`` when true. The third item is
        ``std + eps``, ready for inverse transformation, rather than raw std.

    Notes
    -----
    ``fit`` learns no dataset statistics. Both frameworks share the precision
    policy of :func:`snv`. Statistics use the same framework, device, and dtype
    as the normalized output: scalars (zero-dimensional arrays/tensors) for
    ``(L,)`` inputs, and shape ``(N, 1)`` for ``(N, L)`` inputs.
    """

    eps: float = 1e-12
    copy: bool = True
    transform_stats: bool = False

    def __post_init__(self) -> None:
        self.eps = _validate_eps(self.eps)
        _validate_bool(self.copy, "copy")
        _validate_bool(self.transform_stats, "transform_stats")

    def fit(self, X: SpectrumArray, y: object = None) -> SNVScaler:
        """Validate spectra and return self without learning dataset state.

        Parameters
        ----------
        X : numpy.ndarray or torch.Tensor
            Spectra with shape ``(L,)`` or ``(N, L)``.
        y : object, default=None
            Ignored, for transformer/pipeline conventions.

        Returns
        -------
        SNVScaler
            This unchanged transformer.

        Raises
        ------
        TypeError, ValueError
            If the spectra or eps do not satisfy the SNV input contract.
        """
        _validate_eps(self.eps)
        _validate_input(X)
        return self

    @overload
    def transform(self, X: np.ndarray) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]: ...

    @overload
    def transform(
        self, X: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    def transform(
        self, X: SpectrumArray
    ) -> SpectrumArray | tuple[SpectrumArray, SpectrumArray, SpectrumArray]:
        """Apply SNV, optionally returning the mean and effective scale.

        Parameters
        ----------
        X : numpy.ndarray or torch.Tensor
            Spectra with shape ``(L,)`` or ``(N, L)``.

        Returns
        -------
        normalized : numpy.ndarray or torch.Tensor
            SNV spectra. With ``transform_stats=True``, return
            ``(normalized, mean, scale)`` in the same framework and precision.

        Raises
        ------
        TypeError, ValueError
            If the input or computed statistics violate the contract of :func:`snv`.
        """
        normalized, mu, scale = _standardize(X, self.eps, copy=self.copy)
        return (normalized, mu, scale) if self.transform_stats else normalized

    def fit_transform(
        self,
        X: SpectrumArray,
        y: object = None,
    ) -> SpectrumArray | tuple[SpectrumArray, SpectrumArray, SpectrumArray]:
        """Apply the stateless transform; ``y`` is ignored.

        Parameters
        ----------
        X : numpy.ndarray or torch.Tensor
            Spectra with shape ``(L,)`` or ``(N, L)``.
        y : object, default=None
            Ignored, for transformer/pipeline conventions.

        Returns
        -------
        numpy.ndarray, torch.Tensor, or tuple
            The same output as :meth:`transform`.
        """
        return self.transform(X)

    @overload
    def inverse_transform(self, Y: np.ndarray, *, mu: Statistic, sd: Statistic) -> np.ndarray: ...

    @overload
    def inverse_transform(self, Y: torch.Tensor, *, mu: Statistic, sd: Statistic) -> torch.Tensor: ...

    def inverse_transform(self, Y: SpectrumArray, *, mu: Statistic, sd: Statistic) -> SpectrumArray:
        """Reconstruct spectra using the mean and effective ``std + eps`` scale.

        Parameters
        ----------
        Y : numpy.ndarray or torch.Tensor
            Normalized floating spectra with shape ``(L,)`` or ``(N, L)``.
        mu : numpy.ndarray, torch.Tensor, or float
            Mean returned by :meth:`transform`, or a shared scalar mean.
        sd : numpy.ndarray, torch.Tensor, or float
            Positive effective scale returned by :meth:`transform`, including
            eps. No extra eps is added here. Statistics must use the input
            framework and, for tensors, the same device.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            Reconstructed spectra with the precision policy of :func:`snv`.

        Raises
        ------
        TypeError
            If the spectra/statistics have unsupported types or mixed frameworks.
        ValueError
            If shapes/devices disagree, scales are invalid, or arithmetic overflows.
        """
        _validate_input(Y)
        if isinstance(Y, torch.Tensor):
            work_t = _torch_work(Y, copy=self.copy)
            mean_t = _torch_statistic(mu, work_t, "mu")
            scale_t = _torch_statistic(sd, work_t, "sd")
            _torch_statistics_valid(mean_t, scale_t)
            reconstructed_t = work_t * scale_t + mean_t
            if not bool(torch.isfinite(reconstructed_t).all()):
                raise ValueError("SNV inverse output is nonfinite; use float64.")
            return reconstructed_t

        work = _numpy_work(Y, copy=self.copy)
        mean = _numpy_statistic(mu, work, "mu")
        scale = _numpy_statistic(sd, work, "sd")
        _numpy_statistics_valid(mean, scale)
        try:
            with np.errstate(over="raise", invalid="raise"):
                reconstructed = work * scale + mean
        except FloatingPointError as error:
            raise ValueError("SNV inverse arithmetic overflowed; use float64.") from error
        if not np.isfinite(reconstructed).all():
            raise ValueError("SNV inverse output is nonfinite; use float64.")
        return reconstructed

    def get_params(self, deep: bool = True) -> dict[str, float | bool]:
        """Return constructor parameters; ``deep`` is unused for this flat scaler."""
        return {"eps": self.eps, "copy": self.copy, "transform_stats": self.transform_stats}

    def set_params(self, **params: object) -> SNVScaler:
        """Validate and update constructor parameters, returning self."""
        unknown = set(params) - set(self.get_params())
        if unknown:
            raise ValueError(f"Unknown SNVScaler parameters: {sorted(unknown)}.")
        eps = _validate_eps(params.get("eps", self.eps))
        copy = _validate_bool(params.get("copy", self.copy), "copy")
        transform_stats = _validate_bool(
            params.get("transform_stats", self.transform_stats), "transform_stats"
        )
        self.eps, self.copy, self.transform_stats = eps, copy, transform_stats
        return self
