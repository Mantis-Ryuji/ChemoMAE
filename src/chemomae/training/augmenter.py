from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

__all__ = ["SpectraAugmenterConfig", "SpectraAugmenter"]


def _validate_probability(name: str, value: float) -> None:
    """確率値を検証する。

    Parameters
    ----------
    name : str
        パラメータ名。
    value : float
        検証対象値。

    Returns
    -------
    None

    Raises
    ------
    ValueError
        値が [0, 1] に入らない場合。
    """
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}.")


def _validate_range(name: str, value: tuple[float, float]) -> None:
    """2要素範囲を検証する。

    Parameters
    ----------
    name : str
        パラメータ名。
    value : tuple[float, float]
        検証対象範囲。

    Returns
    -------
    None

    Raises
    ------
    ValueError
        範囲が不正な場合。
    """
    if len(value) != 2:
        raise ValueError(f"{name} must have length 2, got {len(value)}.")

    low, high = value
    if low > high:
        raise ValueError(f"{name} must satisfy low <= high, got {value}.")


def _validate_angle_deg_range(name: str, value: tuple[float, float]) -> None:
    """角度範囲を degree 単位で検証する。

    Parameters
    ----------
    name : str
        パラメータ名。
    value : tuple[float, float]
        角度範囲。単位は degree。

    Returns
    -------
    None

    Raises
    ------
    ValueError
        範囲が不正な場合。
    """
    _validate_range(name=name, value=value)

    low, high = value
    if low < 0.0:
        raise ValueError(f"{name}[0] must be >= 0.0, got {low}.")
    if high > 180.0:
        raise ValueError(f"{name}[1] must be <= 180.0, got {high}.")


@dataclass(frozen=True)
class SpectraAugmenterConfig:
    """SNV 後スペクトル向け augmentation 設定。

    Parameters
    ----------
    shift_prob : float, default=0.5
        fractional shift を適用する確率。
    shift_delta_range : tuple[float, float], default=(-2.0, 2.0)
        fractional shift の候補生成に使う shift 量の範囲。
        単位はチャネル index。
    shift_angle_deg_range : tuple[float, float], default=(0.5, 3.0)
        shift 候補方向へ移動する角度範囲。
        単位は degree。
    noise_prob : float, default=0.5
        tangent Gaussian noise を適用する確率。
    noise_angle_deg_range : tuple[float, float], default=(0.5, 3.0)
        noise 方向へ回転する角度範囲。
        単位は degree。
    shuffle_order_per_batch : bool, default=False
        shift と noise の適用順をバッチごとにランダム化するかどうか。
    recenter_after_each_op : bool, default=True
        各 augmentation 適用後にサンプル平均を 0 へ戻すかどうか。
    renorm_to_input_norm : bool, default=True
        各 augmentation 適用後に入力ノルムへ戻すかどうか。
    eps : float, default=1e-8
        数値安定化用の微小値。

    Notes
    -----
    angle 指定は degree 単位で行う。
    内部計算では radian に変換して使用する。
    """

    shift_prob: float = 0.5
    shift_delta_range: tuple[float, float] = (-2.0, 2.0)
    shift_angle_deg_range: tuple[float, float] = (0.5, 3.0)

    noise_prob: float = 0.5
    noise_angle_deg_range: tuple[float, float] = (0.5, 3.0)

    shuffle_order_per_batch: bool = False
    recenter_after_each_op: bool = True
    renorm_to_input_norm: bool = True
    eps: float = 1.0e-8

    def __post_init__(self) -> None:
        """設定値を検証する。

        Returns
        -------
        None

        Raises
        ------
        ValueError
            設定値が不正な場合。
        """
        _validate_probability("shift_prob", self.shift_prob)
        _validate_probability("noise_prob", self.noise_prob)

        _validate_range("shift_delta_range", self.shift_delta_range)
        _validate_angle_deg_range("shift_angle_deg_range", self.shift_angle_deg_range)
        _validate_angle_deg_range("noise_angle_deg_range", self.noise_angle_deg_range)

        if self.eps <= 0.0:
            raise ValueError(f"eps must be positive, got {self.eps}.")


class SpectraAugmenter(nn.Module):
    """SNV 後スペクトルに対する shift + noise augmentation。

    Parameters
    ----------
    config : SpectraAugmenterConfig
        augmentation 設定。

    Notes
    -----
    本クラスは `nn.Module` として実装されているため、Trainer 側で
    `augmenter.to(device)` および `augmenter(x)` として利用できる。

    学習時のみ augmentation を適用し、評価時には入力をそのまま返す。
    """

    def __init__(self, config: SpectraAugmenterConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """augmentation を適用する。

        Parameters
        ----------
        x : torch.Tensor
            入力スペクトル。shape は (B, L)。

        Returns
        -------
        torch.Tensor
            augmentation 後スペクトル。shape は (B, L)。

        Raises
        ------
        TypeError
            入力 dtype が浮動小数でない場合。
        ValueError
            入力 shape が不正な場合。
        """
        self._validate_input(x)

        if not self.training:
            return x

        batch_size, num_features = x.shape
        if batch_size < 1:
            raise ValueError("x must contain at least one sample.")
        if num_features < 2:
            raise ValueError(f"x must have at least 2 features, got {num_features}.")

        out = x

        for op_name in self._sample_op_order(device=x.device):
            if op_name == "shift":
                out = self._apply_shift(out)
            elif op_name == "noise":
                out = self._apply_noise(out)
            else:
                raise RuntimeError(f"Unknown augmentation op: {op_name}")

        return out

    @staticmethod
    def _validate_input(x: torch.Tensor) -> None:
        """入力テンソルを検証する。

        Parameters
        ----------
        x : torch.Tensor
            検証対象テンソル。

        Returns
        -------
        None

        Raises
        ------
        TypeError
            入力 dtype が浮動小数でない場合。
        ValueError
            入力 shape が不正な場合。
        """
        if x.ndim != 2:
            raise ValueError(f"x must be 2D (B, L), got shape={tuple(x.shape)}.")
        if not x.is_floating_point():
            raise TypeError("x must be a floating tensor.")

    def _sample_op_order(self, device: torch.device) -> list[str]:
        """augmentation の適用順を返す。

        Parameters
        ----------
        device : torch.device
            乱数生成に使うデバイス。

        Returns
        -------
        list[str]
            適用順。
        """
        ops = ["shift", "noise"]

        if not self.config.shuffle_order_per_batch:
            return ops

        perm = torch.randperm(len(ops), device=device).tolist()
        return [ops[int(i)] for i in perm]

    @staticmethod
    def _sample_apply_mask(
        batch_size: int,
        prob: float,
        device: torch.device,
    ) -> torch.Tensor:
        """サンプル単位の適用マスクを生成する。

        Parameters
        ----------
        batch_size : int
            バッチサイズ。
        prob : float
            適用確率。
        device : torch.device
            生成先デバイス。

        Returns
        -------
        torch.Tensor
            shape (B,) の bool マスク。
        """
        return torch.rand(batch_size, device=device) < prob

    @staticmethod
    def _sample_uniform(
        batch_size: int,
        value_range: tuple[float, float],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """一様乱数を生成する。

        Parameters
        ----------
        batch_size : int
            バッチサイズ。
        value_range : tuple[float, float]
            サンプリング範囲。
        device : torch.device
            生成先デバイス。
        dtype : torch.dtype
            生成 dtype。

        Returns
        -------
        torch.Tensor
            shape (B,) の乱数。
        """
        low, high = value_range
        return torch.empty(batch_size, device=device, dtype=dtype).uniform_(low, high)

    def _sample_angle_rad(
        self,
        batch_size: int,
        angle_deg_range: tuple[float, float],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """degree 指定の角度範囲から radian 角度をサンプルする。

        Parameters
        ----------
        batch_size : int
            バッチサイズ。
        angle_deg_range : tuple[float, float]
            degree 単位の角度範囲。
        device : torch.device
            生成先デバイス。
        dtype : torch.dtype
            生成 dtype。

        Returns
        -------
        torch.Tensor
            shape (B,) の radian 角度。
        """
        angle_deg = self._sample_uniform(
            batch_size=batch_size,
            value_range=angle_deg_range,
            device=device,
            dtype=dtype,
        )
        return angle_deg * (math.pi / 180.0)

    @staticmethod
    def _center(x: torch.Tensor) -> torch.Tensor:
        """各サンプルを平均 0 に再中心化する。

        Parameters
        ----------
        x : torch.Tensor
            入力テンソル。shape は (B, L)。

        Returns
        -------
        torch.Tensor
            再中心化後テンソル。
        """
        return x - x.mean(dim=1, keepdim=True)

    def _renorm_like(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """各サンプルを参照テンソルのノルムへ再正規化する。

        Parameters
        ----------
        x : torch.Tensor
            対象テンソル。shape は (B, L)。
        ref : torch.Tensor
            参照テンソル。shape は (B, L)。

        Returns
        -------
        torch.Tensor
            再正規化後テンソル。
        """
        ref_norm = torch.linalg.norm(ref, dim=1, keepdim=True).clamp_min(self.config.eps)
        x_norm = torch.linalg.norm(x, dim=1, keepdim=True)

        bad = x_norm <= self.config.eps
        scaled = x * (ref_norm / x_norm.clamp_min(self.config.eps))
        return torch.where(bad, ref, scaled)

    def _reproject(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """SNV 後スペクトルの幾何へ再投影する。

        Parameters
        ----------
        x : torch.Tensor
            対象テンソル。shape は (B, L)。
        ref : torch.Tensor
            参照テンソル。shape は (B, L)。

        Returns
        -------
        torch.Tensor
            再投影後テンソル。
        """
        out = x
        if self.config.recenter_after_each_op:
            out = self._center(out)
        if self.config.renorm_to_input_norm:
            out = self._renorm_like(out, ref=ref)
        return out

    def _project_to_tangent(self, base: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """方向ベクトルを接空間へ射影する。

        Parameters
        ----------
        base : torch.Tensor
            基準点。shape は (B, L)。
        direction : torch.Tensor
            射影対象方向。shape は (B, L)。

        Returns
        -------
        torch.Tensor
            接空間へ射影された方向。
        """
        direction_centered = self._center(direction)

        denom = torch.sum(base * base, dim=1, keepdim=True).clamp_min(self.config.eps)
        coeff = torch.sum(direction_centered * base, dim=1, keepdim=True) / denom
        return direction_centered - coeff * base

    def _rotate_along_tangent_angle(
        self,
        base: torch.Tensor,
        direction: torch.Tensor,
        angle_deg_range: tuple[float, float],
    ) -> torch.Tensor:
        """接空間方向へ指定角度だけ回転する。

        Parameters
        ----------
        base : torch.Tensor
            基準スペクトル。shape は (B, L)。
        direction : torch.Tensor
            回転方向の候補。shape は (B, L)。
        angle_deg_range : tuple[float, float]
            回転角度範囲。単位は degree。

        Returns
        -------
        torch.Tensor
            回転後スペクトル。
        """
        batch_size = base.shape[0]
        device = base.device
        dtype = base.dtype

        tangent = self._project_to_tangent(base=base, direction=direction)
        tangent_norm = torch.linalg.norm(tangent, dim=1, keepdim=True)

        valid = tangent_norm.squeeze(1) > self.config.eps
        if not torch.any(valid):
            return base

        base_norm = torch.linalg.norm(base, dim=1, keepdim=True).clamp_min(self.config.eps)
        unit_base = base / base_norm
        unit_tangent = tangent / tangent_norm.clamp_min(self.config.eps)

        angle = self._sample_angle_rad(
            batch_size=batch_size,
            angle_deg_range=angle_deg_range,
            device=device,
            dtype=dtype,
        ).unsqueeze(1)

        rotated = base_norm * (
            torch.cos(angle) * unit_base
            + torch.sin(angle) * unit_tangent
        )

        out = base.clone()
        out[valid] = rotated[valid]
        return self._reproject(out, ref=base)

    def _slerp_to_target_angle(
        self,
        base: torch.Tensor,
        target: torch.Tensor,
        angle_deg_range: tuple[float, float],
    ) -> torch.Tensor:
        """target 方向へ指定角度だけ球面補間する。

        Parameters
        ----------
        base : torch.Tensor
            基準スペクトル。shape は (B, L)。
        target : torch.Tensor
            補間先候補スペクトル。shape は (B, L)。
        angle_deg_range : tuple[float, float]
            base から移動する角度範囲。単位は degree。

        Returns
        -------
        torch.Tensor
            球面補間後スペクトル。
        """
        batch_size = base.shape[0]
        device = base.device
        dtype = base.dtype

        base_norm = torch.linalg.norm(base, dim=1, keepdim=True).clamp_min(self.config.eps)
        target_norm = torch.linalg.norm(target, dim=1, keepdim=True).clamp_min(self.config.eps)

        u = base / base_norm
        v = target / target_norm

        dot = torch.sum(u * v, dim=1).clamp(-1.0, 1.0)
        omega = torch.arccos(dot)
        omega_safe = omega.clamp_min(self.config.eps)

        sampled_angle = self._sample_angle_rad(
            batch_size=batch_size,
            angle_deg_range=angle_deg_range,
            device=device,
            dtype=dtype,
        )

        target_angle = torch.minimum(sampled_angle, omega)
        t = (target_angle / omega_safe).clamp(0.0, 1.0)

        sin_omega = torch.sin(omega_safe).clamp_min(self.config.eps)

        coeff_base = torch.sin((1.0 - t) * omega_safe) / sin_omega
        coeff_target = torch.sin(t * omega_safe) / sin_omega

        slerped_unit = coeff_base.unsqueeze(1) * u + coeff_target.unsqueeze(1) * v

        lerped_unit = (1.0 - t).unsqueeze(1) * u + t.unsqueeze(1) * v
        lerped_unit = lerped_unit / torch.linalg.norm(
            lerped_unit,
            dim=1,
            keepdim=True,
        ).clamp_min(self.config.eps)

        near_collinear = omega <= 1.0e-6
        out_unit = torch.where(
            near_collinear.unsqueeze(1),
            lerped_unit,
            slerped_unit,
        )

        out = base_norm * out_unit
        return self._reproject(out, ref=base)

    def _fractional_shift_batch(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """バッチ単位で fractional shift を適用する。

        Parameters
        ----------
        x : torch.Tensor
            入力スペクトル。shape は (B, L)。
        delta : torch.Tensor
            shift 量。shape は (B,)。

        Returns
        -------
        torch.Tensor
            shift 後スペクトル。shape は (B, L)。

        Raises
        ------
        ValueError
            入力 shape が不正な場合。
        """
        if x.ndim != 2:
            raise ValueError(f"x must be 2D, got shape={tuple(x.shape)}.")
        if delta.ndim != 1:
            raise ValueError(f"delta must be 1D, got shape={tuple(delta.shape)}.")
        if x.shape[0] != delta.shape[0]:
            raise ValueError(
                "delta.shape[0] must match batch size, "
                f"got {delta.shape[0]} and {x.shape[0]}."
            )

        batch_size, num_features = x.shape
        device = x.device
        dtype = x.dtype

        grid = torch.arange(num_features, device=device, dtype=dtype).unsqueeze(0)
        src_pos = grid - delta.unsqueeze(1)

        left = torch.floor(src_pos).to(torch.long)
        right = left + 1

        left_clamped = left.clamp(0, num_features - 1)
        right_clamped = right.clamp(0, num_features - 1)

        alpha = (src_pos - torch.floor(src_pos)).clamp(0.0, 1.0)

        x_left = torch.gather(x, dim=1, index=left_clamped)
        x_right = torch.gather(x, dim=1, index=right_clamped)

        return (1.0 - alpha) * x_left + alpha * x_right

    def _apply_shift(self, x: torch.Tensor) -> torch.Tensor:
        """fractional shift augmentation を適用する。

        Parameters
        ----------
        x : torch.Tensor
            入力スペクトル。shape は (B, L)。

        Returns
        -------
        torch.Tensor
            augmentation 後スペクトル。
        """
        cfg = self.config
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        apply_mask = self._sample_apply_mask(
            batch_size=batch_size,
            prob=cfg.shift_prob,
            device=device,
        )
        if not torch.any(apply_mask):
            return x

        rows = torch.nonzero(apply_mask, as_tuple=False).view(-1)

        deltas = self._sample_uniform(
            batch_size=batch_size,
            value_range=cfg.shift_delta_range,
            device=device,
            dtype=dtype,
        )

        x_selected = x[rows]
        delta_selected = deltas[rows]

        candidate = self._fractional_shift_batch(
            x=x_selected,
            delta=delta_selected,
        )
        candidate = self._reproject(candidate, ref=x_selected)

        shifted = self._slerp_to_target_angle(
            base=x_selected,
            target=candidate,
            angle_deg_range=cfg.shift_angle_deg_range,
        )

        out = x.clone()
        out[rows] = shifted
        return out

    def _apply_noise(self, x: torch.Tensor) -> torch.Tensor:
        """tangent Gaussian noise augmentation を適用する。

        Parameters
        ----------
        x : torch.Tensor
            入力スペクトル。shape は (B, L)。

        Returns
        -------
        torch.Tensor
            augmentation 後スペクトル。
        """
        cfg = self.config
        batch_size, num_features = x.shape
        device = x.device
        dtype = x.dtype

        apply_mask = self._sample_apply_mask(
            batch_size=batch_size,
            prob=cfg.noise_prob,
            device=device,
        )
        if not torch.any(apply_mask):
            return x

        rows = torch.nonzero(apply_mask, as_tuple=False).view(-1)

        direction = torch.randn(
            (batch_size, num_features),
            device=device,
            dtype=dtype,
        )

        out = x.clone()
        out[rows] = self._rotate_along_tangent_angle(
            base=x[rows],
            direction=direction[rows],
            angle_deg_range=cfg.noise_angle_deg_range,
        )
        return out