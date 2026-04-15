from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

__all__ = ["SpectraAugmenterConfig", "SpectraAugmenter"]


@dataclass(frozen=True)
class SpectraAugmenterConfig:
    """
    スペクトル拡張の設定。

    Parameters
    ----------
    noise_prob : float, default=0.0
        spherical Gaussian noise を適用する確率。
    noise_cos_range : tuple[float, float], default=(1.0, 1.0)
        spherical Gaussian noise の強度レンジを cosine 類似度で指定する。
        値域は (0, 1] を想定し、小さいほど強い摂動となる。
        各サンプルごとにこのレンジから一様にサンプリングした目標 cosine 類似度
        ``cos_target`` に対応する geodesic 距離を用いる。

        例:
        - (0.999, 1.000): ごく弱いノイズ
        - (0.990, 0.999): やや強いノイズ

    tilt_prob : float, default=0.0
        geodesic tilt を適用する確率。
    tilt_cos_range : tuple[float, float], default=(1.0, 1.0)
        geodesic tilt の強度レンジを cosine 類似度で指定する。
        値域は (0, 1] を想定する。
        各サンプルごとにこのレンジから一様にサンプリングした目標 cosine 類似度
        ``cos_target`` を、geodesic 回転角へ変換して適用する。
    eps : float, default=1e-12
        ゼロ除算回避用の微小値。

    Notes
    -----
    cosine 類似度 ``c`` と geodesic 回転角 ``theta`` の関係は

    - ``c = cos(theta)``
    - ``theta = arccos(c)``

    である。したがって ``c`` が 1 に近いほど弱い摂動となる。
    """

    noise_prob: float = 0.0
    noise_cos_range: tuple[float, float] = (1.0, 1.0)
    tilt_prob: float = 0.0
    tilt_cos_range: tuple[float, float] = (1.0, 1.0)
    eps: float = 1e-12


class SpectraAugmenter(nn.Module):
    """
    SNV 後スペクトルに対する球面幾何ベースの拡張器。

    Parameters
    ----------
    config : SpectraAugmenterConfig
        拡張設定。

    Notes
    -----
    - 入力は shape (B, L) の 2 次元テンソルを想定する。
    - spherical Gaussian noise / geodesic tilt はともに各サンプルの L2 ノルムを保持する。
    - 本クラスは SNV 後スペクトルのような「半径一定の球面上のデータ」を主対象とする。
    - noise は「接空間で Gaussian を引いて球面へ戻す」方式である。
    - tilt は波長方向の一次傾き基底を用い、その基底を各サンプルの接空間へ射影して適用する。
    """

    def __init__(self, config: SpectraAugmenterConfig) -> None:
        super().__init__()
        self.config = config
        self._validate_config(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        スペクトル拡張を適用する。

        Parameters
        ----------
        x : torch.Tensor
            入力スペクトル。shape は (B, L)。

        Returns
        -------
        torch.Tensor
            拡張後スペクトル。shape は (B, L)。

        Raises
        ------
        ValueError
            入力 shape が不正な場合。
        TypeError
            入力 dtype が浮動小数でない場合。
        """
        self._validate_input(x)

        if not self.training:
            return x

        out = x

        ops: list = []
        if self.config.noise_prob > 0.0:
            ops.append(self._apply_spherical_gaussian_noise)
        if self.config.tilt_prob > 0.0:
            ops.append(self._apply_geodesic_tilt)

        if len(ops) == 2 and torch.rand(()) < 0.5:
            ops[0], ops[1] = ops[1], ops[0]

        for op in ops:
            out = op(out)

        return out

    @staticmethod
    def _validate_config(config: SpectraAugmenterConfig) -> None:
        if not (0.0 <= config.noise_prob <= 1.0):
            raise ValueError("noise_prob must be in [0, 1].")
        if not (0.0 <= config.tilt_prob <= 1.0):
            raise ValueError("tilt_prob must be in [0, 1].")

        SpectraAugmenter._validate_cos_range(
            name="noise_cos_range",
            cos_range=config.noise_cos_range,
        )
        SpectraAugmenter._validate_cos_range(
            name="tilt_cos_range",
            cos_range=config.tilt_cos_range,
        )

        if config.eps <= 0.0:
            raise ValueError("eps must be positive.")

    @staticmethod
    def _validate_cos_range(name: str, cos_range: tuple[float, float]) -> None:
        cos_min, cos_max = cos_range
        if not (0.0 < cos_min <= 1.0):
            raise ValueError(f"{name}[0] must be in (0, 1].")
        if not (0.0 < cos_max <= 1.0):
            raise ValueError(f"{name}[1] must be in (0, 1].")
        if cos_min > cos_max:
            raise ValueError(f"{name} must satisfy min <= max.")

    @staticmethod
    def _validate_input(x: torch.Tensor) -> None:
        if x.ndim != 2:
            raise ValueError(f"x must be 2D (B, L), got shape={tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError("x must be a floating tensor.")

    def _apply_spherical_gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        球面版 Gaussian noise を適用する。

        Notes
        -----
        各サンプルについて:
        1. 接空間で等方 Gaussian を生成
        2. そのノルムを使って単位接方向を得る
        3. 目標 cosine 類似度から geodesic 回転角を決定
        4. 球面上で回転する

        ここで強度は cosine 類似度で制御される。
        """
        batch_size, _ = x.shape
        device = x.device
        dtype = x.dtype

        apply_mask = torch.rand(batch_size, device=device) < self.config.noise_prob
        if not torch.any(apply_mask):
            return x

        cos_target = self._sample_cos_target(
            batch_size=batch_size,
            cos_range=self.config.noise_cos_range,
            device=device,
            dtype=dtype,
        )
        theta = torch.arccos(cos_target.clamp(-1.0, 1.0))

        gaussian = torch.randn_like(x)
        tangent = self._project_to_tangent(direction=gaussian, base=x)
        tangent_unit = self._normalize_rows_with_fallback(
            vec=tangent,
            fallback=self._build_fallback_tangent(base=x),
        )

        # 接空間 Gaussian の「方向」は tangent_unit、
        # 「Gaussian らしさ」は tangent が Gaussian 射影であることに由来する。
        # ただし強度は運用上扱いやすいよう cosine 類似度で制御する。
        rotated = self._geodesic_rotate(base=x, tangent_unit=tangent_unit, theta=theta)
        return torch.where(apply_mask.unsqueeze(1), rotated, x)

    def _apply_geodesic_tilt(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_features = x.shape
        device = x.device
        dtype = x.dtype

        apply_mask = torch.rand(batch_size, device=device) < self.config.tilt_prob
        if not torch.any(apply_mask):
            return x

        cos_target_abs = self._sample_cos_target(
            batch_size=batch_size,
            cos_range=self.config.tilt_cos_range,
            device=device,
            dtype=dtype,
        )
        theta_abs = torch.arccos(cos_target_abs.clamp(-1.0, 1.0))

        sign = torch.where(
            torch.rand(batch_size, device=device) < 0.5,
            torch.tensor(-1.0, device=device, dtype=dtype),
            torch.tensor(1.0, device=device, dtype=dtype),
        )
        theta = theta_abs * sign

        tilt_basis = self._build_tilt_basis(
            num_features=num_features,
            device=device,
            dtype=dtype,
        )
        tilt_basis_batch = tilt_basis.unsqueeze(0).expand(batch_size, -1)

        tangent = self._project_to_tangent(direction=tilt_basis_batch, base=x)
        tangent_unit = self._normalize_rows_with_fallback(
            vec=tangent,
            fallback=self._build_fallback_tangent(base=x),
        )

        rotated = self._geodesic_rotate(base=x, tangent_unit=tangent_unit, theta=theta)
        return torch.where(apply_mask.unsqueeze(1), rotated, x)

    def _sample_cos_target(
        self,
        batch_size: int,
        cos_range: tuple[float, float],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        cos_min, cos_max = cos_range
        if cos_min == cos_max:
            return torch.full((batch_size,), cos_min, device=device, dtype=dtype)
        u = torch.rand(batch_size, device=device, dtype=dtype)
        return cos_min + (cos_max - cos_min) * u

    def _project_to_tangent(
        self,
        direction: torch.Tensor,
        base: torch.Tensor,
    ) -> torch.Tensor:
        eps = self.config.eps
        base_norm_sq = torch.sum(base * base, dim=1, keepdim=True).clamp_min(eps)
        coeff = torch.sum(direction * base, dim=1, keepdim=True) / base_norm_sq
        return direction - coeff * base

    def _normalize_rows_with_fallback(
        self,
        vec: torch.Tensor,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        eps = self.config.eps

        norm = torch.linalg.norm(vec, dim=1, keepdim=True)
        needs_fallback = norm.squeeze(1) <= eps

        safe_vec = torch.where(needs_fallback.unsqueeze(1), fallback, vec)
        safe_norm = torch.linalg.norm(safe_vec, dim=1, keepdim=True).clamp_min(eps)
        return safe_vec / safe_norm

    def _build_fallback_tangent(self, base: torch.Tensor) -> torch.Tensor:
        """
        接方向の退化時に用いるフォールバック方向を構成する。
        """
        batch_size, num_features = base.shape
        device = base.device
        dtype = base.dtype

        if num_features < 2:
            raise ValueError("num_features must be >= 2 for tangent construction.")

        e1 = torch.zeros((batch_size, num_features), device=device, dtype=dtype)
        e1[:, 0] = 1.0
        tangent = self._project_to_tangent(direction=e1, base=base)

        tangent_norm = torch.linalg.norm(tangent, dim=1, keepdim=True)
        bad = tangent_norm.squeeze(1) <= self.config.eps
        if torch.any(bad):
            e2 = torch.zeros((batch_size, num_features), device=device, dtype=dtype)
            e2[:, 1] = 1.0
            tangent_alt = self._project_to_tangent(direction=e2, base=base)
            tangent = torch.where(bad.unsqueeze(1), tangent_alt, tangent)

        tangent_norm = torch.linalg.norm(tangent, dim=1, keepdim=True).clamp_min(self.config.eps)
        return tangent / tangent_norm

    def _build_tilt_basis(
        self,
        num_features: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        波長方向の一次傾き基底を構成する。
        """
        if num_features < 2:
            raise ValueError("num_features must be >= 2 for tilt basis.")

        coord = torch.linspace(-1.0, 1.0, steps=num_features, device=device, dtype=dtype)
        coord = coord - coord.mean()
        norm = torch.linalg.norm(coord)
        if norm <= self.config.eps:
            raise ValueError("Failed to build tilt basis due to zero norm.")
        return coord / norm

    def _geodesic_rotate(
        self,
        base: torch.Tensor,
        tangent_unit: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        """
        base を tangent_unit 方向へ geodesic 回転する。

        Parameters
        ----------
        base : torch.Tensor
            基点。shape は (B, L)。
        tangent_unit : torch.Tensor
            単位接ベクトル。shape は (B, L)。
        theta : torch.Tensor
            回転角。shape は (B,)。
            正負により回転方向を表す。

        Returns
        -------
        torch.Tensor
            回転後テンソル。shape は (B, L)。
        """
        eps = self.config.eps

        radius = torch.linalg.norm(base, dim=1, keepdim=True).clamp_min(eps)
        unit_base = base / radius

        theta_col = theta.unsqueeze(1)
        rotated_unit = (
            torch.cos(theta_col) * unit_base
            + torch.sin(theta_col) * tangent_unit
        )
        return radius * rotated_unit