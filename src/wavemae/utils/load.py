from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, Any

import torch
from importlib.resources import files as _pkg_files

# WaveMAE 本体
try:
    from wavemae.models.wave_mae import WaveMAE
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Failed to import WaveMAE. Ensure wavemae.models.wave_mae is installed "
        "and importable within the package."
    ) from e

__all__ = [
    "WaveMAEConfig",
    "get_default_config",
    "build_model",
    "load_pretrained",
    "load_default_pretrained",
    "load_weight_with_sha256",
    "verify_sha256",
]

# ------------------------------------------------------------
# 1) コンフィグ定義（固定値）
# ------------------------------------------------------------
@dataclass(frozen=True)
class WaveMAEConfig:
    """Fixed configuration for WaveMAE used by packaged weights."""
    # --- Encoder (Transformer) ---
    seq_len: int = 256           # 入力系列長（波長チャンネル数）
    d_model: int = 256           # 埋め込み次元
    nhead: int = 4               # マルチヘッド数（d_model % nhead == 0）
    num_layers: int = 4          # Transformer Encoder 層数
    dim_feedforward: int = 1024  # FFN の中間次元
    dropout: float = 0.1         # Dropout 率
    use_learnable_pos: bool = True  # True: 学習可能pos, False: サイン波固定
    latent_dim: int = 64         # L2正規化済み潜在表現の次元

    # --- Decoder (MLP: z -> R^L) ---
    dec_hidden: int = 256
    dec_dropout: float = 0.1


def get_default_config() -> WaveMAEConfig:
    """固定コンフィグを返す（assets の学習済み重みはこの設定に対応）."""
    return WaveMAEConfig()


# ------------------------------------------------------------
# 2) 参照ユーティリティ（SHA256）
# ------------------------------------------------------------
def _sha256_file(path: str) -> str:
    """ファイルの SHA256 (hex) を計算."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_expected_sha256(sha256_path: str) -> str:
    """
    .sha256 テキストから期待ハッシュを読む。
    形式は「<hex> [filename]」どちらでも可（先頭トークンを採用）。
    """
    with open(sha256_path, "rt", encoding="utf-8") as f:
        line = f.readline().strip()
    if not line:
        raise ValueError(f"Empty sha256 file: {sha256_path}")
    return line.split()[0]


def verify_sha256(file_path: str, sha256_path: str) -> None:
    """file_path の実ハッシュが sha256_path の期待ハッシュと一致するか検証。"""
    actual = _sha256_file(file_path)
    expected = _read_expected_sha256(sha256_path)
    if actual.lower() != expected.lower():
        raise ValueError(
            "SHA256 mismatch for pretrained weights:\n"
            f"  file:     {file_path}\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            "The file may be corrupted. Reinstall the package or replace the asset."
        )


# ------------------------------------------------------------
# 3) パッケージ内アセットのパス解決
# ------------------------------------------------------------
def _asset_path(relative: str) -> str:
    """
    パッケージ同梱ファイルを `importlib.resources` で解決。
    - wheel に含めるには pyproject.toml の package-data 設定が必要。
    """
    return str(_pkg_files("wavemae").joinpath(f"assets/{relative}"))


# ------------------------------------------------------------
# 4) モデル構築 & 重みロード
# ------------------------------------------------------------
def build_model(cfg: Optional[WaveMAEConfig] = None, *, device: Optional[str | torch.device] = None) -> WaveMAE:
    """
    固定コンフィグで WaveMAE を構築。
    - device 未指定時は 'cuda' が利用可能なら cuda、なければ cpu。
    """
    cfg = cfg or get_default_config()
    dev = torch.device(device) if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # WaveMAE のコンストラクタが受け取る引数に合わせて渡す
    model = WaveMAE(
        seq_len=cfg.seq_len,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        use_learnable_pos=cfg.use_learnable_pos,
        latent_dim=cfg.latent_dim,
        dec_hidden=cfg.dec_hidden,
        dec_dropout=cfg.dec_dropout,
    )
    return model.to(dev)


def load_pretrained(
    *,
    cfg: Optional[WaveMAEConfig] = None,
    device: Optional[str | torch.device] = None,
    strict: bool = True,
    verify_hash: bool = True,
) -> Tuple[WaveMAE, Dict[str, Any]]:
    """
    パッケージ同梱のデモ重みをロードして WaveMAE を返す。
    - cfg: 省略時は固定コンフィグ（get_default_config）
    - device: 省略時は自動判定（cuda→cpu）
    - strict: state_dict の厳密一致
    - verify_hash: .sha256 による整合性検証を行うか
    戻り値: (model, meta) … meta には version/config/path などの情報を入れる
    """
    cfg = cfg or get_default_config()
    model = build_model(cfg, device=device)

    # アセットのパス取得
    weight_path = _asset_path("wavemae_base_256.pt")
    sha256_path = _asset_path("wavemae_base_256.pt.sha256")

    # 整合性検証（任意）
    if verify_hash:
        verify_sha256(weight_path, sha256_path)

    # ロード
    map_location = model.device if hasattr(model, "device") else (torch.device(device) if device is not None else "cpu")
    state = torch.load(weight_path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:
        # checkpoint 形式に対応
        state = state["state_dict"]

    # state_dict ロード
    model.load_state_dict(state, strict=strict)

    meta = {
        "config": asdict(cfg),
        "weight_path": weight_path,
        "sha256_path": sha256_path,
        "device": str(map_location),
        "strict": strict,
    }
    return model, meta


# ------------------------------------------------------------
# 5) ユーティリティ：ファイル経由で任意重みを検証付きロード
# ------------------------------------------------------------
def load_weight_with_sha256(
    model: WaveMAE,
    weight_path: str,
    sha256_path: Optional[str] = None,
    *,
    strict: bool = True,
) -> None:
    """
    任意の weight_path（ローカル or ダウンロード済み）をモデルへロード。
    - sha256_path が与えられた場合は整合性検証を行う。
    """
    if sha256_path is not None:
        verify_sha256(weight_path, sha256_path)

    state = torch.load(weight_path, map_location=model.device if hasattr(model, "device") else "cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=strict)


# ------------------------------------------------------------
# 6) 便利関数：一発でモデルを用意
# ------------------------------------------------------------
def load_default_pretrained(
    *,
    device: Optional[str | torch.device] = None,
    strict: bool = True,
) -> WaveMAE:
    """
    最短パス：固定コンフィグ + パッケージ同梱のデモ重みをロードして返す。
    """
    model, _ = load_pretrained(cfg=None, device=device, strict=strict, verify_hash=True)
    return model