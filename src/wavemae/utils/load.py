from __future__ import annotations

import hashlib
import logging
import os
from importlib.resources import files as _pkg_files
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from wavemae.models.wave_mae import WaveMAE

__all__ = ["load_default_pretrained"]

logger = logging.getLogger(__name__)

# 固定 GitHub リポジトリ
_REPOS = {
    "library": "https://github.com/Mantis-Ryuji/WaveMAE",
    "pretraining": "https://github.com/Mantis-Ryuji/UnsupervisedWoodSegmentation-NIRHSI",
}

# 既定ハイパーパラメータ（モデル構築＆meta公開の単一情報源）
_DEFAULT_CFG: Dict[str, Any] = dict(
    # Encoder
    seq_len=256,
    d_model=384,
    nhead=6,
    num_layers=6,
    dim_feedforward=1536,
    dropout=0.1,
    use_learnable_pos=True,
    latent_dim=64,
    # Decoder (MLP)
    dec_hidden=256,
    dec_dropout=0.1,
)

# 既定の同梱重みベース名（存在しなければ assets/*.pt の先頭を使う）
_DEFAULT_WEIGHT_NAME = "wavemae_base_256.pt"


# -------------------------
# 内部ユーティリティ
# -------------------------
def _asset_dir() -> Path:
    """パッケージ同梱 assets ディレクトリへのパス（存在しない場合もある）。"""
    try:
        pkg_root = _pkg_files("wavemae")
        p = Path(pkg_root.joinpath("assets"))
        return p
    except Exception:
        return Path("src/wavemae/assets")  # editable install などでのフォールバック


def _available_weights() -> list[Path]:
    d = _asset_dir()
    if not d.exists():
        return []
    return sorted(p for p in d.glob("*.pt") if p.is_file())


def _default_weight_path() -> Optional[Path]:
    """既定の重みファイル Path を返す。見つからなければ None。"""
    # 明示名があればそれを優先
    cand = _asset_dir() / _DEFAULT_WEIGHT_NAME
    if cand.exists():
        return cand
    # ない場合は assets/*.pt の最初
    ws = _available_weights()
    return ws[0] if ws else None


def _sha256(path: Path) -> str:
    """ファイルの SHA256（hex）を返す。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_expected_sha256(path: Path) -> Optional[str]:
    """
    期待ハッシュを .sha256 から読む。
    - <file>.sha256
    - <file>.sha256.txt
    のどちらかに hex が書かれている想定（先頭のトークンを読む）。
    """
    for suffix in (".sha256", ".sha256.txt"):
        p = path.with_suffix(path.suffix + suffix)
        if p.exists():
            txt = p.read_text(encoding="utf-8").strip()
            return txt.split()[0]
    return None


def _verify_sha256_if_available(path: Path) -> Tuple[bool, Optional[str]]:
    """
    sha256 ファイルがあれば検証して (ok, warn) を返す。
    sha256 が無ければ (True, None) （＝スキップ）を返す。
    """
    exp = _read_expected_sha256(path)
    if not exp:
        return True, None
    act = _sha256(path)
    if act.lower() == exp.lower():
        return True, None
    return False, f"SHA256 mismatch: expected {exp[:12]}..., got {act[:12]}..."


def _build_model_default(device: Optional[str | torch.device]) -> WaveMAE:
    """既定構成で WaveMAE を構築し、device へ移す。"""
    dev = torch.device(device) if device is not None else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    model = WaveMAE(**_DEFAULT_CFG)
    return model.to(dev)


# -------------------------
# 公開 API
# -------------------------
def load_default_pretrained(
    weight_path: Optional[str | os.PathLike[str]] = None,
    *,
    device: Optional[str | torch.device] = None,
    strict: bool = True,
) -> Tuple[WaveMAE, Dict[str, Any]]:
    """
    既定構成の WaveMAE を構築し、可能なら同梱重みをロードして返す。

    Parameters
    ----------
    weight_path : str | Path | None
        明示的に重みファイル（.pt）を指定する場合のパス。
        None の場合はパッケージ同梱 assets から自動選択。
    device : str | torch.device | None
        'cuda' / 'cpu' / None（自動）。モデルを移す先。
    strict : bool
        state_dict の strict ロードを使うか。

    Returns
    -------
    model : WaveMAE
        既定構成で構築され、（可能なら）同梱重みをロード済みのモデル。
    meta : dict
        付随情報（公開フィールドのみ）:
          - "name"   : 重みベース名（例: "wavemae_base_256"）
          - "config" : 既定ハイパーパラメータ
                       (seq_len, d_model, nhead, num_layers, dim_feedforward,
                        dropout, use_learnable_pos, latent_dim, dec_hidden, dec_dropout)
          - "repos"  : 固定 GitHub URL（{"library": ..., "pretraining": ...}）
          - "device" : 読み込み時の map 先（文字列）
          - "strict" : strict フラグ
          - "pretrained_loaded" : bool（ロード成否）
          - "warning" : 失敗時のみ、**パスを含まない**警告メッセージ
    """
    model = _build_model_default(device=device)

    # 重みの決定
    wp: Optional[Path]
    if weight_path is not None:
        wp = Path(weight_path)
    else:
        wp = _default_weight_path()

    loaded_ok = False
    warn_msg: Optional[str] = None

    if wp is None or not wp.exists():
        warn_msg = (
            "pretrained weights not found in assets (and no explicit path given). "
            "Returning randomly-initialized model."
        )
        logger.warning(warn_msg)
    else:
        try:
            # 可能なら整合性チェック
            ok, w = _verify_sha256_if_available(wp)
            if not ok:
                warn_msg = f"weights found but integrity check failed: {w}"
                logger.warning(warn_msg)

            # ロード
            map_loc = "cpu" if (device is None and not torch.cuda.is_available()) else "cuda" if torch.cuda.is_available() else "cpu"
            state = torch.load(str(wp), map_location=map_loc)
            # PyTorch の一般的な構成：{'model': state_dict, ...} or そのまま state_dict
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            elif isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
                state = state["model"]

            model.load_state_dict(state, strict=strict)
            loaded_ok = True
        except Exception as e:
            warn_msg = f"failed to load pretrained weights: {type(e).__name__}: {e}"
            logger.warning(warn_msg)
            logger.debug("pretrained load error", exc_info=True)

    meta: Dict[str, Any] = {
        "name": os.path.splitext(os.path.basename(str(wp))) [0] if wp else "(none)",
        "config": dict(_DEFAULT_CFG),
        "repos": dict(_REPOS),
        "device": str(getattr(model, "device", device or ("cuda" if torch.cuda.is_available() else "cpu"))),
        "strict": bool(strict),
        "pretrained_loaded": bool(loaded_ok),
    }
    if not loaded_ok and warn_msg:
        meta["warning"] = warn_msg

    return model, meta
