from __future__ import annotations

import hashlib
import os
import logging
from importlib.resources import files as _pkg_files
from typing import Any, Dict, Optional, Tuple

import torch

from wavemae.models.wave_mae import WaveMAE

__all__ = ["load_default_pretrained"]

logger = logging.getLogger(__name__)

# 固定 GitHub リポジトリ
_REPOS = {
    "library": "https://github.com/Mantis-Ryuji/WaveMAE",                                  # ライブラリ
    "pretraining": "https://github.com/Mantis-Ryuji/UnsupervisedWoodSegmentation-NIRHSI",  # 事前学習内容
}


# ----------------------------- 内部ユーティリティ ----------------------------- #
def _asset_path(relative: str) -> str:
    """パッケージ同梱 assets への相対パスを解決（wheel に同梱必須）。"""
    return str(_pkg_files("wavemae").joinpath(f"assets/{relative}"))

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _read_expected_sha256(sha256_path: str) -> str:
    with open(sha256_path, "rt", encoding="utf-8") as f:
        line = f.readline().strip()
    if not line:
        raise ValueError(f"Empty sha256 file: {sha256_path}")
    return line.split()[0]

def _verify_sha256(file_path: str, sha256_path: str) -> None:
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

def _build_model_default(device: Optional[str | torch.device]) -> WaveMAE:
    """既定構成で WaveMAE を構築し、device へ移す。"""
    dev = torch.device(device) if device is not None else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    model = WaveMAE(
        seq_len=256,
        d_model=256,
        nhead=4,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        use_learnable_pos=True,
        latent_dim=64,
        dec_hidden=256,
        dec_dropout=0.1,
    )
    return model.to(dev)

def _introspect_shape(model: WaveMAE) -> Dict[str, Any]:
    """モデル主要形状を辞書で返す（将来の互換性のため動的に取得）。"""
    enc = getattr(model, "encoder", None)
    dec = getattr(model, "decoder", None)
    return {
        "seq_len": getattr(model, "seq_len", None),
        "latent_dim": getattr(getattr(enc, "to_latent", None), "out_features", None) if enc else None,
        "d_model": getattr(getattr(enc, "token_proj", None), "out_features", None) if enc else None,
        "num_layers": len(getattr(getattr(enc, "encoder", None), "layers", [])) if enc else None,
        "nhead": (
            getattr(getattr(getattr(enc, "encoder", None), "layers", [None])[0].self_attn, "num_heads", None)
            if enc and getattr(enc, "encoder", None) and getattr(enc.encoder, "layers", None) else None
        ),
        "decoder_hidden": (getattr(getattr(dec, "net", [None])[0], "out_features", None) if dec else None),
        "n_blocks": getattr(model, "n_blocks", None),
        "n_mask": getattr(model, "n_mask", None),
    }


# ------------------------------ 公開関数 ------------------------------ #
def load_default_pretrained(
    *,
    device: Optional[str | torch.device] = None,
    strict: bool = True,
    verify_hash: bool = True,
) -> Tuple[WaveMAE, Dict[str, Any]]:
    r"""
    Load the default pretrained WaveMAE shipped with this package.

    概要
    ----
    パッケージ同梱（`assets/`）の 既定重みを、固定構成の `WaveMAE` にロードして返す。
    返り値は `(model, meta)`。`meta` には モデル名・主要形状・GitHub リポジトリ URL を含む。
    もし重みが見つからない / 読み込めない場合でも、未初期化（ランダム初期化）の model と meta を返す。
    その際は `logger.warning` に 「事前学習済み重みをloadできませんでした」を出力し、
    `meta["pretrained_loaded"] = False` を立てる。

    Parameters
    ----------
    device : str | torch.device, optional (default: auto)
        モデルを配置するデバイス。未指定なら CUDA 可なら "cuda"、それ以外は "cpu"。
    strict : bool, default=True
        `model.load_state_dict` の strict フラグ。
    verify_hash : bool, default=True
        `assets/*.pt` に対する `.sha256` 検証を有効化（.sha256 が存在する場合のみ照合）。

    Returns
    -------
    model : WaveMAE
        既定構成で構築され、同梱重みをロード済み（もしくは未ロード）のモデル。
    meta : dict
        付随情報（最低限以下のキーを含む）:
          - `"name"` : 重みファイル名（拡張子除く）。例 `"wavemae_base_256"`
          - `"shape"` : 主要ハイパラの要約（`seq_len`, `latent_dim`, `d_model`, `num_layers`, `nhead`, …）
          - `"repos"` : 固定 GitHub URL 辞書 `{"library": ..., "pretraining": ...}`
          - `"weight_path"` : 使用を試みた重みファイルのパス（存在しない場合も含む）
          - `"sha256_path"` : 参照した .sha256 のパス（存在しない場合は空文字）
          - `"device"` : 読み込み時の `map_location`
          - `"strict"` : `strict` の値
          - `"pretrained_loaded"` : bool（重み読み込みに成功したか）
          - `"warning"` : 失敗時のみ、警告メッセージ文字列

    Notes
    -----
    - 既定構成:
      `seq_len=256, d_model=256, nhead=4, num_layers=4, dim_feedforward=1024,
       dropout=0.1, use_learnable_pos=True, latent_dim=64, dec_hidden=256, dec_dropout=0.1`
    - `.sha256` が同梱されていない環境ではハッシュ照合は自動的にスキップされます。

    Examples
    --------
    >>> from wavemae.load import load_default_pretrained
    >>> model, meta = load_default_pretrained(device="cuda")
    >>> meta["name"], meta["shape"]["latent_dim"], meta["repos"]["library"]
    ('wavemae_base_256', 64, 'https://github.com/Mantis-Ryuji/WaveMAE')
    """
    # 1) モデル構築
    model = _build_model_default(device=device)

    # 2) アセット解決（存在しなくても meta にパスは入れる）
    weight_file = "wavemae_base_256.pt"   # 同梱する既定重みファイル名
    weight_path = _asset_path(weight_file)
    sha256_path = _asset_path(weight_file + ".sha256")

    # 3) ロード試行
    loaded_ok = False
    warn_msg = ""

    try:
        if not os.path.exists(weight_path):
            raise FileNotFoundError(weight_path)

        # (任意) 整合性検証
        if verify_hash and os.path.exists(sha256_path):
            _verify_sha256(weight_path, sha256_path)

        # 読み込み
        map_location = model.device if hasattr(model, "device") else (
            torch.device(device) if device is not None else "cpu"
        )
        state = torch.load(weight_path, map_location=map_location)
        if isinstance(state, dict) and "state_dict" in state:  # checkpoint 形式でもOK
            state = state["state_dict"]
        model.load_state_dict(state, strict=strict)
        loaded_ok = True

    except Exception as e:
        warn_msg = f"事前学習済み重みをloadできませんでした: {e}"
        logger.warning(warn_msg)

    # 4) meta を整える（repo は固定）
    map_location_str = str(getattr(model, "device", device or ("cuda" if torch.cuda.is_available() else "cpu")))
    meta: Dict[str, Any] = {
        "name": os.path.splitext(os.path.basename(weight_path))[0],  # "wavemae_base_256"
        "shape": _introspect_shape(model),
        "repos": dict(_REPOS),  # {"library": ..., "pretraining": ...}
        "weight_path": weight_path,
        "sha256_path": sha256_path if os.path.exists(sha256_path) else "",
        "device": map_location_str,
        "strict": bool(strict),
        "pretrained_loaded": bool(loaded_ok),
    }
    if not loaded_ok:
        meta["warning"] = warn_msg

    return model, meta
