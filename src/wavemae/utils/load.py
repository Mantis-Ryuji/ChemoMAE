from __future__ import annotations
import hashlib, os, logging
from importlib.resources import files as _pkg_files
from typing import Any, Dict, Optional, Tuple
import torch
from wavemae.models.wave_mae import WaveMAE

__all__ = ["load_default_pretrained"]
logger = logging.getLogger(__name__)

_REPOS = {
    "library": "https://github.com/Mantis-Ryuji/WaveMAE",
    "pretraining": "https://github.com/Mantis-Ryuji/UnsupervisedWoodSegmentation-NIRHSI",
}

def _asset_path(relative: str) -> str:
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
    dev = torch.device(device) if device is not None else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    model = WaveMAE(
        seq_len=256, d_model=256, nhead=4, num_layers=4,
        dim_feedforward=1024, dropout=0.1, use_learnable_pos=True,
        latent_dim=64, dec_hidden=256, dec_dropout=0.1,
    )
    return model.to(dev)

def _introspect_shape(model: WaveMAE) -> Dict[str, Any]:
    enc = getattr(model, "encoder", None); dec = getattr(model, "decoder", None)
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
    - パッケージ同梱（``assets/``）の既定重みを固定構成の `WaveMAE`
      にロードし、``(model, meta)`` を返します。
    - 重みが見つからない / 読み込めない場合も例外を投げず、未ロードのモデルと
      ``meta["pretrained_loaded"]=False`` を返します（警告ログを出力）。

    Parameters
    ----------
    device : str | torch.device, optional
        未指定時は自動判定（CUDA 利用可なら ``"cuda"``、それ以外は ``"cpu"``）。
    strict : bool, default=True
        ``load_state_dict`` の strict フラグ。
    verify_hash : bool, default=True
        同名 ``.sha256`` が存在する場合のみ SHA256 を照合。

    Returns
    -------
    model : WaveMAE
        既定構成で構築され、（可能なら）同梱重みをロード済みのモデル。
    meta : dict
        付随情報（公開フィールドのみ）:
          - ``"name"`` : 重みベース名（例: ``"wavemae_base_256"``）
          - ``"shape"`` : 主要形状要約（``seq_len``, ``latent_dim``, ``d_model``,
            ``num_layers``, ``nhead``, ``decoder_hidden``, ``n_blocks``, ``n_mask``）
          - ``"repos"`` : GitHub URL（``{"library": ..., "pretraining": ...}``)
          - ``"device"`` : 読み込み時の map 先（文字列）
          - ``"strict"`` : strict フラグ
          - ``"pretrained_loaded"`` : bool（ロード成否）
          - ``"warning"`` : 失敗時のみ警告メッセージ

    Notes
    -----
    既定構成:
    ``seq_len=256, d_model=256, nhead=4, num_layers=4, dim_feedforward=1024,
      dropout=0.1, use_learnable_pos=True, latent_dim=64, dec_hidden=256, dec_dropout=0.1``
    """
    model = _build_model_default(device=device)

    weight_file = "wavemae_base_256.pt"
    weight_path = _asset_path(weight_file)
    sha256_path = _asset_path(weight_file + ".sha256")

    loaded_ok, warn_msg = False, ""
    try:
        if not os.path.exists(weight_path):
            raise FileNotFoundError(weight_path)
        if verify_hash and os.path.exists(sha256_path):
            _verify_sha256(weight_path, sha256_path)
        map_location = model.device if hasattr(model, "device") else (
            torch.device(device) if device is not None else "cpu"
        )
        state = torch.load(weight_path, map_location=map_location)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=strict)
        loaded_ok = True
    except Exception as e:
        warn_msg = f"事前学習済み重みをloadできませんでした: {e}"
        logger.warning(warn_msg)

    meta: Dict[str, Any] = {
        "name": os.path.splitext(os.path.basename(weight_path))[0],
        "shape": _introspect_shape(model),
        "repos": dict(_REPOS),
        "device": str(getattr(model, "device", device or ("cuda" if torch.cuda.is_available() else "cpu"))),
        "strict": bool(strict),
        "pretrained_loaded": bool(loaded_ok),
    }
    if not loaded_ok:
        meta["warning"] = warn_msg
    return model, meta
