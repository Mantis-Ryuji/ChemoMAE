from __future__ import annotations
import os
import hashlib
from pathlib import Path
from typing import Dict, Optional
import urllib.request

# 環境変数でキャッシュ先を上書き可
_DEFAULT_CACHE = Path(os.environ.get("WAVEMAE_CACHE", Path.home() / ".cache" / "wavemae"))

# ここに "タグ名: URL/メタ" を登録していく想定（いまは空）
_REGISTRY: Dict[str, Dict[str, str]] = {
    # 例:
    # "wavemae-base-v0": {"url": "https://github.com/xxx/releases/download/v0/wavemae-base-v0.pt",
    #                     "sha256": "<optional>"},
}


def get_cache_dir() -> Path:
    _DEFAULT_CACHE.mkdir(parents=True, exist_ok=True)
    return _DEFAULT_CACHE


def list_available() -> Dict[str, Dict[str, str]]:
    """
    現在登録されている事前重みのレジストリを返す。
    いまは空だが、将来 from_pretrained() で参照。
    """
    return dict(_REGISTRY)


def register_weight(tag: str, url: str, *, sha256: Optional[str] = None) -> None:
    """
    外部コードから動的に事前重みを登録できるフック。
    研究中の私家版重みを配布前に使う用途などに。
    """
    _REGISTRY[str(tag)] = {"url": str(url), **({"sha256": sha256} if sha256 else {})}


def _sha256sum(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_weight(tag: str, *, filename: Optional[str] = None, force: bool = False) -> Path:
    """
    レジストリにある重みをキャッシュへ取得してパスを返す。
    まだ公式Weightsはないので、呼び出すと KeyError を投げるのが現在の仕様。
    将来、WaveMAE.from_pretrained(...) からこれを呼ぶ想定。
    """
    if tag not in _REGISTRY:
        raise KeyError(
            f"Unknown weight tag: {tag}. No pretrained weights are published yet. "
            f"Use `register_weight(tag, url)` to add your own entry during experiments."
        )
    url = _REGISTRY[tag]["url"]
    dst = get_cache_dir() / (filename or Path(url).name)

    if force or (not dst.exists()):
        tmp = dst.with_suffix(dst.suffix + ".part")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, tmp.as_posix())
        if "sha256" in _REGISTRY[tag]:
            got = _sha256sum(tmp)
            want = _REGISTRY[tag]["sha256"]
            if want and got.lower() != want.lower():
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"Checksum mismatch for {tag}: expected {want}, got {got}")
        tmp.replace(dst)

    return dst
