from pathlib import Path
import hashlib
import pytest
import torch

from wavemae.utils.load import (
    WaveMAEConfig,
    get_default_config,
    build_model,
    verify_sha256,
    load_weight_with_sha256,
)


def _write_bytes(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return str(path)


def _sha256_hex(data: bytes) -> str:
    h = hashlib.sha256(); h.update(data); return h.hexdigest()


def test_get_default_config_and_build_model_forward_smoke():
    cfg = get_default_config()
    assert isinstance(cfg, WaveMAEConfig)
    # 代表的フィールドが正の値
    assert cfg.seq_len > 0 and cfg.d_model > 0 and cfg.latent_dim > 0

    model = build_model(cfg, device="cpu")
    x = torch.randn(2, cfg.seq_len)
    # reconstruct() はマスク自動生成で出力長 = seq_len
    x_rec = model.reconstruct(x)
    assert x_rec.shape == (2, cfg.seq_len)


def test_verify_sha256_ok_and_mismatch(tmp_path):
    # ダミーのファイルと .sha256 を作成
    data = b"hello wavemae"
    fpath = tmp_path / "w.pt"
    spath = tmp_path / "w.pt.sha256"
    _write_bytes(fpath, data)
    spath.write_text(_sha256_hex(data) + "\n", encoding="utf-8")

    # 正常：一致
    verify_sha256(str(fpath), str(spath))  # 例外なしで通るはず

    # 異常：ミスマッチ
    spath.write_text("deadbeef" * 8 + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        verify_sha256(str(fpath), str(spath))


def test_load_weight_with_sha256_roundtrip(tmp_path):
    # 極小モデルを構築し、その state_dict を保存 → 同じモデルに読み戻す
    cfg = get_default_config()
    model_a = build_model(cfg, device="cpu")
    model_b = build_model(cfg, device="cpu")

    # state_dict 保存
    wpath = tmp_path / "state.pt"
    torch.save(model_a.state_dict(), wpath)

    # 正しい SHA256 ファイルを作成
    hexsum = _sha256_hex(wpath.read_bytes())
    spath = tmp_path / "state.pt.sha256"
    spath.write_text(hexsum + "\n", encoding="utf-8")

    # 検証付きロード
    load_weight_with_sha256(model_b, str(wpath), str(spath), strict=True)

    # パラメータが一致することを確認
    for (k1, v1), (k2, v2) in zip(model_a.state_dict().items(), model_b.state_dict().items()):
        assert k1 == k2
        assert torch.allclose(v1, v2)


@pytest.mark.skip(reason="パッケージ同梱アセット（weights/.sha256）が無い環境ではスキップ")
def test_assets_presence_and_hash():
    # パッケージに assets が同梱されている環境向けの追加検証（CI では通常 skip）
    from importlib.resources import files as _pkg_files
    base = _pkg_files("wavemae").joinpath("assets")
    w = base / "wavemae_base_256.pt"
    s = base / "wavemae_base_256.pt.sha256"
    assert w.exists() and s.exists()
    verify_sha256(str(w), str(s))
