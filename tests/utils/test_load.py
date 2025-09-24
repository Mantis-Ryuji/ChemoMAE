import hashlib
import os
from pathlib import Path

import pytest
import torch


def test___all___exports_only_loader():
    import wavemae.utils.load as ld
    assert getattr(ld, "__all__", []) == ["load_default_pretrained"]


def test_load_default_pretrained_missing_weights(monkeypatch, caplog, tmp_path):
    """
    既定探索で重みが見つからない場合：
      - pretrained_loaded=False
      - meta['warning'] があり、ファイルパス断片（/, \, :）を含まない
      - repos, config の基本項目は存在
    """
    import wavemae.utils.load as ld

    # assets ディレクトリを空の一時ディレクトリへ差し替え
    monkeypatch.setattr(ld, "_asset_dir", lambda: Path(tmp_path))

    caplog.set_level("WARNING", logger="wavemae.utils.load")
    model, meta = ld.load_default_pretrained(device="cpu")

    # 公開フィールドのみ（パスは非公開）
    assert "weight_path" not in meta
    assert "sha256_path" not in meta

    # 失敗フラグとパス非含有の短い警告文
    assert meta["pretrained_loaded"] is False
    assert "warning" in meta
    msg = meta["warning"]
    assert isinstance(msg, str) and len(msg) > 0
    assert all(sep not in msg for sep in ("/", "\\", ":"))

    # WARNING ログも同様にパスを含まない
    warn_texts = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any(isinstance(w, str) and len(w) > 0 for w in warn_texts)
    assert all(all(sep not in w for sep in ("/", "\\", ":")) for w in warn_texts)

    # 固定URL
    assert meta["repos"]["library"] == "https://github.com/Mantis-Ryuji/WaveMAE"
    assert meta["repos"]["pretraining"] == "https://github.com/Mantis-Ryuji/UnsupervisedWoodSegmentation-NIRHSI"

    # config の既定値（load.py の _DEFAULT_CFG に追従）
    cfg = meta["config"]
    assert cfg["seq_len"] == 256
    assert cfg["d_model"] == 384
    assert cfg["nhead"] == 6
    assert cfg["num_layers"] == 6
    assert cfg["dim_feedforward"] == 1536
    assert cfg["dropout"] == 0.1
    assert cfg["use_learnable_pos"] is True
    assert cfg["latent_dim"] == 64
    assert cfg["dec_hidden"] == 256
    assert cfg["dec_dropout"] == 0.1
    assert cfg["n_blocks"] == 32
    assert cfg["n_mask"] == 24

def test_load_default_pretrained_success_with_hash(tmp_path):
    """
    明示 weight_path 指定でロード成功：
      - pretrained_loaded=True / warning なし
      - sha256 ファイルがあれば検証される
      - config が既定値に一致
    """
    import wavemae.utils.load as ld

    # 既定構成で state_dict を保存
    base_model = ld._build_model_default(device="cpu")
    weight_file = tmp_path / "wavemae_base_256.pt"
    torch.save(base_model.state_dict(), weight_file)

    # 対応する .sha256 を作成（<file>.pt.sha256）
    h = hashlib.sha256()
    with open(weight_file, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    sha_file = tmp_path / (weight_file.name + ".sha256")
    sha_file.write_text(h.hexdigest() + "\n", encoding="utf-8")

    # 明示パスでロード（内部の assets 探索は使わない）
    model, meta = ld.load_default_pretrained(weight_path=str(weight_file), device="cpu")

    # 公開のみ（パスは存在しない）
    assert "weight_path" not in meta
    assert "sha256_path" not in meta

    # 成功フラグ
    assert meta["pretrained_loaded"] is True
    assert "warning" not in meta

    # URL 固定
    assert meta["repos"]["library"] == "https://github.com/Mantis-Ryuji/WaveMAE"
    assert meta["repos"]["pretraining"] == "https://github.com/Mantis-Ryuji/UnsupervisedWoodSegmentation-NIRHSI"

    # config（_DEFAULT_CFG と一致）
    cfg = meta["config"]
    assert cfg == {
        "seq_len": 256,
        "d_model": 384,
        "nhead": 6,
        "num_layers": 6,
        "dim_feedforward": 1536,
        "dropout": 0.1,
        "use_learnable_pos": True,
        "latent_dim": 64,
        "dec_hidden": 256,
        "dec_dropout": 0.1,
        "n_blocks": 32,
        "n_mask": 24,
    }
