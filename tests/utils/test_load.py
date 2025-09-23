# tests/utils/test_load_default_pretrained.py

import os
import hashlib
import torch
import pytest

def test___all___exports_only_loader():
    import wavemae.utils.load as ld
    assert getattr(ld, "__all__", []) == ["load_default_pretrained"]

def test_load_default_pretrained_missing_weights(monkeypatch, caplog, tmp_path):
    import wavemae.utils.load as ld
    monkeypatch.setattr(ld, "_asset_path", lambda rel: str(tmp_path / rel))
    caplog.set_level("WARNING", logger="wavemae.utils.load")

    model, meta = ld.load_default_pretrained(device="cpu", verify_hash=True)

    # 公開フィールドのみ
    assert "weight_path" not in meta
    assert "sha256_path" not in meta

    # 失敗フラグとログ
    assert meta["pretrained_loaded"] is False
    assert "warning" in meta and "事前学習済み重みをloadできませんでした" in meta["warning"]
    assert any("事前学習済み重みをloadできませんでした" in r.message for r in caplog.records)

    # 固定URL
    assert meta["repos"]["library"] == "https://github.com/Mantis-Ryuji/WaveMAE"
    assert meta["repos"]["pretraining"] == "https://github.com/Mantis-Ryuji/UnsupervisedWoodSegmentation-NIRHSI"

def test_load_default_pretrained_success_with_hash(monkeypatch, tmp_path):
    import wavemae.utils.load as ld

    base_model = ld._build_model_default(device="cpu")
    weight_file = tmp_path / "wavemae_base_256.pt"
    torch.save(base_model.state_dict(), weight_file)

    h = hashlib.sha256()
    with open(weight_file, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    sha_file = tmp_path / (weight_file.name + ".sha256")
    sha_file.write_text(h.hexdigest() + "\n", encoding="utf-8")

    def fake_asset_path(rel: str) -> str:
        if rel.endswith(".pt.sha256"):
            return str(sha_file)
        if rel.endswith(".pt"):
            return str(weight_file)
        return str(tmp_path / rel)

    monkeypatch.setattr(ld, "_asset_path", fake_asset_path)

    model, meta = ld.load_default_pretrained(device="cpu", verify_hash=True)

    # 公開のみ（パスは存在しない）
    assert "weight_path" not in meta
    assert "sha256_path" not in meta

    # 成功フラグ
    assert meta["pretrained_loaded"] is True
    assert "warning" not in meta

    # URL 固定
    assert meta["repos"]["library"] == "https://github.com/Mantis-Ryuji/WaveMAE"
    assert meta["repos"]["pretraining"] == "https://github.com/Mantis-Ryuji/UnsupervisedWoodSegmentation-NIRHSI"

    # 形状（全フィールド）完全一致
    shape = meta["shape"]
    expected = {
        "seq_len": 256,
        "latent_dim": 64,
        "d_model": 256,
        "num_layers": 4,
        "nhead": 4,
        "decoder_hidden": 256,
        "n_blocks": 16,
        "n_mask": 8,
    }
    assert set(shape) == set(expected)
    for k, v in expected.items():
        assert shape[k] == v, f"{k}: expected {v}, got {shape[k]}"
