import os
import hashlib
import torch
import pytest

def test___all___exports_only_loader():
    import wavemae.load as ld
    assert "load_default_pretrained" in getattr(ld, "__all__", [])
    # 余計な公開をしていないことのサニティチェック
    assert all(name == "load_default_pretrained" for name in ld.__all__)

def test_load_default_pretrained_missing_weights(monkeypatch, caplog, tmp_path):
    """
    重みが存在しない場合でも (model, meta) が返り、
    - meta["pretrained_loaded"] is False
    - meta["warning"] を含む
    - logger.warning が出力される
    - repos 固定URLが入っている
    を確認する。
    """
    import wavemae.utils.load as ld

    # どの相対パスを渡されても存在しない tmp_path を指すようにする
    monkeypatch.setattr(ld, "_asset_path", lambda rel: str(tmp_path / rel))

    caplog.set_level("WARNING", logger="wavemae.load")
    model, meta = ld.load_default_pretrained(device="cpu", verify_hash=True)

    # 返り値の型/基本フィールド
    from wavemae.models.wave_mae import WaveMAE
    assert isinstance(model, WaveMAE)
    assert isinstance(meta, dict)
    assert meta["name"]  # 例: "wavemae_base_256"
    assert meta["shape"]["seq_len"] == 256

    # 期待する失敗フラグとログ
    assert meta["pretrained_loaded"] is False
    assert "warning" in meta
    assert "事前学習済み重みをloadできませんでした" in meta["warning"]
    assert any("事前学習済み重みをloadできませんでした" in rec.message for rec in caplog.records)

    # 固定URL
    assert meta["repos"]["library"] == "https://github.com/Mantis-Ryuji/WaveMAE"
    assert meta["repos"]["pretraining"] == "https://github.com/Mantis-Ryuji/UnsupervisedWoodSegmentation-NIRHSI"

def test_load_default_pretrained_success_with_hash(monkeypatch, tmp_path):
    import wavemae.load as ld

    # 既定構成で state_dict を保存
    base_model = ld._build_model_default(device="cpu")
    weight_file = tmp_path / "wavemae_base_256.pt"
    torch.save(base_model.state_dict(), weight_file)

    # 対応する .sha256 を作成
    h = hashlib.sha256()
    with open(weight_file, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    sha_file = tmp_path / (weight_file.name + ".sha256")
    sha_file.write_text(h.hexdigest() + "\n", encoding="utf-8")

    # _asset_path を一時ファイルへ差し替え
    def fake_asset_path(rel: str) -> str:
        if rel.endswith(".pt.sha256"):
            return str(sha_file)
        if rel.endswith(".pt"):
            return str(weight_file)
        return str(tmp_path / rel)

    monkeypatch.setattr(ld, "_asset_path", fake_asset_path)

    model, meta = ld.load_default_pretrained(device="cpu", verify_hash=True)

    from wavemae.models.wave_mae import WaveMAE
    assert isinstance(model, WaveMAE)
    assert meta["pretrained_loaded"] is True
    assert "warning" not in meta
    assert os.path.samefile(meta["weight_path"], str(weight_file))
    assert os.path.samefile(meta["sha256_path"], str(sha_file))
    assert meta["repos"]["library"] == "https://github.com/Mantis-Ryuji/WaveMAE"
    assert meta["repos"]["pretraining"] == "https://github.com/Mantis-Ryuji/UnsupervisedWoodSegmentation-NIRHSI"

    # 形状メタ：全フィールドを検証
    shape = meta["shape"]
    expected_shape = {
        "seq_len": 256,
        "latent_dim": 64,
        "d_model": 256,
        "num_layers": 4,
        "nhead": 4,
        "decoder_hidden": 256,
        "n_blocks": 16,  # WaveMAE のデフォルト
        "n_mask": 8,     # WaveMAE のデフォルト
    }
    # キーの完全一致
    assert set(shape.keys()) == set(expected_shape.keys())
    # 値の完全一致
    for k, v in expected_shape.items():
        assert shape[k] == v, f"shape[{k}] expected {v}, got {shape[k]}"
