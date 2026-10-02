"""Runtime-only CPU smoke check, intended for a freshly installed wheel.

Run from outside the source checkout. This file uses no pytest/development extras
and verifies the installed public workflow with small synthetic inputs.
"""

from __future__ import annotations

import importlib.metadata
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import chemomae
from chemomae.clustering import CosineKMeans, local_label_agreement
from chemomae.models import ChemoMAE
from chemomae.preprocessing import snv
from chemomae.training import Extractor, ExtractorConfig, Tester, TesterConfig, Trainer, TrainerConfig


def main() -> None:
    assert importlib.metadata.version("chemomae") == chemomae.__version__ == "0.2.3"
    package_dir = Path(chemomae.__file__).resolve().parent
    assert (package_dir / "py.typed").is_file()
    distribution = importlib.metadata.distribution("chemomae")
    files = {str(path).replace("\\", "/") for path in distribution.files or ()}
    for name in ("LICENSE", "NOTICE"):
        assert any(path.endswith("/" + name) for path in files), f"Missing installed {name}"

    torch.manual_seed(42)
    spectra = snv(torch.randn(12, 16))
    loader = DataLoader(TensorDataset(spectra), batch_size=4, shuffle=False)
    with tempfile.TemporaryDirectory(prefix="chemomae-installed-") as directory:
        out_dir = Path(directory)
        model = ChemoMAE(
            seq_len=16, n_patches=4, n_mask=2, d_model=8,
            nhead=2, num_layers=1, latent_dim=4,
        )
        trainer = Trainer(
            model, torch.optim.AdamW(model.parameters(), lr=1e-3), loader,
            cfg=TrainerConfig(
                out_dir=out_dir, device="cpu", use_ema=False, resume_from=None,
                progress=False, verbose=False,
            ),
        )
        result = trainer.fit(epochs=1)
        assert result["completed"] and result["optimizer_updates"] == 3
        restored = ChemoMAE.load(out_dir / "last_model.artifact.pt")
        assert restored.get_config() == model.get_config()
        extractor = Extractor(restored, ExtractorConfig(output_device="cpu"))
        features = extractor(loader)
        streamed = torch.cat(list(extractor.iter_transform(loader)))
        torch.testing.assert_close(features, streamed, rtol=0, atol=0)
        clusterer = CosineKMeans(n_components=2, max_iter=5, device="cpu", random_state=42)
        clusterer.fit(features)
        labels = clusterer.predict(features)
        clusterer.save_centroids(out_dir / "clusters.pt")
        reloaded = CosineKMeans(n_components=2, device="cpu").load_centroids(out_dir / "clusters.pt")
        assert torch.equal(labels, reloaded.predict(features))
        loss = Tester(restored, TesterConfig(
            device="cpu", loss_region="all", fixed_visible=torch.ones(16, dtype=torch.bool),
            log_history=False, progress=False,
        ))(loader)
        assert isinstance(loss, float) and loss >= 0
        lla = local_label_agreement(
            torch.tensor([[0, 0], [1, 1]]), torch.ones(2, 2, dtype=torch.bool),
            windows=(3,), device="cpu",
        )
        assert lla.valid_pixels == 4 and lla.windows[0].valid_pairs == 12
    print("Installed ChemoMAE CPU workflow passed.")


if __name__ == "__main__":
    main()
