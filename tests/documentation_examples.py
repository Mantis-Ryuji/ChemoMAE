"""Run selected, actual Markdown Python examples using runtime dependencies only.

The examples are read from this checkout, but ``chemomae`` is imported normally.
Use ``python -I`` from outside the checkout to validate an installed wheel. No
example code is copied or rewritten, and no installation commands are executed.
Each recipe has its own namespace and a temporary directory for all outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import re
import tempfile
import traceback
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TypeVar


REPOSITORY = Path(__file__).resolve().parents[1]
Namespace = dict[str, object]
Value = TypeVar("Value")


@dataclass(frozen=True)
class PythonBlock:
    path: Path
    index: int
    section: str
    first_line: int
    source: str

    @property
    def context(self) -> str:
        return f"{self.path.relative_to(REPOSITORY)}:{self.first_line} (Python block {self.index})"


@dataclass(frozen=True)
class Recipe:
    name: str
    relative_path: str
    sections: tuple[tuple[str, int], ...]
    check: Callable[[Mapping[str, object]], None] | None = None


def require(namespace: Mapping[str, object], name: str, kind: type[Value]) -> Value:
    value = namespace.get(name)
    if not isinstance(value, kind):
        raise AssertionError(f"Expected {name!r} to be {kind.__name__}; got {type(value).__name__}.")
    return value


def check_readme(namespace: Mapping[str, object]) -> None:
    import torch
    from chemomae.training import Trainer

    trainer = require(namespace, "trainer", Trainer)
    assert trainer.optimizer_updates == 8
    assert trainer.amp_skips == 0
    train = require(namespace, "train_features", torch.Tensor)
    test = require(namespace, "test_features", torch.Tensor)
    labels = require(namespace, "test_labels", torch.Tensor)
    assert train.shape == (64, 8) and test.shape == (16, 8)
    assert labels.shape == (16,) and labels.device.type == "cpu"
    assert torch.isfinite(train).all() and torch.isfinite(test).all()
    run_dir = require(namespace, "run_dir", Path)
    result = require(namespace, "result", dict)
    artifact = require(result, "final_artifact", str)
    assert Path(artifact).is_absolute()
    assert Path(artifact) == (run_dir / "last_model.artifact.pt").resolve()
    assert (run_dir / "last_model.artifact.pt").is_file()
    assert (run_dir / "clusters.pt").is_file()


def check_workflow(namespace: Mapping[str, object]) -> None:
    import numpy as np
    import torch
    from chemomae.clustering import LLAResult

    result = require(namespace, "result", dict)
    assert result["completed"] and result["epochs"] == 2
    assert result["optimizer_updates"] == 8 and result["amp_skips"] == 0
    assert result["final_model"] == "last_model.pt"
    artifact = require(result, "final_artifact", str)
    assert Path(artifact).is_absolute() and Path(artifact).is_file()
    assert require(namespace, "train_array", np.ndarray).dtype == np.float64
    assert require(namespace, "train_x", torch.Tensor).dtype == torch.float32
    resumed_result = require(namespace, "resumed_result", dict)
    assert resumed_result["completed"] and resumed_result["epochs"] == 2
    assert resumed_result["optimizer_updates"] == 8
    assert require(namespace, "train_features", torch.Tensor).shape == (64, 8)
    assert require(namespace, "test_features", torch.Tensor).shape == (16, 8)
    fps = require(namespace, "fps_indices", torch.Tensor)
    assert fps.shape == (32,) and torch.unique(fps).numel() == 32
    assert require(namespace, "label_map", torch.Tensor).shape == (12, 16)
    assert require(namespace, "lla", LLAResult).valid_pixels == 172
    for name in ("validation_mse", "test_mse"):
        value = require(namespace, name, float)
        assert math.isfinite(value) and value >= 0
    assert require(namespace, "selected_artifact", Path).is_file()
    run_dir = require(namespace, "run_dir", Path)
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    assert report["chemomae_version"] == "0.2.4" and report["device"] == "cpu"
    assert report["synthetic_counts"] == {"train": 64, "validation": 16, "test": 16}
    saved_map = torch.load(run_dir / "spatial_map.pt", map_location="cpu", weights_only=True)
    assert torch.equal(saved_map["labels"], namespace["label_map"])
    assert torch.equal(saved_map["valid_mask"], namespace["valid_mask"])


def check_trainer(namespace: Mapping[str, object]) -> None:
    import numpy as np
    import torch

    result = require(namespace, "result", dict)
    assert result["completed"] and result["epochs"] == 2
    assert result["optimizer_updates"] == 6 and result["amp_skips"] == 0
    assert result["final_model"] == "last_model.pt"
    selected = require(namespace, "selected", Path)
    assert selected.is_absolute() and selected.is_file()
    assert result["final_artifact"] == str(selected)
    model = require(namespace, "model", torch.nn.Module)
    assert not model.training
    assert require(namespace, "numpy_spectra", np.ndarray).dtype == np.float64
    assert require(namespace, "spectra", torch.Tensor).dtype == next(model.parameters()).dtype


def check_custom_trainer(namespace: Mapping[str, object]) -> None:
    import torch
    from chemomae.training import Trainer

    result = require(namespace, "result", dict)
    assert result["completed"] and result["epochs"] == 2
    assert result["attempted_steps"] == result["optimizer_updates"] == 6
    assert result["amp_skips"] == 0
    resumed = require(namespace, "resumed", Trainer)
    assert len(resumed.history) == 2
    assert all(record["application_note"] == "explicit masks and caller-owned ordering"
               for record in resumed.history)
    checkpoint = torch.load(
        require(namespace, "run_dir", Path) / "checkpoints" / "last.pt",
        map_location="cpu", weights_only=False,
    )
    caller_state = resumed.checkpoint_extra_state()
    assert torch.equal(checkpoint["extension_state"]["order_generator"],
                       require(caller_state, "order_generator", torch.Tensor))


def check_plain_loop(namespace: Mapping[str, object]) -> None:
    import torch

    loss = require(namespace, "loss", torch.Tensor)
    assert loss.ndim == 0 and torch.isfinite(loss)
    optimizer = require(namespace, "optimizer", torch.optim.Optimizer)
    assert optimizer.state
    assert all(int(state["step"]) == 2 for state in optimizer.state.values())
    model = require(namespace, "model", torch.nn.Module)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def check_optimizer_builders(namespace: Mapping[str, object]) -> None:
    import torch
    from torch.utils.data import DataLoader

    optimizer = require(namespace, "optimizer", torch.optim.Optimizer)
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    model = require(namespace, "model", torch.nn.Module)
    expected = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert {id(parameter) for parameter in parameters} == {id(parameter) for parameter in expected}
    assert len(parameters) == len(expected)
    assert all(parameter.device.type == "cpu" for parameter in parameters)
    assert len(require(namespace, "train_loader", DataLoader)) == 3
    assert require(namespace, "scheduler", torch.optim.lr_scheduler.LRScheduler).last_epoch == 0


def check_optimizer_trace(namespace: Mapping[str, object]) -> None:
    import torch

    assert namespace["update"] == 6
    scheduler = require(namespace, "scheduler", torch.optim.lr_scheduler.LRScheduler)
    assert scheduler.last_epoch == 6
    assert math.isclose(scheduler.get_last_lr()[0], 1e-4, rel_tol=1e-12)
    assert require(namespace, "consumed_lr", float) > scheduler.get_last_lr()[0]
    assert torch.isfinite(require(namespace, "parameter", torch.Tensor))


RECIPES = (
    Recipe("readme", "README.md", (("ChemoMAE Example", 1),), check_readme),
    Recipe("workflow", "docs/tutorials/workflow.md", (
        ("1. Set up and prepare spectra", 1),
        ("2. Train a reconstruction model", 1),
        ("3. Save and reload the selected model", 1),
        ("4. Extract features", 2),
        ("Optional: select training rows with FPS", 1),
        ("Optional: add spectral augmentation", 1),
        ("Optional: resume a completed epoch", 1),
        ("Optional: evaluate reconstruction", 1),
        ("Optional: cluster directional features", 1),
        ("Optional: evaluate a spatial label map", 2),
        ("Optional: save a workflow report", 1),
    ), check_workflow),
    # The NumPy subsection belongs to this level-two section in python_blocks.
    Recipe("trainer-minimal", "docs/training/trainer.md",
           (("Configuration and the simple path", 2),),
           check_trainer),
    Recipe("trainer-custom", "docs/training/trainer.md",
           (("Custom ordering, masks, and caller state", 2),), check_custom_trainer),
    Recipe("trainer-plain", "docs/training/trainer.md",
           (("A plain PyTorch loop", 1),), check_plain_loop),
    Recipe("optimizer-builders", "docs/training/optim.md",
           (("AdamW parameter groups", 1), ("Epoch-sized scheduler budgets", 2)),
           check_optimizer_builders),
    Recipe("optimizer-trace", "docs/training/optim.md",
           (("Exact multiplier and update indexing", 1),), check_optimizer_trace),
    # These standalone examples carry their own assertions beside the API calls.
    Recipe("snv", "docs/preprocessing/snv.md", (("Quick start", 1),)),
    Recipe("fps", "docs/preprocessing/dowmsampling.md", (("Quick start", 1),)),
    Recipe("model", "docs/models/chemo_mae.md", (("Quick start", 1),)),
    Recipe("losses", "docs/models/losses.md", (("Quick start", 1),)),
    Recipe("persistence", "docs/models/persistence.md", (("Quick start", 1),)),
    Recipe("augmenter", "docs/training/augmenter.md", (("Quick start", 1),)),
    Recipe("tester", "docs/training/tester.md", (("Quick start", 1),)),
    Recipe("extractor", "docs/training/extractor.md", (("Quick start", 1),)),
    Recipe("seed", "docs/utils/seed.md", (("Quick start", 1),)),
    Recipe("cosine-kmeans", "docs/clustering/cosine_kmeans.md", (("Quick start", 1),)),
    Recipe("vmf", "docs/clustering/vmf_mixture.md", (("Quick start", 1),)),
    Recipe("cosine-ops", "docs/clustering/ops.md", (("Quick start", 1),)),
    Recipe("silhouette", "docs/clustering/metric.md", (("Quick start", 1),)),
    Recipe("spatial", "docs/clustering/spatial.md", (("Quick start", 1),)),
)


def python_blocks(path: Path) -> list[PythonBlock]:
    """Read top-level fenced Python source while preserving Markdown line numbers."""
    blocks: list[PythonBlock] = []
    section = ""
    fence = ""
    language = ""
    first_line = 0
    source: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if fence:
            if re.fullmatch(rf"\s*{re.escape(fence[0])}{{{len(fence)},}}\s*", line):
                if language == "python":
                    blocks.append(PythonBlock(path, len(blocks) + 1, section, first_line,
                                              "\n".join(source) + "\n"))
                fence, language, source = "", "", []
            else:
                source.append(line)
            continue
        match = re.fullmatch(r"\s*(`{3,}|~{3,})([^`~]*)", line)
        if match:
            fence, language = match.group(1), match.group(2).strip().lower()
            first_line = number + 1
        elif line.startswith("## "):
            section = line[3:].strip()
    if fence:
        raise ValueError(f"{path}: Unclosed fence beginning at line {first_line - 1}.")
    return blocks


def select_blocks(recipe: Recipe, blocks: list[PythonBlock]) -> list[PythonBlock]:
    selected: list[PythonBlock] = []
    for section, expected_count in recipe.sections:
        found = [block for block in blocks if block.section == section]
        if len(found) != expected_count:
            raise ValueError(
                f"{recipe.relative_path}: expected {expected_count} Python block(s) under "
                f"{section!r}, found {len(found)}. Update the documented recipe and selector together."
            )
        selected.extend(found)
    if selected != sorted(selected, key=lambda block: block.first_line):
        raise ValueError(f"{recipe.relative_path}: selected sections are no longer in documented order.")
    return selected


@contextmanager
def temporary_recipe_directory() -> Iterator[None]:
    original_directory = Path.cwd()
    original_tempdir = tempfile.tempdir
    with tempfile.TemporaryDirectory(prefix="chemomae-docs-") as directory:
        try:
            os.chdir(directory)
            # Actual examples call mkdtemp; place their directories inside this
            # scope as well so cleanup includes every output they create.
            tempfile.tempdir = directory
            yield
        finally:
            tempfile.tempdir = original_tempdir
            os.chdir(original_directory)


def run_recipe(recipe: Recipe, blocks: list[PythonBlock]) -> None:
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    namespace: Namespace = {"__name__": "__documentation__"}
    context = recipe.relative_path
    with temporary_recipe_directory():
        try:
            for block in blocks:
                context = block.context
                namespace["__file__"] = str(block.path)
                # Prefix only blank lines so compile/traceback line numbers
                # point at the exact source line in the Markdown document.
                code = compile("\n" * (block.first_line - 1) + block.source,
                               str(block.path), "exec")
                exec(code, namespace)
                print(f"PASS source: {context}")
            if recipe.check is not None:
                context = f"postconditions after {context}"
                recipe.check(namespace)
            print(f"PASS recipe: {recipe.name} ({len(blocks)} Python blocks; assertions passed)")
        except Exception:
            # Snippet exceptions are arbitrary. Preserve their original
            # traceback while adding the Markdown source location.
            print(f"FAIL recipe: {recipe.name}; {context}", flush=True)
            raise
        finally:
            plt.close("all")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", action="append", choices=[recipe.name for recipe in RECIPES],
                        help="Run only these recipes; repeat the option to select several.")
    parser.add_argument("--list", action="store_true", help="Show selected/skipped blocks without execution.")
    args = parser.parse_args()
    recipes = [recipe for recipe in RECIPES if not args.recipe or recipe.name in args.recipe]
    try:
        documents = {recipe.relative_path: python_blocks(REPOSITORY / recipe.relative_path)
                     for recipe in recipes}
        selections = [(recipe, select_blocks(recipe, documents[recipe.relative_path]))
                      for recipe in recipes]
        selected_locations = {(block.path, block.first_line)
                              for _, blocks in selections for block in blocks}
        for recipe, blocks in selections:
            print(f"SELECT {recipe.name}: {len(blocks)} Python blocks")
        for relative_path, blocks in documents.items():
            for block in blocks:
                if (block.path, block.first_line) in selected_locations:
                    continue
                reason = "outside the selected recipes; may be an API signature or need prior setup/CUDA"
                print(f"SKIP {block.context}: {reason}")
        print("SKIP installation/shell fences: use the chosen environment's installed ChemoMAE.")
        print("Scope: README, staged workflow, Trainer/optimizer recipes, and standalone "
              "CPU Quick start examples across the API guides. Other blocks are not executed.")
        if args.list:
            return 0
        os.environ["MPLBACKEND"] = "Agg"
        import chemomae
        import numpy as np
        import torch

        print(f"Python {platform.python_version()}; NumPy {np.__version__}")
        print(f"ChemoMAE {chemomae.__version__} imported from {chemomae.__file__}")
        print(f"Torch {torch.__version__}; CPU examples; default dtype float32; Agg; outputs are temporary.")
        torch.set_num_threads(1)
        for recipe, blocks in selections:
            run_recipe(recipe, blocks)
    except Exception:
        # CLI boundary: report the original error and a failing exit status.
        traceback.print_exc()
        return 1
    print(f"Documentation examples passed: {len(recipes)} recipes, "
          f"{sum(len(blocks) for _, blocks in selections)} actual Python blocks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
