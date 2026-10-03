"""Bounded synthetic LLA latency/memory checks; run explicitly, never by pytest.

No files are written unless ``--output`` is supplied. Timings cover the public
API, including validation and the returned CPU diagnostics, rather than isolated
convolution kernels. Synthetic input generation is excluded from those timings.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
import itertools
import json
import math
from pathlib import Path
import platform
import statistics
import time
from typing import TypeAlias

import numpy as np
import torch

import chemomae
from chemomae.clustering import LLAResult, local_label_agreement


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
LLACall: TypeAlias = Callable[[], LLAResult]


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return parsed


def _shape(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("must be HEIGHTxWIDTH, for example 64x64")
    return _positive_int(parts[0]), _positive_int(parts[1])


def _window(value: str) -> int:
    parsed = _positive_int(value)
    if parsed % 2 == 0 or parsed * parsed - 1 > 2**24:
        raise argparse.ArgumentTypeError("must be odd and within exact FP32 count capacity")
    return parsed


def _chunk(value: str) -> int | None:
    return None if value.lower() == "all" else _positive_int(value)


def _density(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite number in (0, 1]") from exc
    if not math.isfinite(parsed) or not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("must be a finite number in (0, 1]")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "both"), default="both")
    parser.add_argument("--cuda-device", type=_nonnegative_int, default=0)
    parser.add_argument("--shapes", nargs="+", type=_shape, default=[(64, 64), (128, 128)])
    parser.add_argument("--classes", nargs="+", type=_positive_int, default=[4, 16])
    parser.add_argument("--windows", nargs="+", type=_window, default=[3, 9])
    parser.add_argument("--class-chunks", nargs="+", type=_chunk, default=[4, None])
    parser.add_argument("--block-size", type=_positive_int, default=8)
    parser.add_argument("--mask-density", type=_density, default=0.9)
    parser.add_argument("--seed", type=_nonnegative_int, default=42)
    parser.add_argument("--warmup", type=_positive_int, default=1)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--threads", type=_positive_int, help="explicitly set Torch CPU threads")
    parser.add_argument("--output", type=Path, help="explicit JSON destination; parent must exist")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing --output")
    args = parser.parse_args()
    for name in ("shapes", "classes", "windows", "class_chunks"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            parser.error(f"--{name.replace('_', '-')} must not contain duplicates")
    if any(classes > height * width for (height, width), classes in
           itertools.product(args.shapes, args.classes)):
        parser.error("every shape must have at least as many pixels as the largest class count")
    if args.seed + len(args.shapes) * len(args.classes) > 2**63 - 1:
        parser.error("seed plus input-case count must fit signed int64")
    if args.overwrite and args.output is None:
        parser.error("--overwrite requires --output")
    if args.output is not None:
        args.output = args.output.expanduser().resolve()
        if not args.output.parent.is_dir():
            parser.error("--output parent directory must already exist")
        if args.output.exists() and (not args.overwrite or not args.output.is_file()):
            parser.error("--output exists; use --overwrite only for a file you intend to replace")
    if args.device != "cpu":
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable; use --device cpu for CPU-only measurement")
        if args.cuda_device >= torch.cuda.device_count():
            parser.error("--cuda-device is outside the available CUDA device range")
    return args


def _json_value(value: object) -> JSONValue:
    """Represent undefined scores as JSON null, preserving reason diagnostics."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if not math.isfinite(value):
            raise ValueError("unexpected infinite value in LLA benchmark report")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    raise TypeError(f"unsupported report value: {type(value).__name__}")


def _same_result(expected: LLAResult, actual: LLAResult, context: str) -> None:
    # Compare every count, score, occupancy and reason exactly. JSON conversion
    # only makes two undefined NaN values comparable; there is no float tolerance.
    if _json_value(asdict(expected)) != _json_value(asdict(actual)):
        raise AssertionError(f"Exact CPU/chunk/CUDA LLA diagnostics differ: {context}")


def _synthetic_map(
    shape: tuple[int, int], classes: int, block_size: int, density: float, seed: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    height, width = shape
    rows = torch.arange(height)[:, None]
    cols = torch.arange(width)[None, :]
    tile_cols = (width + block_size - 1) // block_size
    tile_count = ((height + block_size - 1) // block_size) * tile_cols
    if tile_count >= classes:
        indices = ((rows // block_size) * tile_cols + cols // block_size) % classes
        pattern = "repeated_spatial_blocks"
    else:
        indices = (rows * width + cols) % classes
        pattern = "pixel_cycle_for_class_coverage"
    labels = (indices * 7 - 3).to(dtype=torch.int64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    valid = torch.rand(shape, generator=generator) < density
    # Guarantee each requested class occurs in the evaluated region, even for
    # small maps/sparse masks. Record this intervention rather than imply an
    # exactly Bernoulli mask or an exactly requested final valid fraction.
    flat_valid = valid.view(-1)
    flat_indices = indices.reshape(-1)
    for label in range(classes):
        position = torch.nonzero(flat_indices == label, as_tuple=False)[0, 0]
        flat_valid[position] = True
    return labels, valid, pattern


def _sample_summary(values: list[float]) -> dict[str, JSONValue]:
    return {
        "samples_ms": values,
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _time_public_call(
    call: LLACall, warmup: int, repeats: int, device: torch.device,
) -> dict[str, JSONValue]:
    for _ in range(warmup):
        call()
    wall_ms: list[float] = []
    event_ms: list[float] = []
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(device)
        for _ in range(repeats):
            torch.cuda.synchronize(device)
            start_event.record(stream)
            start_ns = time.perf_counter_ns()
            call()
            end_event.record(stream)
            end_event.synchronize()
            wall_ms.append((time.perf_counter_ns() - start_ns) / 1e6)
            event_ms.append(start_event.elapsed_time(end_event))
    else:
        for _ in range(repeats):
            start_ns = time.perf_counter_ns()
            call()
            wall_ms.append((time.perf_counter_ns() - start_ns) / 1e6)
    result: dict[str, JSONValue] = {"wall": _sample_summary(wall_ms)}
    if event_ms:
        result["cuda_event_api_span"] = _sample_summary(event_ms)
    return result


def _cuda_memory(call: LLACall, device: torch.device) -> dict[str, JSONValue]:
    """Measure one separate warm public call with current allocator caches."""
    torch.cuda.synchronize(device)
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    call()
    torch.cuda.synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    return {
        "baseline_allocated_bytes": allocated_before,
        "peak_allocated_bytes": peak_allocated,
        "incremental_peak_allocated_bytes": peak_allocated - allocated_before,
        "baseline_reserved_bytes": reserved_before,
        "peak_reserved_bytes": peak_reserved,
        "incremental_peak_reserved_bytes": peak_reserved - reserved_before,
    }


def _environment(device: torch.device | None) -> dict[str, JSONValue]:
    result: dict[str, JSONValue] = {
        "python": platform.python_version(), "platform": platform.platform(),
        "chemomae": chemomae.__version__, "numpy": np.__version__,
        "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "torch_cpu_threads": torch.get_num_threads(),
        "torch_cpu_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if device is not None:
        properties = torch.cuda.get_device_properties(device)
        result["cuda_device"] = str(device)
        result["cuda_name"] = properties.name
        result["cuda_total_memory_bytes"] = properties.total_memory
        result["cuda_capability"] = list(torch.cuda.get_device_capability(device))
    return result


def main() -> None:
    args = _parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    cpu = torch.device("cpu")
    cuda = None if args.device == "cpu" else torch.device("cuda", args.cuda_device)
    if cuda is not None:
        torch.cuda.set_device(cuda)
    cases: list[JSONValue] = []
    for input_index, (shape, classes) in enumerate(itertools.product(args.shapes, args.classes)):
        seed = args.seed + input_index
        prep_start_ns = time.perf_counter_ns()
        labels, valid, pattern = _synthetic_map(
            shape, classes, args.block_size, args.mask_density, seed,
        )
        cpu_prep_ms = (time.perf_counter_ns() - prep_start_ns) / 1e6
        references: dict[int, LLAResult] = {}
        # Validate all selected chunks once, outside the measured iterations.
        if cuda is not None:
            labels_cuda, valid_cuda = labels.to(cuda), valid.to(cuda)
        for width in args.windows:
            reference = local_label_agreement(labels, valid, windows=(width,), class_chunk=None)
            references[width] = reference
            for chunk in args.class_chunks:
                _same_result(reference, local_label_agreement(
                    labels, valid, windows=(width,), class_chunk=chunk,
                ), f"CPU shape={shape}, classes={classes}, window={width}, chunk={chunk}")
                if cuda is not None:
                    _same_result(reference, local_label_agreement(
                        labels_cuda, valid_cuda, windows=(width,), class_chunk=chunk,
                    ), f"CUDA shape={shape}, classes={classes}, window={width}, chunk={chunk}")
                    _same_result(reference, local_label_agreement(
                        labels, valid, windows=(width,), device=cuda, class_chunk=chunk,
                    ), f"CPU-to-CUDA shape={shape}, classes={classes}, window={width}, chunk={chunk}")
        if cuda is not None:
            del labels_cuda, valid_cuda
        for width, chunk in itertools.product(args.windows, args.class_chunks):
            row: dict[str, JSONValue] = {
                "shape": list(shape), "requested_classes": classes, "window": width,
                "class_chunk": chunk, "seed": seed, "pattern": pattern,
                "labels_dtype": str(labels.dtype), "mask_dtype": str(valid.dtype),
                "actual_valid_fraction": references[width].valid_pixels / labels.numel(),
                "cpu_input_generation_wall_ms": cpu_prep_ms,
                "exact_cpu_chunk_comparison": "passed",
                "diagnostics": _json_value(asdict(references[width])),
            }
            if args.device != "cuda":
                cpu_call = partial(
                    local_label_agreement, labels, valid, windows=(width,), class_chunk=chunk,
                )
                row["cpu_resident_public_call"] = _time_public_call(
                    cpu_call, args.warmup, args.repeats, cpu,
                )
            if cuda is not None:
                torch.cuda.synchronize(cuda)
                transfer_start_ns = time.perf_counter_ns()
                labels_cuda, valid_cuda = labels.to(cuda), valid.to(cuda)
                torch.cuda.synchronize(cuda)
                row["resident_cuda_input_transfer_wall_ms"] = (
                    time.perf_counter_ns() - transfer_start_ns
                ) / 1e6
                resident_call = partial(
                    local_label_agreement, labels_cuda, valid_cuda,
                    windows=(width,), class_chunk=chunk,
                )
                resident_timing = _time_public_call(resident_call, args.warmup, args.repeats, cuda)
                resident_timing["memory"] = _cuda_memory(resident_call, cuda)
                row["cuda_resident_public_call"] = resident_timing
                del resident_call, labels_cuda, valid_cuda
                # Resident inputs are released before measuring this mode so
                # incremental allocation includes CPU-to-GPU input transfers.
                transfer_call = partial(
                    local_label_agreement, labels, valid, windows=(width,),
                    device=cuda, class_chunk=chunk,
                )
                transfer_timing = _time_public_call(transfer_call, args.warmup, args.repeats, cuda)
                transfer_timing["memory"] = _cuda_memory(transfer_call, cuda)
                row["cpu_input_to_cuda_public_call"] = transfer_timing
                row["exact_cpu_cuda_comparison"] = "passed"
            cases.append(row)
    report = {
        "schema_version": 1, "benchmark": "synthetic_lla_public_api",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(cuda),
        "configuration": _json_value({
            "device": args.device, "shapes": args.shapes, "classes": args.classes,
            "windows": args.windows, "class_chunks": args.class_chunks,
            "block_size": args.block_size, "requested_mask_density": args.mask_density,
            "seed": args.seed, "warmup": args.warmup, "repeats": args.repeats,
        }),
        "measurement_notes": [
            "All call timings include public input validation and returned compact CPU diagnostics.",
            "CUDA-resident timings exclude input transfer; CPU-input-to-CUDA timings include it.",
            "Wall time uses perf_counter_ns with CUDA synchronization around timed work.",
            "CUDA event API spans include host scheduling gaps and are not isolated kernel time.",
            "CUDA wall timings include end-event recording/synchronization instrumentation.",
            "Input generation and standalone resident transfers are separate single observations.",
            "Memory uses one separate warmed call; peak stats are reset but allocator caches are kept.",
            "Resident memory baseline includes input tensors; CPU-input-to-CUDA baseline does not.",
            "Reserved memory includes prior cached allocations and is not a workspace-only measure.",
            "Allocator stats omit CUDA context, external allocations and other processes' memory.",
            "Backend convolution workspace can vary with shape, backend, hardware and warmup.",
            "Masks force one valid pixel per class; actual valid fractions are reported.",
            "Undefined floating diagnostics are JSON null with their original reason strings.",
            "These synthetic sizes and repetitions do not establish performance on research images.",
        ],
        "cases": cases,
    }
    payload = json.dumps(report, indent=2, allow_nan=False)
    if args.output is None:
        print(payload)
    else:
        # Exclusive creation protects against a file appearing during the run.
        with args.output.open("w" if args.overwrite else "x", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        print(f"LLA benchmark report written to {args.output}")


if __name__ == "__main__":
    main()
