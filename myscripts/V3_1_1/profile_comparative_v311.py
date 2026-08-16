"""Profile Baseline, CA, and CA+Refine with balanced isolated FP32 runs.

Each method is measured in a fresh worker process so absolute CUDA peak memory
cannot inherit allocations from another model.  Three full validation repeats
use a 3x3 Latin-square order, placing every method first, second, and third
once.  All workers share the preprocessing, synchronization, rotated-NMS,
dataloader, and input-shape policy used by :mod:`profile_refine_v311`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

from myscripts.V3.train_refine_v3 import write_csv, write_json
from myscripts.V3_1_1.evidence_runtime import (
    CANONICAL_BASELINE_WEIGHTS,
    CANONICAL_CA_WEIGHTS,
    ensure_omp_threads,
    load_obb_detector,
    require_canonical_path,
)
from myscripts.V3_1_1.profile_refine_v311 import _format_nms, _sync, _timed, summarize_timings


METHODS = ("Baseline", "CA", "CA+Refine")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-weights", default=CANONICAL_BASELINE_WEIGHTS)
    parser.add_argument("--ca-weights", default=CANONICAL_CA_WEIGHTS)
    parser.add_argument("--refine-profile-summary", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val", choices=("val",))
    parser.add_argument("--imgsz", type=int, default=640, choices=(640,))
    parser.add_argument("--batch", type=int, default=1, choices=(1,))
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3, choices=(3,))
    parser.add_argument(
        "--stability-relative-tolerance",
        type=float,
        default=0.05,
        help="Maximum relative spread across the three balanced latency repeats",
    )
    parser.add_argument(
        "--memory-relative-tolerance",
        type=float,
        default=0.02,
        help="Maximum relative spread across isolated peak-memory repeats",
    )
    parser.add_argument("--max-images", type=int, default=0, help="0 profiles the complete validation split")
    parser.add_argument("--output-dir", required=True)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    require_canonical_path(parser, args.baseline_weights, CANONICAL_BASELINE_WEIGHTS, "Baseline checkpoint")
    require_canonical_path(parser, args.ca_weights, CANONICAL_CA_WEIGHTS, "CA checkpoint")
    if args.amp:
        parser.error("official comparative profiling is locked to FP32; use --no-amp")
    if args.workers < 0 or args.warmup < 1 or args.max_images < 0:
        parser.error("workers/max-images must be non-negative and warmup must be positive")
    if not 0.0 < args.stability_relative_tolerance <= 0.25:
        parser.error("--stability-relative-tolerance must be in (0, 0.25]")
    if not 0.0 < args.memory_relative_tolerance <= 0.25:
        parser.error("--memory-relative-tolerance must be in (0, 0.25]")
    if not Path(args.refine_profile_summary).is_file():
        parser.error(f"Refine profile summary not found: {args.refine_profile_summary}")


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _forward_detector(torch, extractor, images):
    extractor.cache.clear()
    with torch.no_grad(), torch.autocast(
        device_type=extractor.device.type,
        dtype=torch.float16,
        enabled=extractor.amp,
    ):
        outputs = extractor.core_model(images)
    if "p2" not in extractor.cache or "p3" not in extractor.cache:
        raise RuntimeError("common P2/P3 evidence hooks did not fire")
    inference = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    if not isinstance(inference, torch.Tensor) or inference.ndim != 3:
        raise TypeError(f"expected OBB inference tensor [B,C,N], received {type(inference)}")
    return inference


def _run_detector_batch(torch, extractor, batch: dict[str, Any], *, timed: bool):
    device = extractor.device

    def measure(operation):
        if timed:
            return _timed(torch, device, operation)
        return operation(), 0.0

    images, preprocess_ms = measure(
        lambda: batch["img"].to(device, non_blocking=True).float().div_(255.0)
    )
    inference, detector_forward_ms = measure(lambda: _forward_detector(torch, extractor, images))
    detections, decode_nms_ms = measure(lambda: _format_nms(torch, inference, extractor))
    return detections, {
        "preprocess_ms": preprocess_ms,
        "detector_forward_ms": detector_forward_ms,
        "decode_nms_ms": decode_nms_ms,
        "total_compute_ms": preprocess_ms + detector_forward_ms + decode_nms_ms,
    }


def _detector_gflops(bundle, imgsz: int) -> float:
    """Run THOP after temporarily removing bound forward hooks."""
    from ultralytics.utils.torch_utils import get_flops

    from myscripts.V3.runtime import FrozenCAExtractor

    extractor = bundle.extractor
    settings = {
        "device": extractor.device,
        "nc": extractor.nc,
        "conf": extractor.conf,
        "nms_iou": extractor.nms_iou,
        "max_det": extractor.max_det,
        "amp": extractor.amp,
    }
    extractor.close()
    try:
        value = float(get_flops(bundle.model, imgsz=imgsz))
    finally:
        bundle.extractor = FrozenCAExtractor(bundle.model, **settings)
    if value <= 0:
        raise RuntimeError("detector FLOPs could not be calculated; verify ultralytics-thop")
    return value


def _profile_detector(
    *,
    label: str,
    weights: str,
    expected_reg_max: int,
    torch,
    device,
    dataset,
    batch: int,
    workers: int,
    warmup: int,
    max_images: int,
    conf: float,
    nms_iou: float,
    max_det: int,
    imgsz: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from myscripts.V3.runtime import full_loader, sha256_file

    bundle = load_obb_detector(
        weights,
        device=device,
        amp=False,
        conf=conf,
        nms_iou=nms_iou,
        max_det=max_det,
        expected_reg_max=expected_reg_max,
    )
    try:
        loader = full_loader(dataset, batch, workers, shuffle=False)
        try:
            warmup_batch = next(iter(loader))
        except StopIteration as exc:
            raise RuntimeError("validation loader is empty") from exc
        with torch.inference_mode():
            for _ in range(warmup):
                _run_detector_batch(torch, bundle.extractor, warmup_batch, timed=False)
        _sync(torch, device)

        if device.type == "cuda":
            _sync(torch, device)
            bundle.extractor.cache.clear()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            baseline_memory = torch.cuda.memory_allocated(device)
        else:
            baseline_memory = 0

        if hasattr(loader, "reset"):
            loader.reset()
        rows: list[dict[str, Any]] = []
        iterator = iter(loader)
        measured = 0
        while max_images == 0 or measured < max_images:
            try:
                current = next(iterator)
            except StopIteration:
                break
            if int(current["img"].shape[0]) != 1:
                raise RuntimeError("official comparative profiler requires batch=1")
            with torch.inference_mode():
                detections, stage = _run_detector_batch(torch, bundle.extractor, current, timed=True)
            rows.append(
                {
                    "method": label,
                    "image_index": measured,
                    "image_path": str(current["im_file"][0]),
                    "input_height": int(current["img"].shape[2]),
                    "input_width": int(current["img"].shape[3]),
                    "proposal_count": int(detections[0]["bboxes"].shape[0]),
                    **stage,
                }
            )
            measured += 1
        if not rows:
            raise RuntimeError(f"no images profiled for {label}")

        if device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated(device)
        else:
            peak_memory = 0

        # Keep THOP outside the official latency and memory window.  Its CUDA
        # kernels and temporary allocations otherwise heat/pollute one method
        # differently from another.
        gflops = _detector_gflops(bundle, imgsz)

        state_hash_after = sha256_file(bundle.weights_path)
        if state_hash_after != bundle.weights_hash:
            raise RuntimeError(f"{label} checkpoint changed during profiling")
        input_shape_counts: dict[str, int] = {}
        for row in rows:
            shape = f"{row['input_height']}x{row['input_width']}"
            input_shape_counts[shape] = input_shape_counts.get(shape, 0) + 1
        timing = {
            key: summarize_timings(rows, key)
            for key in ("preprocess_ms", "detector_forward_ms", "decode_nms_ms", "total_compute_ms")
        }
        mean_total = timing["total_compute_ms"]["mean"]
        summary = {
            "method": label,
            "weights": str(bundle.weights_path),
            "weights_sha256": bundle.weights_hash,
            "weights_unchanged": True,
            "reg_max": bundle.reg_max,
            "parameters": sum(parameter.numel() for parameter in bundle.model.parameters()),
            "gflops": gflops,
            "gflops_scope": "detector forward at nominal imgsz=640",
            "measured_images": len(rows),
            "input_shape_counts": input_shape_counts,
            "timing_ms_per_image": timing,
            "fps_from_mean_compute": 1000.0 / mean_total,
            "gpu_memory": {
                "baseline_allocated_gb": baseline_memory / 1024**3,
                "peak_allocated_gb": peak_memory / 1024**3,
                "incremental_peak_gb": max(peak_memory - baseline_memory, 0) / 1024**3,
                "scope": "absolute allocated memory in this standalone process; peak window excludes complexity profiling",
            },
        }
        return summary, rows
    finally:
        bundle.close()
        del bundle
        if device.type == "cuda":
            torch.cuda.empty_cache()


def balanced_method_orders(repeats: int = 3) -> tuple[tuple[str, ...], ...]:
    """Return the locked 3x3 Latin-square process order."""
    if repeats != len(METHODS):
        raise ValueError(f"official protocol requires {len(METHODS)} repeats")
    return tuple(METHODS[shift:] + METHODS[:shift] for shift in range(repeats))


def _relative_spread(values: list[float]) -> float:
    mean = statistics.fmean(values)
    return (max(values) - min(values)) / mean if mean > 0 else 0.0


def _run_isolated_worker(
    *,
    method: str,
    repeat: int,
    order_position: int,
    args: argparse.Namespace,
    refine_summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    run_slug = method.lower().replace("+", "_").replace(" ", "_")
    run_dir = output_dir / "isolated_runs" / f"repeat_{repeat:02d}_position_{order_position:02d}_{run_slug}"
    common = [
        "--data",
        args.data,
        "--split",
        args.split,
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--device",
        str(args.device),
        "--workers",
        str(args.workers),
        "--warmup",
        str(args.warmup),
        "--max-images",
        str(args.max_images),
        "--output-dir",
        str(run_dir),
    ]
    if method in {"Baseline", "CA"}:
        weights = args.baseline_weights if method == "Baseline" else args.ca_weights
        command = [
            sys.executable,
            "-m",
            "myscripts.V3_1_1.profile_detector_worker_v311",
            "--method",
            method,
            "--weights",
            weights,
            "--conf",
            str(refine_summary["confidence"]),
            "--nms-iou",
            str(refine_summary["nms_iou"]),
            "--max-det",
            str(refine_summary["max_det"]),
            *common,
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "myscripts.V3_1_1.profile_refine_v311",
            "--checkpoint",
            str(refine_summary["refine_checkpoint"]),
            "--ca-weights",
            args.ca_weights,
            "--no-amp",
            *common,
        ]
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    print(
        f"[comparative] repeat={repeat} position={order_position} method={method} "
        f"isolated_output={run_dir}"
    )
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"isolated profile failed: repeat={repeat}, position={order_position}, "
            f"method={method}, returncode={result.returncode}"
        )
    summary_path = run_dir / "profile_summary.json"
    rows_path = run_dir / "profile_per_image.csv"
    if not summary_path.is_file() or not rows_path.is_file():
        raise RuntimeError(f"isolated worker did not create complete outputs: {run_dir}")
    return {
        "method": method,
        "repeat": repeat,
        "order_position": order_position,
        "run_dir": str(run_dir),
        "summary": _read_json(summary_path),
        "rows": _read_csv(rows_path),
    }


def _repeat_row(record: dict[str, Any]) -> dict[str, Any]:
    summary = record["summary"]
    total = summary["timing_ms_per_image"]["total_compute_ms"]
    coarse = summary["timing_ms_per_image"].get("coarse_compute_ms")
    return {
        "method": record["method"],
        "repeat": record["repeat"],
        "order_position": record["order_position"],
        "process_id": summary.get("process_id"),
        "measured_images": summary["measured_images"],
        "latency_mean_ms": total["mean"],
        "latency_median_ms": total["median"],
        "latency_p95_ms": total["p95"],
        "coarse_mean_ms": coarse["mean"] if coarse else "",
        "peak_gpu_memory_gb": summary["gpu_memory"]["peak_allocated_gb"],
        "incremental_peak_gpu_memory_gb": summary["gpu_memory"].get("incremental_peak_gb", 0.0),
        "run_dir": record["run_dir"],
    }


def _aggregate_method(
    method: str,
    records: list[dict[str, Any]],
    *,
    latency_tolerance: float,
    memory_tolerance: float,
) -> dict[str, Any]:
    if len(records) != 3:
        raise RuntimeError(f"{method} does not have exactly three isolated repeats")
    summaries = [record["summary"] for record in records]
    rows = [row for record in records for row in record["rows"]]
    repeat_means = [float(summary["timing_ms_per_image"]["total_compute_ms"]["mean"]) for summary in summaries]
    peaks = [float(summary["gpu_memory"]["peak_allocated_gb"]) for summary in summaries]
    increments = [float(summary["gpu_memory"].get("incremental_peak_gb", 0.0)) for summary in summaries]
    pooled_timing = summarize_timings(rows, "total_compute_ms")
    first = summaries[0]
    if method == "CA+Refine":
        parameters = int(first["total_parameters"])
        gflops = float(first["detector_gflops"]) + float(first["refiner_profiled_gflops"])
        gflops_scope = "lower bound: detector + profiler-attributed Refine operators"
        identity = {
            "ca_weights": first["ca_weights"],
            "ca_sha256": first["ca_sha256"],
            "refine_checkpoint": first["refine_checkpoint"],
            "refine_sha256": first["refine_sha256"],
        }
        parameter_values = [int(summary["total_parameters"]) for summary in summaries]
        gflop_values = [
            float(summary["detector_gflops"]) + float(summary["refiner_profiled_gflops"])
            for summary in summaries
        ]
    else:
        parameters = int(first["parameters"])
        gflops = float(first["gflops"])
        gflops_scope = first["gflops_scope"]
        identity = {
            "weights": first["weights"],
            "weights_sha256": first["weights_sha256"],
        }
        parameter_values = [int(summary["parameters"]) for summary in summaries]
        gflop_values = [float(summary["gflops"]) for summary in summaries]
    gflop_tolerance = max(1e-6, abs(statistics.fmean(gflop_values)) * 1e-6)
    if len(set(parameter_values)) != 1 or max(gflop_values) - min(gflop_values) > gflop_tolerance:
        raise RuntimeError(f"{method} parameters or FLOPs changed across repeats")
    if any(summary["input_shape_counts"] != first["input_shape_counts"] for summary in summaries):
        raise RuntimeError(f"{method} input shapes changed across repeats")
    if any(int(summary["measured_images"]) != int(first["measured_images"]) for summary in summaries):
        raise RuntimeError(f"{method} image count changed across repeats")
    latency_spread = _relative_spread(repeat_means)
    memory_spread = _relative_spread(peaks)
    mean_latency = statistics.fmean(repeat_means)
    return {
        "method": method,
        **identity,
        "weights_unchanged": all(summary.get("weights_unchanged") is True for summary in summaries),
        "repeat_count": len(records),
        "measured_images_per_repeat": int(first["measured_images"]),
        "total_timed_observations": len(rows),
        "input_shape_counts": first["input_shape_counts"],
        "parameters": parameters,
        "gflops": gflops,
        "gflops_scope": gflops_scope,
        "repeat_latency_means_ms": repeat_means,
        "timing_ms_per_image": pooled_timing,
        "latency_mean_of_repeat_means_ms": mean_latency,
        "latency_between_repeat_std_ms": statistics.pstdev(repeat_means),
        "latency_relative_spread": latency_spread,
        "latency_stability_pass": latency_spread <= latency_tolerance,
        "fps_from_mean_compute": 1000.0 / mean_latency,
        "isolated_peak_gpu_memory_gb": {
            "values": peaks,
            "median": statistics.median(peaks),
            "min": min(peaks),
            "max": max(peaks),
            "relative_spread": memory_spread,
            "stability_pass": memory_spread <= memory_tolerance,
        },
        "isolated_incremental_peak_gpu_memory_gb": {
            "values": increments,
            "median": statistics.median(increments),
        },
    }


def _comparison_row(summary: dict[str, Any]) -> dict[str, Any]:
    timing = summary["timing_ms_per_image"]
    memory = summary["isolated_peak_gpu_memory_gb"]
    return {
        "method": summary["method"],
        "parameters_m": summary["parameters"] / 1e6,
        "gflops": summary["gflops"],
        "gflops_scope": summary["gflops_scope"],
        "latency_mean_ms": summary["latency_mean_of_repeat_means_ms"],
        "latency_between_repeat_std_ms": summary["latency_between_repeat_std_ms"],
        "latency_median_ms": timing["median"],
        "latency_p95_ms": timing["p95"],
        "latency_relative_spread": summary["latency_relative_spread"],
        "fps_from_mean_compute": summary["fps_from_mean_compute"],
        "peak_gpu_memory_gb": memory["median"],
        "peak_gpu_memory_min_gb": memory["min"],
        "peak_gpu_memory_max_gb": memory["max"],
        "incremental_peak_gpu_memory_gb": summary["isolated_incremental_peak_gpu_memory_gb"]["median"],
        "repeat_count": summary["repeat_count"],
        "measured_images_per_repeat": summary["measured_images_per_repeat"],
        "input_shape_counts": json.dumps(summary["input_shape_counts"], sort_keys=True),
    }


def main(argv: list[str] | None = None) -> None:
    ensure_omp_threads()
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    refine_profile_path = Path(args.refine_profile_summary)
    refine_reference = _read_json(refine_profile_path)
    refine_reference_rows_path = refine_profile_path.with_name("profile_per_image.csv")
    if not refine_reference_rows_path.is_file():
        raise FileNotFoundError(f"Refine per-image profile not found: {refine_reference_rows_path}")
    refine_reference_rows = _read_csv(refine_reference_rows_path)
    expected_refine = {
        "split": args.split,
        "test_used": False,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "amp": False,
        "workers": args.workers,
        "warmup_passes": args.warmup,
        "ca_weights": str(Path(args.ca_weights)),
        "weights_unchanged": True,
    }
    mismatches = {
        key: {"expected": value, "observed": refine_reference.get(key)}
        for key, value in expected_refine.items()
        if refine_reference.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Refine profile protocol mismatch: {mismatches}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    orders = balanced_method_orders(args.repeats)
    records: list[dict[str, Any]] = []
    for repeat, order in enumerate(orders, start=1):
        for position, method in enumerate(order, start=1):
            records.append(
                _run_isolated_worker(
                    method=method,
                    repeat=repeat,
                    order_position=position,
                    args=args,
                    refine_summary=refine_reference,
                    output_dir=output_dir,
                )
            )

    canonical_paths = [row["image_path"] for row in records[0]["rows"]]
    canonical_shapes = records[0]["summary"]["input_shape_counts"]
    same_image_order = all([row["image_path"] for row in record["rows"]] == canonical_paths for record in records)
    same_shapes = all(record["summary"]["input_shape_counts"] == canonical_shapes for record in records)
    same_count = all(len(record["rows"]) == len(canonical_paths) for record in records)
    ca_proposals_match = True
    for repeat in range(1, args.repeats + 1):
        ca_record = next(record for record in records if record["repeat"] == repeat and record["method"] == "CA")
        refine_record = next(
            record for record in records if record["repeat"] == repeat and record["method"] == "CA+Refine"
        )
        if [int(row["proposal_count"]) for row in ca_record["rows"]] != [
            int(row["proposal_count"]) for row in refine_record["rows"]
        ]:
            ca_proposals_match = False
            break
    reference_compare_rows = (
        refine_reference_rows[: args.max_images] if args.max_images > 0 else refine_reference_rows
    )
    reference_paths_match = [row["image_path"] for row in reference_compare_rows] == canonical_paths
    reference_proposals = [int(row["proposal_count"]) for row in reference_compare_rows]
    external_proposals_match = all(
        [int(row["proposal_count"]) for row in record["rows"]] == reference_proposals
        for record in records
        if record["method"] in {"CA", "CA+Refine"}
    )
    if not (
        same_image_order
        and same_shapes
        and same_count
        and ca_proposals_match
        and reference_paths_match
        and external_proposals_match
    ):
        raise RuntimeError(
            "comparative profile mismatch in image count/order, input tensor shapes, or CA proposals"
        )

    grouped = {method: [record for record in records if record["method"] == method] for method in METHODS}
    aggregates = {
        method: _aggregate_method(
            method,
            grouped[method],
            latency_tolerance=args.stability_relative_tolerance,
            memory_tolerance=args.memory_relative_tolerance,
        )
        for method in METHODS
    }
    ca_mean = aggregates["CA"]["latency_mean_of_repeat_means_ms"]
    refine_coarse_means = [
        float(record["summary"]["timing_ms_per_image"]["coarse_compute_ms"]["mean"])
        for record in grouped["CA+Refine"]
    ]
    refine_coarse_mean = statistics.fmean(refine_coarse_means)
    ca_refine_coarse_relative_delta = (refine_coarse_mean - ca_mean) / ca_mean
    ca_refine_coarse_consistency = abs(ca_refine_coarse_relative_delta) <= args.stability_relative_tolerance
    latency_stability = all(summary["latency_stability_pass"] for summary in aggregates.values())
    memory_stability = all(
        summary["isolated_peak_gpu_memory_gb"]["stability_pass"] for summary in aggregates.values()
    )
    worker_isolation = all(
        record["summary"].get("isolated_process") is True
        and int(record["summary"].get("process_id", os.getpid())) != os.getpid()
        for record in records
    )
    comparison = [_comparison_row(aggregates[method]) for method in METHODS]
    repeat_rows = [_repeat_row(record) for record in records]
    per_image_rows: list[dict[str, Any]] = []
    for record in records:
        for row in record["rows"]:
            per_image_rows.append(
                {
                    **row,
                    "method": record["method"],
                    "repeat": record["repeat"],
                    "order_position": record["order_position"],
                }
            )

    external_refine_mean = float(refine_reference["timing_ms_per_image"]["total_compute_ms"]["mean"])
    new_refine_mean = aggregates["CA+Refine"]["latency_mean_of_repeat_means_ms"]
    audit = {
        "tool": "profile_comparative_v311",
        "protocol_version": 2,
        "latency_method": "three full-split synchronized repeats with a 3x3 Latin-square process order",
        "latency_scope": "preprocess + unfused detector/P2-P3 evidence hooks + decode/rotated NMS; Refine adds proposal packing, rotated ROI refinement, and geometry write-back",
        "memory_method": "one method per fresh worker process; absolute peak allocated memory excludes THOP/torch.profiler",
        "data": args.data,
        "split": args.split,
        "test_used": False,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "amp": False,
        "device": records[0]["summary"].get("device", str(args.device)),
        "device_arg": str(args.device),
        "workers": args.workers,
        "warmup_passes": args.warmup,
        "repeat_count": args.repeats,
        "balanced_run_orders": [list(order) for order in orders],
        "measured_images": len(canonical_paths),
        "input_shape_counts": canonical_shapes,
        "same_images_and_shapes": same_shapes and same_count,
        "same_image_order": same_image_order,
        "ca_proposals_match_refine_profile": ca_proposals_match,
        "external_refine_paths_match": reference_paths_match,
        "external_refine_proposals_match": external_proposals_match,
        "worker_process_isolation": worker_isolation,
        "worker_process_ids": [record["summary"].get("process_id") for record in records],
        "latency_stability_relative_tolerance": args.stability_relative_tolerance,
        "latency_stability_pass": latency_stability,
        "memory_stability_relative_tolerance": args.memory_relative_tolerance,
        "memory_stability_pass": memory_stability,
        "ca_refine_coarse_mean_ms": refine_coarse_mean,
        "ca_standalone_mean_ms": ca_mean,
        "ca_refine_coarse_relative_delta": ca_refine_coarse_relative_delta,
        "ca_refine_coarse_consistency_pass": ca_refine_coarse_consistency,
        "reportable_efficiency_pass": (
            worker_isolation and latency_stability and memory_stability and ca_refine_coarse_consistency
        ),
        "confidence": float(refine_reference["confidence"]),
        "nms_iou": float(refine_reference["nms_iou"]),
        "max_det": int(refine_reference["max_det"]),
        "feature_hook_policy": "identical P2/P3 reference-capture hooks enabled for Baseline, CA, and CA+Refine detector forward",
        "refine_profile_summary": str(refine_profile_path),
        "external_refine_latency_relative_delta": (
            (new_refine_mean - external_refine_mean) / external_refine_mean
            if args.max_images == 0
            else None
        ),
        "baseline": aggregates["Baseline"],
        "ca": aggregates["CA"],
        "refine": aggregates["CA+Refine"],
        "isolated_runs": [
            {
                "method": record["method"],
                "repeat": record["repeat"],
                "order_position": record["order_position"],
                "run_dir": record["run_dir"],
                "process_id": record["summary"].get("process_id"),
            }
            for record in records
        ],
        "comparison": comparison,
        "gflops_note": "CA+Refine FLOPs are a lower bound because torch.profiler may not attribute grid_sample/NMS FLOPs; synchronized full-chain latency is authoritative.",
    }
    write_csv(output_dir / "comparative_per_image.csv", per_image_rows)
    write_csv(output_dir / "comparative_repeat_summary.csv", repeat_rows)
    write_csv(output_dir / "comparative_latency.csv", comparison)
    write_json(output_dir / "comparative_profile.json", audit)
    print(output_dir / "comparative_latency.csv")


if __name__ == "__main__":
    main()
