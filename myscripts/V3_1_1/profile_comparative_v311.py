"""Profile Baseline, CA, and CA+Refine with one synchronized FP32 pipeline.

The script profiles the two frozen detectors with the exact preprocessing,
forward, rotated-NMS, synchronization, dataloader, and input-shape policy used
by :mod:`profile_refine_v311`.  It then combines those measurements with the
already generated full-chain Refine profile, producing an apples-to-apples
latency table without mixing Ultralytics validator timing with the custom
proposal-refinement runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
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
        gflops = _detector_gflops(bundle, imgsz)

        if device.type == "cuda":
            _sync(torch, device)
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
                "peak_allocated_gb": (
                    torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
                ),
            },
        }
        return summary, rows
    finally:
        bundle.close()
        del bundle
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _comparison_row(method: str, summary: dict[str, Any], *, refined: bool) -> dict[str, Any]:
    if refined:
        timing = summary["timing_ms_per_image"]["total_compute_ms"]
        return {
            "method": method,
            "parameters_m": summary["total_parameters"] / 1e6,
            "gflops": summary["detector_gflops"] + summary["refiner_profiled_gflops"],
            "gflops_scope": "lower bound: detector + profiler-attributed Refine operators",
            "latency_mean_ms": timing["mean"],
            "latency_median_ms": timing["median"],
            "latency_p95_ms": timing["p95"],
            "fps_from_mean_compute": summary["refined_fps_from_mean_compute"],
            "peak_gpu_memory_gb": summary["gpu_memory"]["peak_allocated_gb"],
            "measured_images": summary["measured_images"],
            "input_shape_counts": json.dumps(summary["input_shape_counts"], sort_keys=True),
        }
    timing = summary["timing_ms_per_image"]["total_compute_ms"]
    return {
        "method": method,
        "parameters_m": summary["parameters"] / 1e6,
        "gflops": summary["gflops"],
        "gflops_scope": summary["gflops_scope"],
        "latency_mean_ms": timing["mean"],
        "latency_median_ms": timing["median"],
        "latency_p95_ms": timing["p95"],
        "fps_from_mean_compute": summary["fps_from_mean_compute"],
        "peak_gpu_memory_gb": summary["gpu_memory"]["peak_allocated_gb"],
        "measured_images": summary["measured_images"],
        "input_shape_counts": json.dumps(summary["input_shape_counts"], sort_keys=True),
    }


def main(argv: list[str] | None = None) -> None:
    ensure_omp_threads()
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    import torch

    from ultralytics.utils.torch_utils import select_device

    from myscripts.V3.runtime import build_dataset

    refine_profile_path = Path(args.refine_profile_summary)
    refine_summary = _read_json(refine_profile_path)
    refine_per_image_path = refine_profile_path.with_name("profile_per_image.csv")
    if not refine_per_image_path.is_file():
        raise FileNotFoundError(f"Refine per-image profile not found: {refine_per_image_path}")
    refine_rows = _read_csv(refine_per_image_path)
    device = select_device(args.device)
    expected_refine = {
        "split": args.split,
        "test_used": False,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "amp": False,
        "device": str(device),
        "workers": args.workers,
        "warmup_passes": args.warmup,
        "ca_weights": str(Path(args.ca_weights)),
        "weights_unchanged": True,
    }
    mismatches = {
        key: {"expected": value, "observed": refine_summary.get(key)}
        for key, value in expected_refine.items()
        if refine_summary.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Refine profile protocol mismatch: {mismatches}")

    dataset, _data = build_dataset(args.data, args.split, args.imgsz, args.batch, args.workers, rect=True)
    common = {
        "torch": torch,
        "device": device,
        "dataset": dataset,
        "batch": args.batch,
        "workers": args.workers,
        "warmup": args.warmup,
        "max_images": args.max_images,
        "conf": float(refine_summary["confidence"]),
        "nms_iou": float(refine_summary["nms_iou"]),
        "max_det": int(refine_summary["max_det"]),
        "imgsz": args.imgsz,
    }
    baseline_summary, baseline_rows = _profile_detector(
        label="Baseline",
        weights=args.baseline_weights,
        expected_reg_max=16,
        **common,
    )
    ca_summary, ca_rows = _profile_detector(
        label="CA",
        weights=args.ca_weights,
        expected_reg_max=32,
        **common,
    )

    shape_match = (
        baseline_summary["input_shape_counts"]
        == ca_summary["input_shape_counts"]
        == refine_summary["input_shape_counts"]
    )
    count_match = (
        baseline_summary["measured_images"]
        == ca_summary["measured_images"]
        == refine_summary["measured_images"]
    )
    baseline_paths = [row["image_path"] for row in baseline_rows]
    ca_paths = [row["image_path"] for row in ca_rows]
    refine_paths = [row["image_path"] for row in refine_rows]
    path_order_match = baseline_paths == ca_paths == refine_paths
    ca_proposals_match = [int(row["proposal_count"]) for row in ca_rows] == [
        int(row["proposal_count"]) for row in refine_rows
    ]
    if not shape_match or not count_match or not path_order_match or not ca_proposals_match:
        raise RuntimeError(
            "comparative profile mismatch in image count/order, input tensor shapes, or CA proposals"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _comparison_row("Baseline", baseline_summary, refined=False),
        _comparison_row("CA", ca_summary, refined=False),
        _comparison_row("CA+Refine", refine_summary, refined=True),
    ]
    audit = {
        "tool": "profile_comparative_v311",
        "latency_method": "synchronized wall clock at every declared stage",
        "latency_scope": "preprocess + unfused detector/P2-P3 evidence hooks + decode/rotated NMS; Refine adds proposal packing, rotated ROI refinement, and geometry write-back",
        "data": args.data,
        "split": args.split,
        "test_used": False,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "amp": False,
        "device": str(device),
        "workers": args.workers,
        "warmup_passes": args.warmup,
        "measured_images": baseline_summary["measured_images"],
        "input_shape_counts": baseline_summary["input_shape_counts"],
        "same_images_and_shapes": shape_match and count_match,
        "same_image_order": path_order_match,
        "ca_proposals_match_refine_profile": ca_proposals_match,
        "confidence": common["conf"],
        "nms_iou": common["nms_iou"],
        "max_det": common["max_det"],
        "feature_hook_policy": "identical P2/P3 reference-capture hooks enabled for Baseline, CA, and CA+Refine detector forward",
        "refine_profile_summary": str(refine_profile_path),
        "baseline": baseline_summary,
        "ca": ca_summary,
        "refine": refine_summary,
        "comparison": rows,
        "gflops_note": "CA+Refine FLOPs are a lower bound because torch.profiler may not attribute grid_sample/NMS FLOPs; synchronized full-chain latency is authoritative.",
    }
    refine_rows_with_method = [dict(row, method="CA+Refine") for row in refine_rows]
    write_csv(output_dir / "comparative_per_image.csv", baseline_rows + ca_rows + refine_rows_with_method)
    write_csv(output_dir / "comparative_latency.csv", rows)
    write_json(output_dir / "comparative_profile.json", audit)
    print(output_dir / "comparative_latency.csv")


if __name__ == "__main__":
    main()
