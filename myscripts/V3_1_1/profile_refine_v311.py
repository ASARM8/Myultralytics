"""Profile the complete frozen CA -> NMS -> proposal Refine V3.1.1 chain.

The official paper command is intentionally locked to FP32, batch size 1,
``imgsz=640`` and the validation split.  The output separates data loading,
tensor preprocessing, detector forward (including P2/P3 capture), rotated NMS,
proposal packing, rotated-ROI refinement, and final geometry write-back.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from myscripts.V3.train_refine_v3 import write_csv, write_json
from myscripts.V3_1_1.evidence_runtime import (
    CANONICAL_CA_WEIGHTS,
    load_refine_bundle,
    require_canonical_path,
)


STAGES = (
    "preprocess_ms",
    "detector_forward_ms",
    "decode_nms_ms",
    "proposal_pack_ms",
    "refiner_ms",
    "writeback_ms",
)
PROFILE_PROTOCOL_VERSION = 3
OFFICIAL_WARMUP_PASSES = 500


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ca-weights", default=CANONICAL_CA_WEIGHTS)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val", choices=("val",))
    parser.add_argument("--imgsz", type=int, default=640, choices=(640,))
    parser.add_argument("--batch", type=int, default=1, choices=(1,))
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--warmup",
        type=int,
        default=OFFICIAL_WARMUP_PASSES,
        help="Warmup passes on the first validation image",
    )
    parser.add_argument("--max-images", type=int, default=0, help="0 profiles the entire validation split")
    parser.add_argument("--output-dir", required=True)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    require_canonical_path(parser, args.ca_weights, CANONICAL_CA_WEIGHTS, "CA checkpoint")
    if args.workers < 0 or args.warmup < 1 or args.max_images < 0:
        parser.error("workers/max-images must be non-negative and warmup must be positive")
    if args.amp:
        parser.error("official V3.1.1 profiling is locked to FP32; use --no-amp")


def _percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile without a NumPy dependency."""
    if not values:
        return math.nan
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_timings(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    """Summarize a per-image timing field for the audit JSON."""
    values = [float(row[key]) for row in rows]
    return {
        "mean": statistics.fmean(values) if values else math.nan,
        "median": statistics.median(values) if values else math.nan,
        "p95": _percentile(values, 0.95),
        "min": min(values) if values else math.nan,
        "max": max(values) if values else math.nan,
    }


def _sync(torch, device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(torch, device, operation: Callable[[], Any]) -> tuple[Any, float]:
    _sync(torch, device)
    start = time.perf_counter()
    value = operation()
    _sync(torch, device)
    return value, (time.perf_counter() - start) * 1000.0


def _format_nms(torch, inference, extractor):
    from ultralytics.utils import nms

    detections = nms.non_max_suppression(
        inference,
        extractor.conf,
        extractor.nms_iou,
        nc=extractor.nc,
        multi_label=True,
        agnostic=False,
        max_det=extractor.max_det,
        rotated=True,
    )
    return [
        {
            "bboxes": torch.cat((item[:, :4], item[:, -1:]), dim=1),
            "conf": item[:, 4],
            "cls": item[:, 5],
        }
        for item in detections
    ]


def _run_pipeline(bundle, batch: dict[str, Any], *, timed: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one batch and optionally return synchronized wall-clock stage times."""
    torch = bundle.torch
    device = bundle.device
    extractor = bundle.extractor

    def measure(operation):
        if timed:
            return _timed(torch, device, operation)
        return operation(), 0.0

    images, preprocess_ms = measure(
        lambda: batch["img"].to(device, non_blocking=True).float().div_(255.0)
    )

    def detector_forward():
        extractor.cache.clear()
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=bundle.use_amp,
        ):
            outputs = bundle.ca_model(images)
        if "p2" not in extractor.cache or "p3" not in extractor.cache:
            raise RuntimeError("P2/P3 feature hooks did not fire during profiling")
        inference = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if not isinstance(inference, torch.Tensor) or inference.ndim != 3:
            raise TypeError(f"expected CA inference tensor [B,C,N], received {type(inference)}")
        return inference

    inference, detector_forward_ms = measure(detector_forward)
    detections, decode_nms_ms = measure(lambda: _format_nms(torch, inference, extractor))

    from myscripts.V3.runtime import pad_detections

    packed, proposal_pack_ms = measure(lambda: pad_detections(detections))
    boxes, scores, classes, valid = packed

    def refine_forward():
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=bundle.use_amp,
        ):
            return bundle.refiner(
                extractor.cache["p2"],
                extractor.cache["p3"],
                boxes,
                scores,
                images.shape[2:],
                valid,
            )

    output, refiner_ms = measure(refine_forward)
    residual = output["residual"].float()

    def writeback():
        refined = []
        for image_index in range(images.shape[0]):
            count = int(valid[image_index].sum().item())
            refined.append(
                {
                    "bboxes": bundle.refiner.apply_residual(
                        boxes[image_index, :count].float(), residual[image_index, :count]
                    ),
                    "conf": scores[image_index, :count],
                    "cls": classes[image_index, :count],
                }
            )
        return refined

    refined, writeback_ms = measure(writeback)
    times = {
        "preprocess_ms": preprocess_ms,
        "detector_forward_ms": detector_forward_ms,
        "decode_nms_ms": decode_nms_ms,
        "proposal_pack_ms": proposal_pack_ms,
        "refiner_ms": refiner_ms,
        "writeback_ms": writeback_ms,
    }
    return {
        "images": images,
        "p2": extractor.cache["p2"],
        "p3": extractor.cache["p3"],
        "boxes": boxes,
        "scores": scores,
        "classes": classes,
        "valid": valid,
        "refined": refined,
    }, times


def _profile_refiner_flops(bundle, sample: dict[str, Any]) -> tuple[float, int]:
    """Profile supported Refine operators on one real batch using torch.profiler."""
    torch = bundle.torch
    activities = [torch.profiler.ProfilerActivity.CPU]
    if bundle.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.inference_mode(), torch.profiler.profile(activities=activities, with_flops=True) as profile:
        bundle.refiner(
            sample["p2"],
            sample["p3"],
            sample["boxes"],
            sample["scores"],
            sample["images"].shape[2:],
            sample["valid"],
        )
    _sync(torch, bundle.device)
    flops = sum(float(item.flops or 0.0) for item in profile.key_averages())
    proposals = int(sample["valid"].sum().item())
    if flops <= 0:
        raise RuntimeError("torch.profiler reported zero Refine FLOPs; verify the cloud PyTorch profiler build")
    return flops / 1e9, proposals


def _detector_gflops_without_evidence_hooks(bundle, imgsz: int) -> float:
    """Run THOP on the detector without deep-copying bound evidence hooks."""
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
        value = float(get_flops(bundle.ca_model, imgsz=imgsz))
    finally:
        bundle.extractor = FrozenCAExtractor(bundle.ca_model, **settings)
    return value


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    from myscripts.V3.runtime import build_dataset, full_loader

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_refine_bundle(
        args.checkpoint,
        args.ca_weights,
        device_arg=args.device,
        amp=args.amp,
        imgsz=args.imgsz,
    )
    torch = bundle.torch
    try:
        dataset, _data = build_dataset(args.data, args.split, args.imgsz, args.batch, args.workers, rect=True)
        loader = full_loader(dataset, args.batch, args.workers)
        try:
            warmup_batch = next(iter(loader))
        except StopIteration as exc:
            raise RuntimeError("validation loader is empty") from exc
        with torch.inference_mode():
            for _ in range(args.warmup):
                sample, _ = _run_pipeline(bundle, warmup_batch, timed=False)
        _sync(torch, bundle.device)

        del sample, warmup_batch

        if bundle.device.type == "cuda":
            _sync(torch, bundle.device)
            bundle.extractor.cache.clear()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(bundle.device)
            baseline_memory = torch.cuda.memory_allocated(bundle.device)
        else:
            baseline_memory = 0

        rows: list[dict[str, Any]] = []
        if hasattr(loader, "reset"):
            loader.reset()
        iterator = iter(loader)
        measured = 0
        while args.max_images == 0 or measured < args.max_images:
            load_start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            data_load_ms = (time.perf_counter() - load_start) * 1000.0
            batch_images = int(batch["img"].shape[0])
            if batch_images != 1:
                raise RuntimeError("official profiler requires batch=1")
            with torch.inference_mode():
                sample, stage = _run_pipeline(bundle, batch, timed=True)
            proposal_count = int(sample["valid"].sum().item())
            coarse_ms = stage["preprocess_ms"] + stage["detector_forward_ms"] + stage["decode_nms_ms"]
            refine_added_ms = stage["proposal_pack_ms"] + stage["refiner_ms"] + stage["writeback_ms"]
            total_compute_ms = coarse_ms + refine_added_ms
            row = {
                "image_index": measured,
                "image_path": str(batch["im_file"][0]),
                "input_height": int(batch["img"].shape[2]),
                "input_width": int(batch["img"].shape[3]),
                "proposal_count": proposal_count,
                "data_load_ms": data_load_ms,
                **stage,
                "coarse_compute_ms": coarse_ms,
                "refine_added_ms": refine_added_ms,
                "total_compute_ms": total_compute_ms,
                "total_with_data_ms": total_compute_ms + data_load_ms,
            }
            rows.append(row)
            measured += 1

        if not rows:
            raise RuntimeError("no validation image was profiled")
        if bundle.device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated(bundle.device)
        else:
            peak_memory = 0

        # Complexity profiling is intentionally delayed until after the latency
        # and peak-memory window.  THOP and torch.profiler launch additional CUDA
        # kernels and allocate temporary buffers; running them before the official
        # measurements made the Refine process systematically hotter than the two
        # detector-only processes.
        if int(sample["valid"].sum().item()) == 0:
            if hasattr(loader, "reset"):
                loader.reset()
            for candidate_batch in loader:
                with torch.inference_mode():
                    candidate, _ = _run_pipeline(bundle, candidate_batch, timed=False)
                if int(candidate["valid"].sum().item()) > 0:
                    sample = candidate
                    break
            else:
                raise RuntimeError("no post-NMS proposal is available for Refine FLOP profiling")
        detector_gflops = _detector_gflops_without_evidence_hooks(bundle, args.imgsz)
        if detector_gflops <= 0:
            raise RuntimeError(
                "detector FLOPs could not be calculated; install/repair ultralytics-thop in the cloud environment"
            )
        refiner_gflops, flop_proposals = _profile_refiner_flops(bundle, sample)

        detector_parameters = sum(parameter.numel() for parameter in bundle.ca_model.parameters())
        refiner_parameters = sum(parameter.numel() for parameter in bundle.refiner.parameters())
        from myscripts.V3.runtime import sha256_file

        ca_hash_after = sha256_file(bundle.ca_path)
        refine_hash_after = sha256_file(bundle.checkpoint_path)
        weights_unchanged = ca_hash_after == bundle.ca_hash and refine_hash_after == bundle.checkpoint_hash
        if not weights_unchanged:
            raise RuntimeError("CA or Refine checkpoint changed during profiling")
        timing_keys = ("data_load_ms",) + STAGES + (
            "coarse_compute_ms",
            "refine_added_ms",
            "total_compute_ms",
            "total_with_data_ms",
        )
        timing = {key: summarize_timings(rows, key) for key in timing_keys}
        input_shape_counts: dict[str, int] = {}
        for row in rows:
            shape = f"{row['input_height']}x{row['input_width']}"
            input_shape_counts[shape] = input_shape_counts.get(shape, 0) + 1
        mean_total = timing["total_compute_ms"]["mean"]
        mean_coarse = timing["coarse_compute_ms"]["mean"]
        summary = {
            "tool": "profile_refine_v311",
            "protocol_version": PROFILE_PROTOCOL_VERSION,
            "process_id": os.getpid(),
            "isolated_process": True,
            "measurement_order": "warmup -> latency/peak memory -> complexity",
            "latency_method": "synchronized wall clock at every declared stage",
            "latency_scope": (
                "preprocess + CA detector/P2-P3 capture + decode/rotated NMS + proposal packing + "
                "rotated-ROI Refine + geometry write-back; data loading reported separately"
            ),
            "data": args.data,
            "split": args.split,
            "test_used": False,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "amp": bundle.use_amp,
            "device": str(bundle.device),
            "workers": args.workers,
            "warmup_passes": args.warmup,
            "measured_images": len(rows),
            "input_shape_counts": input_shape_counts,
            "proposal_policy": "all post-NMS proposals",
            "rerun_nms": False,
            "confidence": bundle.extractor.conf,
            "nms_iou": bundle.extractor.nms_iou,
            "max_det": bundle.extractor.max_det,
            "ca_weights": str(bundle.ca_path),
            "ca_sha256": bundle.ca_hash,
            "refine_checkpoint": str(bundle.checkpoint_path),
            "refine_sha256": bundle.checkpoint_hash,
            "refine_epoch": bundle.checkpoint.get("epoch"),
            "weights_unchanged": weights_unchanged,
            "detector_parameters": detector_parameters,
            "refiner_parameters": refiner_parameters,
            "total_parameters": detector_parameters + refiner_parameters,
            "detector_gflops": detector_gflops,
            "refiner_profiled_gflops": refiner_gflops,
            "refiner_flop_profile_proposals": flop_proposals,
            "flop_scope_note": (
                "Refiner FLOPs are torch.profiler-supported operators on one real image with "
                f"{flop_proposals} post-NMS proposals; the value is proposal-count dependent, and grid_sample "
                "and NMS may not be assigned FLOPs, so latency is the authoritative complete-chain cost."
            ),
            "proposal_count": {
                "mean": statistics.fmean(float(row["proposal_count"]) for row in rows),
                "median": statistics.median(float(row["proposal_count"]) for row in rows),
                "p95": _percentile([float(row["proposal_count"]) for row in rows], 0.95),
                "max": max(int(row["proposal_count"]) for row in rows),
            },
            "timing_ms_per_image": timing,
            "coarse_fps_from_mean_compute": 1000.0 / mean_coarse,
            "refined_fps_from_mean_compute": 1000.0 / mean_total,
            "gpu_memory": {
                "baseline_allocated_gib": baseline_memory / 1024**3,
                "peak_allocated_gib": peak_memory / 1024**3,
                "incremental_peak_gib": max(peak_memory - baseline_memory, 0) / 1024**3,
                "scope": (
                    "GiB (2^30 bytes) of absolute allocated memory in this standalone process; "
                    "peak window excludes complexity profiling"
                ),
            },
        }
        write_csv(output_dir / "profile_per_image.csv", rows)
        write_json(output_dir / "profile_summary.json", summary)
        print(output_dir / "profile_summary.json")
    finally:
        bundle.close()


if __name__ == "__main__":
    main()
