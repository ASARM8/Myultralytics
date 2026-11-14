"""Measure the online CA -> Refine -> wire-mask pipeline without training.

Run with ``python -m myscripts.wire_seg.profile_pipeline --help``. End-to-end
timing and synchronized stage attribution use separate passes. No proposal
cache is used for inference; an optional cache only checks output consistency
outside the timed region. Results describe serial image-tile processing, not
camera-to-display latency or full-resolution aerial-frame throughput.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import tarfile
from collections import Counter
from pathlib import Path
from time import perf_counter


STAGES = ("image_read_ms", "detector_input_ms", "detector_and_nms_ms", "refine_ms",
          "geometry_transfer_ms", "crop_prompt_ms", "mask_inference_ms", "mask_restore_ms")


def describe(values):
    values = sorted(map(float, values))
    if not values:
        return {"count": 0, **dict.fromkeys(("mean", "median", "p95", "p99", "min", "max", "std"))}

    def percentile(p):
        pos = (len(values) - 1) * p
        low = math.floor(pos)
        return values[low] + (values[math.ceil(pos)] - values[low]) * (pos - low)

    return {"count": len(values), "mean": statistics.fmean(values), "median": percentile(.5),
            "p95": percentile(.95), "p99": percentile(.99), "min": values[0], "max": values[-1],
            "std": statistics.pstdev(values)}


def summarize(rows, repeat_peaks, tolerance):
    """Keep synchronized attribution out of the ordinary latency population."""
    end = [r for r in rows if r["phase"] == "end_to_end"]
    stages = [r for r in rows if r["phase"] == "stages"]
    repeats = sorted({r["repeat"] for r in end})
    means = [statistics.fmean(r["total_ms"] for r in end if r["repeat"] == repeat) for repeat in repeats]
    latency = describe(r["total_ms"] for r in end)
    spread = (max(means) - min(means)) / statistics.fmean(means) if means else None
    bins = []
    for low, high in ((0, 0), (1, 1), (2, 4), (5, 9), (10, 19), (20, None)):
        selected = [r for r in end if r["proposals"] >= low and (high is None or r["proposals"] <= high)]
        bins.append({"min_proposals": low, "max_proposals": high,
                     "unique_tiles": len({r["tile"] for r in selected}),
                     "latency_ms": describe(r["total_ms"] for r in selected),
                     "crop_pixels": describe(r["crop_pixels"] for r in selected)})
    return {"end_to_end_ms": latency, "serial_tiles_per_second": 1000 / latency["mean"] if end else None,
            "repeat_means_ms": means, "repeat_mean_relative_spread": spread,
            "stability_tolerance": tolerance,
            "stability_pass": len(means) >= 3 and spread <= tolerance,
            "stage_ms_separate_pass": {key: describe(r[key] for r in stages) for key in STAGES},
            "stage_instrumented_total_ms": describe(r["total_ms"] for r in stages),
            "by_proposal_count": bins, "gpu_memory_by_pass": repeat_peaks}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_frame(actual, expected, atol=1e-4):
    """Compare counts, order and geometry; report drift rather than retune inputs."""
    import numpy as np

    errors = {}
    for key in ("coarse", "refined", "scores"):
        a, b = np.asarray(actual[key]), np.asarray(expected[key])
        if a.shape != b.shape:
            errors[key] = {"shape_match": False, "max_abs_error": None, "matches": False}
        else:
            errors[key] = {"shape_match": True, "max_abs_error": float(np.max(np.abs(a - b))) if a.size else 0.,
                           "matches": bool(np.allclose(a, b, atol=atol, rtol=1e-5))}
    return {"matches": actual["size"] == expected["size"] and all(e["matches"] for e in errors.values()),
            "fields": errors}


def run_one(path, bundle, checkpoint, model, wire_class, threshold, *, stage_timing=False):
    import numpy as np
    from PIL import Image
    from .data import example
    from .learning import crop_probability
    from .proposals import infer_frame

    torch, device = bundle.torch, bundle.device
    timings = dict.fromkeys(STAGES, 0.)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def measure(key, operation):
        if not stage_timing:
            return operation()
        sync()
        start = perf_counter()
        value = operation()
        sync()
        timings[key] += (perf_counter() - start) * 1000
        return value

    def read_image():
        with Image.open(path) as source:
            return source.convert("RGB")

    sync()
    start = perf_counter()
    image = measure("image_read_ms", read_image)
    frame, detector_shape = infer_frame(bundle, image, wire_class, measure)
    masks, crop_sizes = [], []
    for i in range(len(frame["refined"])):
        crop, feature = measure("crop_prompt_ms", lambda: example(
            image, frame["refined"], i, checkpoint["prompt"], checkpoint["margin"]))
        probability = measure("mask_inference_ms", lambda: crop_probability(model, feature, device))
        restored = measure("mask_restore_ms", lambda: np.asarray(crop.restore(
            Image.fromarray(probability >= threshold), image.size)))
        masks.append(restored)
        crop_sizes.append(crop.size)
    sync()
    total_ms = (perf_counter() - start) * 1000
    return {"tile": path.name, "width": image.width, "height": image.height,
            "detector_height": detector_shape[0], "detector_width": detector_shape[1],
            "proposals": len(masks), "crop_pixels": sum(w * h for w, h in crop_sizes),
            "max_crop_pixels": max((w * h for w, h in crop_sizes), default=0),
            "total_ms": total_ms, **timings}, frame, masks


def build_parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--images", required=True)
    cli.add_argument("--ca", required=True)
    cli.add_argument("--refine", required=True)
    cli.add_argument("--weights", required=True, help="Frozen mask.pt")
    cli.add_argument("--cache", help="Optional comparison only; never substitutes for live detection")
    cli.add_argument("--output-dir", required=True, help="New directory; the same name plus .tar.gz is also created")
    cli.add_argument("--device", default="0")
    cli.add_argument("--wire-class", type=int)
    cli.add_argument("--threshold", type=float, default=.5)
    cli.add_argument("--warmup", type=int, default=50, help="Untimed full-pipeline calls before each pass")
    cli.add_argument("--repeats", type=int, default=3, help="Natural end-to-end passes")
    cli.add_argument("--stage-repeats", type=int, default=1, help="Separate synchronized attribution passes")
    cli.add_argument("--limit", type=int, default=0, help="0: all tiles; otherwise evenly spaced tiles")
    cli.add_argument("--cpu-threads", type=int, default=1)
    cli.add_argument("--stability-tolerance", type=float, default=.05)
    return cli


def run_profile(args):
    from myscripts.V3_1_1.evidence_runtime import ensure_omp_threads, load_refine_bundle
    ensure_omp_threads()
    import cv2
    import numpy as np
    import torch
    from .learning import TileProgress, load_mask_head
    from .proposals import image_paths, save_json
    from myscripts.V3_1_1_low_gain.config import DEFAULT_RESIDUAL_SCALE

    torch.set_num_threads(args.cpu_threads)
    cv2.setNumThreads(args.cpu_threads)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    all_paths = image_paths(args.images)
    paths = all_paths
    if args.limit:
        paths = [all_paths[i] for i in np.linspace(0, len(all_paths) - 1, min(args.limit, len(all_paths)), dtype=int)]
    cache = json.loads(Path(args.cache).read_text(encoding="utf-8")) if args.cache else None
    if cache and any(p.name not in cache["frames"] for p in paths):
        raise ValueError("Comparison cache is missing selected image tiles")
    print(f"[profile] loading live detector, refiner and mask head | {len(paths)} tiles", flush=True)
    bundle = load_refine_bundle(args.refine, args.ca, device_arg=args.device, amp=False, imgsz=640)
    try:
        checkpoint, model = load_mask_head(args.weights, bundle.device)
        if checkpoint["boxes"] != "refined":
            raise ValueError("This profile measures the refined-box mask pipeline")
        wire_class = args.wire_class
        if wire_class is None:
            wire_class = checkpoint["detector"].get("wire_class")
        if wire_class is None:
            if len(bundle.ca_model.names) != 1:
                raise ValueError("Specify --wire-class for this multi-class detector")
            wire_class = 0
        if not 0 <= wire_class < len(bundle.ca_model.names):
            raise ValueError(f"Invalid wire class: {wire_class}")
        detector = {"version": 1, "ca_sha256": bundle.ca_hash, "refine_sha256": bundle.checkpoint_hash,
                    "scale": DEFAULT_RESIDUAL_SCALE, "wire_class": wire_class}
        cuda = bundle.device.type == "cuda"
        properties = torch.cuda.get_device_properties(bundle.device) if cuda else None
        try:
            revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
            commit = revision.stdout.strip() if revision.returncode == 0 else None
        except OSError:
            commit = None  # A copied server checkout need not have Git installed.
        manifest = {
            "protocol_version": 1, "arguments": vars(args), "prompt": checkpoint["prompt"],
            "margin_px": checkpoint["margin"], "precision": "FP32", "image_batch": 1, "mask_batch": 1,
            "tiles": len(paths), "available_tiles": len(all_paths), "image_order": [p.name for p in paths],
            "detector": detector, "mask_weights_sha256": sha256(args.weights),
            "mask_training_detector_metadata_matches": checkpoint["detector"] == detector,
            "cache_sha256": sha256(args.cache) if cache else None,
            "git_commit": commit,
            "implementation_sha256": {name: sha256(Path(__file__).with_name(name))
                                      for name in ("profile_pipeline.py", "proposals.py", "learning.py", "geometry.py", "data.py")},
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "torch": torch.__version__, "cuda": torch.version.cuda,
                            "cudnn": torch.backends.cudnn.version(), "device": str(bundle.device),
                            "device_name": properties.name if properties else platform.processor(),
                            "total_device_memory_bytes": properties.total_memory if properties else None,
                            "cpu_threads": torch.get_num_threads(), "opencv_threads": cv2.getNumThreads(),
                            "cudnn_benchmark": torch.backends.cudnn.benchmark,
                            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32},
            "scope": {
                "included": "image file read/decode, detector preprocessing/forward/NMS, Refine, geometry transfer, crops/prompts, mask forward/CPU transfer, binary mask restoration",
                "excluded": "model loading, warmup, labels/AP, consistency checks, progress printing, report/mask file writing, camera/display",
                "end_to_end": "wall clock with synchronization only at image boundaries; serial execution",
                "serial_tiles_per_second": "1000 / mean measured end-to-end milliseconds; excludes collection gaps, not sustained camera throughput",
                "stages": "separate passes with synchronization at stage boundaries; not used for main FPS",
                "image_read": "filesystem reads with warmup and repeated access; not a cold-disk throughput measurement",
                "memory": "PyTorch allocated and reserved GPU memory with all three models resident; not total board RAM",
                "power": "not measured", "candidate_policy": "all post-NMS wire candidates, no additional truncation",
            },
        }
        save_json(output / "manifest.json", manifest)
        rows, peaks, reference, mismatches = [], [], {}, []
        checks = {"cache_tiles_checked": 0, "cache_mismatched_tiles": 0,
                  "repeat_output_comparisons": 0, "repeat_output_mismatches": 0,
                  "stage_output_comparisons": 0, "stage_output_mismatches": 0}
        passes = [("end_to_end", i + 1) for i in range(args.repeats)]
        passes += [("stages", i + 1) for i in range(args.stage_repeats)]
        warm_paths = [paths[i] for i in np.linspace(0, len(paths) - 1, min(args.warmup, len(paths)), dtype=int)]
        fields = ["phase", "repeat", "tile", "width", "height", "detector_height", "detector_width",
                  "proposals", "crop_pixels", "max_crop_pixels", "total_ms", *STAGES]
        with (output / "per_image.csv").open("w", encoding="utf-8-sig", newline="") as handle, torch.inference_mode():
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for phase, repeat in passes:
                print(f"[profile {phase} {repeat}] warming up {args.warmup} images", flush=True)
                for i in range(args.warmup):
                    run_one(warm_paths[i % len(warm_paths)], bundle, checkpoint, model, wire_class, args.threshold)
                if cuda:
                    torch.cuda.synchronize(bundle.device)
                    torch.cuda.reset_peak_memory_stats(bundle.device)
                print(f"[profile {phase} {repeat}] starting {len(paths)} tiles", flush=True)
                progress = TileProgress(len(paths))
                for index, path in enumerate(paths):
                    row, frame, masks = run_one(path, bundle, checkpoint, model, wire_class, args.threshold,
                                                stage_timing=phase == "stages")
                    row.update(phase=phase, repeat=repeat)
                    # Checks and serialization are deliberately outside the timed region.
                    digest = hashlib.sha256(json.dumps(frame, sort_keys=True).encode())
                    for mask in masks:
                        digest.update(np.packbits(mask).tobytes())
                    fingerprint = digest.hexdigest()
                    del masks
                    if phase == "end_to_end" and repeat == 1:
                        reference[path.name] = fingerprint
                        if cache:
                            comparison = compare_frame(frame, cache["frames"][path.name])
                            checks["cache_tiles_checked"] += 1
                            if not comparison["matches"]:
                                checks["cache_mismatched_tiles"] += 1
                                mismatches.append({"tile": path.name, **comparison})
                    else:
                        prefix = "stage" if phase == "stages" else "repeat"
                        checks[f"{prefix}_output_comparisons"] += 1
                        checks[f"{prefix}_output_mismatches"] += int(reference[path.name] != fingerprint)
                    rows.append(row)
                    writer.writerow(row)
                    progress.update(index + 1, f"[profile {phase} {repeat}]",
                                    f"| last {row['total_ms']:.2f} ms | proposals {row['proposals']}")
                handle.flush()
                peak = {"phase": phase, "repeat": repeat,
                        "peak_allocated_gib": torch.cuda.max_memory_allocated(bundle.device) / 2**30 if cuda else None,
                        "peak_reserved_gib": torch.cuda.max_memory_reserved(bundle.device) / 2**30 if cuda else None}
                peaks.append(peak)
                save_json(output / "partial_summary.json", {**summarize(rows, peaks, args.stability_tolerance),
                                                           "completed_passes": len(peaks), "planned_passes": len(passes)})
        report = {**summarize(rows, peaks, args.stability_tolerance), "consistency": checks,
                  "input_shape_counts_first_pass": dict(Counter(
                      f"{r['detector_height']}x{r['detector_width']}" for r in rows
                      if r["phase"] == "end_to_end" and r["repeat"] == 1)),
                  "cache_mismatches": mismatches, "manifest": "manifest.json", "per_image": "per_image.csv"}
        save_json(output / "summary.json", report)
        archive = Path(str(output) + ".tar.gz")
        with tarfile.open(archive, "w:gz") as pack:
            pack.add(output, arcname=output.name)
        print(f"[profile] mean {report['end_to_end_ms']['mean']:.2f} ms | "
              f"P95 {report['end_to_end_ms']['p95']:.2f} ms | "
              f"serial tiles/s {report['serial_tiles_per_second']:.2f} | "
              f"stability_pass={report['stability_pass']}", flush=True)
        print(f"[profile] consistency {checks}", flush=True)
        print(f"[profile] download: {archive}", flush=True)
    finally:
        bundle.close()


def main(argv=None):
    cli = build_parser()
    args = cli.parse_args(argv)
    if min(args.warmup, args.repeats, args.cpu_threads) < 1 or min(args.stage_repeats, args.limit) < 0:
        cli.error("warmup/repeats/cpu-threads must be positive; stage-repeats/limit must be nonnegative")
    if not 0 < args.threshold < 1 or not 0 < args.stability_tolerance < 1:
        cli.error("threshold and stability-tolerance must be in (0, 1)")
    run_profile(args)


if __name__ == "__main__":
    main()
