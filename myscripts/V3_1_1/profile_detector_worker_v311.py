"""Profile one frozen OBB detector in an isolated process.

This is an implementation worker for :mod:`profile_comparative_v311`.  A
separate process is required for each method so CUDA's absolute peak allocated
memory contains only that method's model, buffers, and activations.  The worker
is still directly runnable for diagnosis, but the paper protocol should invoke
the comparative parent, which balances process order across three repeats.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from myscripts.V3.train_refine_v3 import write_csv, write_json
from myscripts.V3_1_1.evidence_runtime import (
    CANONICAL_BASELINE_WEIGHTS,
    CANONICAL_CA_WEIGHTS,
    ensure_omp_threads,
    require_canonical_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("Baseline", "CA"))
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val", choices=("val",))
    parser.add_argument("--imgsz", type=int, default=640, choices=(640,))
    parser.add_argument("--batch", type=int, default=1, choices=(1,))
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--conf", type=float, required=True)
    parser.add_argument("--nms-iou", type=float, required=True)
    parser.add_argument("--max-det", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    canonical = CANONICAL_BASELINE_WEIGHTS if args.method == "Baseline" else CANONICAL_CA_WEIGHTS
    require_canonical_path(parser, args.weights, canonical, f"{args.method} checkpoint")
    if args.workers < 0 or args.warmup < 1 or args.max_images < 0:
        parser.error("workers/max-images must be non-negative and warmup must be positive")
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.nms_iou <= 1.0:
        parser.error("--conf and --nms-iou must be in [0, 1]")
    if args.max_det < 1:
        parser.error("--max-det must be positive")


def main(argv: list[str] | None = None) -> None:
    ensure_omp_threads()
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    import torch

    from ultralytics.utils.torch_utils import select_device

    from myscripts.V3.runtime import build_dataset
    from myscripts.V3_1_1.profile_comparative_v311 import _profile_detector

    device = select_device(args.device)
    dataset, _data = build_dataset(args.data, args.split, args.imgsz, args.batch, args.workers, rect=True)
    summary, rows = _profile_detector(
        label=args.method,
        weights=args.weights,
        expected_reg_max=16 if args.method == "Baseline" else 32,
        torch=torch,
        device=device,
        dataset=dataset,
        batch=args.batch,
        workers=args.workers,
        warmup=args.warmup,
        max_images=args.max_images,
        conf=args.conf,
        nms_iou=args.nms_iou,
        max_det=args.max_det,
        imgsz=args.imgsz,
    )
    summary.update(
        tool="profile_detector_worker_v311",
        protocol_version=2,
        process_id=os.getpid(),
        isolated_process=True,
        measurement_order="warmup -> latency/peak memory -> complexity",
        data=args.data,
        split=args.split,
        test_used=False,
        imgsz=args.imgsz,
        batch=args.batch,
        amp=False,
        device=str(device),
        workers=args.workers,
        warmup_passes=args.warmup,
        confidence=args.conf,
        nms_iou=args.nms_iou,
        max_det=args.max_det,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "profile_per_image.csv", rows)
    write_json(output_dir / "profile_summary.json", summary)
    print(output_dir / "profile_summary.json")


if __name__ == "__main__":
    main()
