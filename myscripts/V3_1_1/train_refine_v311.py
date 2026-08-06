"""Locked Refine V3.1.1 training entry point."""

from __future__ import annotations

import argparse

from myscripts.V3_1.train_refine_v31 import CANONICAL_CA_WEIGHTS, main as train_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train locked Refine V3.1.1: geometry-only, smooth 80% targets, 15 epochs."
    )
    parser.add_argument("--ca-weights", default=CANONICAL_CA_WEIGHTS)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640, choices=(640,))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_main(
        [
            "--refiner-version",
            "v311",
            "--experiment",
            "geometry_only",
            "--target-margin",
            "0.80",
            "--epochs",
            "15",
            "--eval-interval",
            "1",
            "--max-ap95-drop",
            "0.002",
            "--ca-weights",
            args.ca_weights,
            "--data",
            args.data,
            "--imgsz",
            str(args.imgsz),
            "--batch",
            str(args.batch),
            "--device",
            args.device,
            "--workers",
            str(args.workers),
            "--amp" if args.amp else "--no-amp",
            "--output-dir",
            args.output_dir,
        ]
    )


if __name__ == "__main__":
    main()
