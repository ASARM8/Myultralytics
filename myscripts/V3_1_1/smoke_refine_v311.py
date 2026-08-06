"""Locked one-batch smoke test for Refine V3.1.1."""

from __future__ import annotations

import argparse

from myscripts.V3_1.smoke_refine_v31 import CANONICAL_CA_WEIGHTS, main as smoke_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test one locked Refine V3.1.1 training batch.")
    parser.add_argument("--ca-weights", default=CANONICAL_CA_WEIGHTS)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640, choices=(640,))
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    smoke_main(
        [
            "--refiner-version",
            "v311",
            "--experiment",
            "geometry_only",
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
            "--output",
            args.output,
        ]
    )


if __name__ == "__main__":
    main()
