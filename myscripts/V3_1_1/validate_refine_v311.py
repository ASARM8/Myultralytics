"""Locked independent validation entry point for Refine V3.1.1."""

from __future__ import annotations

import argparse

from myscripts.V3_1.validate_refine_v31 import CANONICAL_CA_WEIGHTS, main as validate_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate one frozen Refine V3.1.1 checkpoint on val.")
    parser.add_argument("--checkpoint", required=True)
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
    validate_main(
        [
            "--checkpoint",
            args.checkpoint,
            "--ca-weights",
            args.ca_weights,
            "--data",
            args.data,
            "--split",
            "val",
            "--imgsz",
            str(args.imgsz),
            "--batch",
            str(args.batch),
            "--device",
            args.device,
            "--workers",
            str(args.workers),
            "--amp" if args.amp else "--no-amp",
            "--expected-ca-map50-95",
            "0.45413",
            "--baseline-tolerance",
            "0.002",
            "--identity-tolerance",
            "0.0005",
            "--minimum-map-gain",
            "0.03",
            "--max-ap90-drop",
            "0.002",
            "--max-ap95-drop",
            "0.002",
            "--max-boundary-ratio",
            "0.10",
            "--output-dir",
            args.output_dir,
        ],
        required_architecture="OBBProposalRefinerV311",
    )


if __name__ == "__main__":
    main()
