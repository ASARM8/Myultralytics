"""Sweep conservative residual strengths without changing the frozen model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from myscripts.V3_1_1.evidence_runtime import CANONICAL_CA_WEIGHTS
from myscripts.V3_1_1_low_gain.config import TARGET_MAP_GAIN
from myscripts.V3_1_1_low_gain.validate_low_gain_v311 import run_validation, validate_args


def parse_scales(value: str) -> tuple[float, ...]:
    """Parse a sorted, unique coefficient grid."""
    try:
        scales = tuple(sorted({round(float(item.strip()), 6) for item in value.split(",") if item.strip()}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("scales must be comma-separated numbers") from error
    if not scales or any(not 0.0 <= scale <= 1.0 for scale in scales):
        raise argparse.ArgumentTypeError("every residual scale must be within [0, 1]")
    return scales


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ca-weights", default=CANONICAL_CA_WEIGHTS)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640, choices=(640,))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--scales",
        type=parse_scales,
        default=parse_scales("0,0.20,0.28,0.32,0.35,0.37,0.39,0.42,0.45,0.50,0.75,1.0"),
    )
    parser.add_argument("--target-gain", type=float, default=TARGET_MAP_GAIN)
    parser.add_argument("--output-dir", required=True)
    return parser


def _scale_dir(scale: float) -> str:
    return f"scale_{scale:.3f}".replace(".", "p")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    probe_args = argparse.Namespace(**vars(args), residual_scale=0.0)
    validate_args(parser, probe_args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scale in args.scales:
        run_args = argparse.Namespace(
            checkpoint=args.checkpoint,
            ca_weights=args.ca_weights,
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            amp=args.amp,
            residual_scale=scale,
            target_gain=args.target_gain,
            output_dir=str(output_dir / _scale_dir(scale)),
        )
        audit = run_validation(run_args)
        rows.append(
            {
                "residual_scale": scale,
                "coarse_map50_95": audit["coarse_map50_95"],
                "refined_map50_95": audit["refined_map50_95"],
                "delta_map50_95": audit["delta_map50_95"],
                "target_abs_error": audit["target_abs_error"],
                "delta_ap75": audit["delta_ap75"],
                "delta_ap90": audit["delta_ap90"],
                "delta_ap95": audit["delta_ap95"],
                "matched_delta_iou_mean": audit["matched_delta_iou_mean"],
                "matched_improved_ratio": audit["matched_improved_ratio"],
                "matched_worsened_ratio": audit["matched_worsened_ratio"],
                "output_dir": run_args.output_dir,
            }
        )

    selected = min(rows, key=lambda row: (row["target_abs_error"], row["residual_scale"]))
    csv_path = output_dir / "residual_scale_sweep.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "purpose": "isolated low-gain sensitivity experiment",
        "target_gain": args.target_gain,
        "closest_observed": selected,
        "weights_modified": False,
        "paper_primary_result": False,
        "warning": (
            "The closest row is diagnostic. A scale may become a primary configuration only after an independent "
            "robustness or deployment selection rule is declared and the full scale sweep is reported."
        ),
    }
    summary_path = output_dir / "residual_scale_selection.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(csv_path)
    print(summary_path)


if __name__ == "__main__":
    main()
