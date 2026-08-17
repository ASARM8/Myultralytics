"""Export GT/Baseline/CA/conservative-Refine predictions under one protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from myscripts.V3_1_1 import export_qualitative_v311 as official
from myscripts.V3_1_1_low_gain.config import DEFAULT_RESIDUAL_SCALE
from myscripts.V3_1_1_low_gain.runtime import ResidualScaledRefiner, require_scale


def build_parser() -> argparse.ArgumentParser:
    parser = official.build_parser()
    parser.description = __doc__
    parser.add_argument("--residual-scale", type=float, default=DEFAULT_RESIDUAL_SCALE)
    return parser


def _base_argv(args: argparse.Namespace) -> list[str]:
    values = [
        "--baseline-weights", args.baseline_weights,
        "--ca-weights", args.ca_weights,
        "--checkpoint", args.checkpoint,
        "--data", args.data,
        "--split", args.split,
        "--imgsz", str(args.imgsz),
        "--batch", str(args.batch),
        "--device", str(args.device),
        "--workers", str(args.workers),
        "--amp" if args.amp else "--no-amp",
        "--max-images", str(args.max_images),
        "--copy-images" if args.copy_images else "--no-copy-images",
        "--output-dir", args.output_dir,
    ]
    if args.exist_ok:
        values.append("--exist-ok")
    return values


def _annotate_outputs(output_dir: Path, residual_scale: float) -> None:
    audit_path = output_dir / "export_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.update(
        {
            "tool": "export_qualitative_low_gain_v311",
            "residual_scale": residual_scale,
            "weights_modified": False,
        }
    )
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest_path = output_dir / "manifest_all.csv"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["residual_scale"] = residual_scale
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    official.validate_args(parser, args)
    require_scale(parser, args.residual_scale)

    original_loader = official.load_refine_bundle

    def scaled_loader(*loader_args, **loader_kwargs):
        bundle = original_loader(*loader_args, **loader_kwargs)
        bundle.refiner = ResidualScaledRefiner(bundle.refiner, args.residual_scale)
        return bundle

    official.load_refine_bundle = scaled_loader
    try:
        official.main(_base_argv(args))
    finally:
        official.load_refine_bundle = original_loader
    _annotate_outputs(Path(args.output_dir), args.residual_scale)


if __name__ == "__main__":
    main()
