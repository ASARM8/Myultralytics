"""Validate a frozen Refine V3.1.1 checkpoint with damped geometry residuals.

This entry point is deliberately isolated from the formal V3.1.1 validation
chain. It changes no weights, proposal identities, scores, classes, or NMS
results. Only the predicted geometry residual is multiplied by a coefficient
before the existing box update is applied.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from myscripts.V3_1_1.evidence_runtime import CANONICAL_CA_WEIGHTS, load_refine_bundle, require_canonical_path
from myscripts.V3_1_1_low_gain.config import DEFAULT_RESIDUAL_SCALE, TARGET_MAP_GAIN
from myscripts.V3_1_1_low_gain.runtime import ResidualScaledRefiner, require_scale


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
        "--residual-scale",
        type=float,
        default=DEFAULT_RESIDUAL_SCALE,
        help="Geometry-residual multiplier in [0, 1]; default is the frozen conservative setting.",
    )
    parser.add_argument("--target-gain", type=float, default=TARGET_MAP_GAIN)
    parser.add_argument("--output-dir", required=True)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    require_canonical_path(parser, args.ca_weights, CANONICAL_CA_WEIGHTS, "CA checkpoint")
    if args.batch <= 0 or args.workers < 0:
        parser.error("batch must be positive and workers must be non-negative")
    require_scale(parser, args.residual_scale)
    if not 0.0 < args.target_gain < 1.0:
        parser.error("target-gain must be within (0, 1)")


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    """Run one low-gain validation and return its audit dictionary."""
    from myscripts.V3.runtime import build_dataset, full_loader
    from myscripts.V3.train_refine_v3 import write_csv, write_json
    from myscripts.V3_1.runtime import evaluate_refiner_v31

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_refine_bundle(
        args.checkpoint,
        args.ca_weights,
        device_arg=args.device,
        amp=args.amp,
        imgsz=args.imgsz,
    )
    try:
        dataset, data = build_dataset(args.data, "val", args.imgsz, args.batch, args.workers, rect=True)
        loader = full_loader(dataset, args.batch, args.workers)
        names = getattr(bundle.ca_model, "names", data["names"])
        refiner = ResidualScaledRefiner(bundle.refiner, args.residual_scale)
        rows, diagnostics = evaluate_refiner_v31(
            bundle.extractor,
            refiner,
            loader,
            names,
            amp=bundle.use_amp,
            match_iou=float(bundle.training_args.get("match_iou", 0.30)),
            quality_min_gain=float(bundle.training_args.get("quality_min_gain", 0.002)),
            tiny_reference_px=float(bundle.training_args.get("tiny_reference_px", 8.0)),
            tiny_weight_floor=float(bundle.training_args.get("tiny_weight_floor", 0.25)),
            identity_tolerance=5e-4,
        )
        lookup = {row["variant"]: row for row in rows}
        coarse = lookup["coarse"]
        refined = lookup["refined"]
        delta_map = float(refined["map50_95"] - coarse["map50_95"])
        for row in rows:
            row["residual_scale"] = args.residual_scale
            row["target_gain"] = args.target_gain
        diagnostics.update(
            {
                "residual_scale": args.residual_scale,
                "raw_checkpoint_residual_scale": 1.0,
                "weights_modified": False,
            }
        )
        audit = {
            "experiment": "V3.1.1 isolated low-gain residual scaling",
            "checkpoint": str(bundle.checkpoint_path),
            "checkpoint_sha256": bundle.checkpoint_hash,
            "ca_weights": str(bundle.ca_path),
            "ca_sha256": bundle.ca_hash,
            "architecture": bundle.checkpoint.get("architecture"),
            "data": args.data,
            "split": "val",
            "imgsz": args.imgsz,
            "batch": args.batch,
            "amp": bundle.use_amp,
            "residual_scale": args.residual_scale,
            "target_gain": args.target_gain,
            "coarse_map50_95": coarse["map50_95"],
            "refined_map50_95": refined["map50_95"],
            "delta_map50_95": delta_map,
            "target_abs_error": abs(delta_map - args.target_gain),
            "delta_ap75": refined["ap75"] - coarse["ap75"],
            "delta_ap90": refined["ap90"] - coarse["ap90"],
            "delta_ap95": refined["ap95"] - coarse["ap95"],
            "identity_max_abs_metric_delta": diagnostics["identity_max_abs_metric_delta"],
            "matched_delta_iou_mean": diagnostics["matched_delta_iou_mean"],
            "matched_improved_ratio": diagnostics["matched_improved_ratio"],
            "matched_worsened_ratio": diagnostics["matched_worsened_ratio"],
            "proposal_policy": "all",
            "scores_changed": False,
            "classes_changed": False,
            "rerun_nms": False,
            "weights_modified": False,
            "test_used": False,
            "paper_primary_result": False,
        }
        write_csv(output_dir / "val_metrics.csv", rows)
        write_json(output_dir / "val_diagnostics.json", diagnostics)
        write_json(output_dir / "low_gain_audit.json", audit)
        print(output_dir / "low_gain_audit.json")
        return audit
    finally:
        bundle.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    run_validation(args)


if __name__ == "__main__":
    main()
