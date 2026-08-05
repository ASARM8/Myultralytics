"""Reproduce one frozen Refine V3.1 checkpoint without retraining or reselection."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from myscripts.V3.train_refine_v3 import write_csv, write_json


CANONICAL_CA_WEIGHTS = "/root/autodl-tmp/work-dirs/yolo11_obb_640_811_ca/weights/best.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a frozen all-proposal Refine V3.1 checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ca-weights", default=CANONICAL_CA_WEIGHTS)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val", choices=("val",))
    parser.add_argument("--imgsz", type=int, default=640, choices=(640,))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected-ca-map50-95", type=float, default=0.45413)
    parser.add_argument("--baseline-tolerance", type=float, default=0.002)
    parser.add_argument("--identity-tolerance", type=float, default=5e-4)
    parser.add_argument("--minimum-map-gain", type=float, default=0.03)
    parser.add_argument("--max-ap90-drop", type=float, default=0.002)
    parser.add_argument("--output-dir", required=True)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if Path(args.ca_weights).as_posix() != Path(CANONICAL_CA_WEIGHTS).as_posix():
        parser.error(f"V3.1 validation is locked to the canonical CA checkpoint: {CANONICAL_CA_WEIGHTS}")
    if args.batch <= 0 or args.workers < 0:
        parser.error("batch must be positive and workers must be non-negative")
    if min(args.baseline_tolerance, args.identity_tolerance, args.minimum_map_gain, args.max_ap90_drop) <= 0:
        parser.error("tolerances and minimum gain must be positive")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    if not os.environ.get("OMP_NUM_THREADS", "").isdigit() or int(os.environ.get("OMP_NUM_THREADS", "0")) <= 0:
        os.environ["OMP_NUM_THREADS"] = "1"

    import torch

    from ultralytics import YOLO
    from ultralytics.nn.modules.refine_v31 import OBBProposalRefinerV31
    from ultralytics.utils.torch_utils import select_device

    from .runtime import FrozenCAExtractor, build_dataset, evaluate_refiner_v31, full_loader, sha256_file

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint)
    ca_path = Path(args.ca_weights)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"V3.1 checkpoint not found: {checkpoint_path}")
    if not ca_path.is_file():
        raise FileNotFoundError(f"canonical CA checkpoint not found: {ca_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1 or checkpoint.get("architecture") != "OBBProposalRefinerV31":
        raise RuntimeError("unsupported or non-V3.1 checkpoint format")
    ca_hash = sha256_file(ca_path)
    if checkpoint.get("ca_sha256") != ca_hash:
        raise RuntimeError("CA checkpoint hash mismatch; V3.1 features/proposals no longer match training")

    device = select_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    yolo = YOLO(str(ca_path), task="obb")
    ca_model = yolo.model.to(device).float().eval()
    head = ca_model.model[-1]
    if type(head).__name__ != "OBB" or int(getattr(head, "reg_max", -1)) != 32:
        raise RuntimeError("V3.1 requires the pure CA OBB head with reg_max=32")
    for parameter in ca_model.parameters():
        parameter.requires_grad_(False)

    training_args = checkpoint.get("arguments", {})
    dataset, data = build_dataset(args.data, args.split, args.imgsz, args.batch, args.workers, rect=True)
    loader = full_loader(dataset, args.batch, args.workers)
    names = getattr(ca_model, "names", data["names"])
    extractor = FrozenCAExtractor(
        ca_model,
        device=device,
        nc=len(names),
        conf=float(training_args.get("proposal_conf", 0.01)),
        nms_iou=float(training_args.get("nms_iou", 0.70)),
        max_det=int(training_args.get("max_det", 300)),
        amp=use_amp,
    )
    try:
        observed_channels = extractor.infer_channels(args.imgsz)
        config = checkpoint["model_config"]
        expected_channels = int(config["p2_channels"]), int(config["p3_channels"])
        if observed_channels != expected_channels:
            raise RuntimeError(f"CA feature-channel mismatch: expected {expected_channels}, got {observed_channels}")
        refiner = OBBProposalRefinerV31(**config).to(device).float().eval()
        refiner.load_state_dict(checkpoint["model_state"], strict=True)
        rows, diagnostics = evaluate_refiner_v31(
            extractor,
            refiner,
            loader,
            names,
            amp=use_amp,
            match_iou=float(training_args.get("match_iou", 0.30)),
            quality_min_gain=float(training_args.get("quality_min_gain", 0.002)),
            tiny_reference_px=float(training_args.get("tiny_reference_px", 8.0)),
            tiny_weight_floor=float(training_args.get("tiny_weight_floor", 0.25)),
            identity_tolerance=args.identity_tolerance,
        )
        lookup = {row["variant"]: row for row in rows}
        coarse, refined = lookup["coarse"], lookup["refined"]
        delta_map = refined["map50_95"] - coarse["map50_95"]
        delta_ap75 = refined["ap75"] - coarse["ap75"]
        delta_ap90 = refined["ap90"] - coarse["ap90"]
        audit = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "experiment": checkpoint.get("experiment"),
            "ca_weights": str(ca_path),
            "ca_sha256": ca_hash,
            "data": args.data,
            "split": args.split,
            "imgsz": args.imgsz,
            "amp": use_amp,
            "batch": args.batch,
            "baseline_abs_error": abs(coarse["map50_95"] - args.expected_ca_map50_95),
            "baseline_pass": abs(coarse["map50_95"] - args.expected_ca_map50_95) <= args.baseline_tolerance,
            "identity_pass": diagnostics["identity_max_abs_metric_delta"] <= args.identity_tolerance,
            "delta_map50_95": delta_map,
            "delta_ap75": delta_ap75,
            "delta_ap90": delta_ap90,
            "minimum_gain_pass": delta_map >= args.minimum_map_gain,
            "ap75_pass": delta_ap75 >= 0,
            "ap90_pass": delta_ap90 >= -args.max_ap90_drop,
            "matched_delta_iou_mean": diagnostics["matched_delta_iou_mean"],
            "matched_improved_ratio": diagnostics["matched_improved_ratio"],
            "matched_worsened_ratio": diagnostics["matched_worsened_ratio"],
            "proposal_policy": "all",
            "quality_used_at_inference": False,
            "rerun_nms": False,
            "test_used": False,
        }
        write_csv(output_dir / "val_metrics.csv", rows)
        write_json(output_dir / "val_diagnostics.json", diagnostics)
        write_json(output_dir / "validation_audit.json", audit)
        if not audit["baseline_pass"]:
            raise RuntimeError(
                f"CA baseline mismatch: observed={coarse['map50_95']:.6f}, "
                f"expected={args.expected_ca_map50_95:.6f}"
            )
        print(output_dir / "validation_audit.json")
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
