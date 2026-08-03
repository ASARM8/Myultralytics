"""Reproduce a frozen Refine V3 checkpoint on val without retraining or reselection."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


CANONICAL_CA_WEIGHTS = "/root/autodl-tmp/work-dirs/yolo11_obb_640_811_ca/weights/best.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate one frozen Refine V3 checkpoint on split=val.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ca-weights", default=CANONICAL_CA_WEIGHTS)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="val", choices=("val",))
    parser.add_argument("--imgsz", type=int, default=640, choices=(640,))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--quality-threshold", type=float, default=None)
    parser.add_argument("--expected-ca-map50-95", type=float, default=0.45413)
    parser.add_argument("--baseline-tolerance", type=float, default=0.002)
    parser.add_argument("--identity-tolerance", type=float, default=5e-4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if Path(args.ca_weights).as_posix() != Path(CANONICAL_CA_WEIGHTS).as_posix():
        parser.error(f"V3 validation is locked to the canonical CA checkpoint: {CANONICAL_CA_WEIGHTS}")
    if args.batch <= 0 or args.workers < 0:
        parser.error("batch must be positive and workers must be non-negative")
    if args.quality_threshold is not None and not 0.0 < args.quality_threshold < 1.0:
        parser.error("--quality-threshold must be in (0, 1)")
    if not 0.0 <= args.expected_ca_map50_95 <= 1.0:
        parser.error("--expected-ca-map50-95 must be in [0, 1]")
    if args.baseline_tolerance <= 0.0 or args.identity_tolerance <= 0.0:
        parser.error("baseline and identity tolerances must be positive")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    omp_threads = os.environ.get("OMP_NUM_THREADS", "")
    if not omp_threads.isdigit() or int(omp_threads) <= 0:
        os.environ["OMP_NUM_THREADS"] = "1"

    import torch

    from ultralytics import YOLO
    from ultralytics.nn.modules.refine_v3 import OBBProposalRefinerV3
    from ultralytics.utils.torch_utils import select_device

    from .runtime import FrozenCAExtractor, build_dataset, evaluate_refiner, full_loader, sha256_file

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint)
    ca_path = Path(args.ca_weights)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"V3 checkpoint not found: {checkpoint_path}")
    if not ca_path.is_file():
        raise FileNotFoundError(f"canonical CA checkpoint not found: {ca_path}")
    device = select_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1 or checkpoint.get("architecture") != "OBBProposalRefinerV3":
        raise RuntimeError("unsupported or non-V3 checkpoint format")
    ca_hash = sha256_file(ca_path)
    if checkpoint.get("ca_sha256") != ca_hash:
        raise RuntimeError(
            "CA checkpoint hash mismatch; V3 proposals/features would no longer match the frozen training source"
        )
    selection = checkpoint.get("holdout_selection") or {}
    threshold = args.quality_threshold
    if threshold is None:
        threshold = selection.get("threshold")
    if threshold is None:
        raise RuntimeError("checkpoint has no selected quality threshold; pass --quality-threshold explicitly")
    threshold = float(threshold)

    yolo = YOLO(str(ca_path), task="obb")
    ca_model = yolo.model.to(device).float().eval()
    head = ca_model.model[-1]
    if type(head).__name__ != "OBB" or int(getattr(head, "reg_max", -1)) != 32:
        raise RuntimeError(
            f"V3 requires pure CA OBB(reg_max=32); received head={type(head).__name__}, "
            f"reg_max={getattr(head, 'reg_max', None)}"
        )
    for parameter in ca_model.parameters():
        parameter.requires_grad_(False)
    training_args = checkpoint.get("arguments", {})
    proposal_conf = float(training_args.get("proposal_conf", 0.01))
    nms_iou = float(training_args.get("nms_iou", 0.70))
    max_det = int(training_args.get("max_det", 300))

    dataset, data = build_dataset(args.data, args.split, args.imgsz, args.batch, args.workers, rect=True)
    loader = full_loader(dataset, args.batch, args.workers)
    names = getattr(ca_model, "names", data["names"])
    extractor = FrozenCAExtractor(
        ca_model,
        device=device,
        nc=len(names),
        conf=proposal_conf,
        nms_iou=nms_iou,
        max_det=max_det,
        amp=use_amp,
    )
    try:
        observed_channels = extractor.infer_channels(args.imgsz)
        model_config = checkpoint["model_config"]
        expected_channels = (int(model_config["p2_channels"]), int(model_config["p3_channels"]))
        if observed_channels != expected_channels:
            raise RuntimeError(
                f"CA feature-channel mismatch: checkpoint expects {expected_channels}, observed {observed_channels}"
            )
        refiner = OBBProposalRefinerV3(**model_config).to(device).float().eval()
        refiner.load_state_dict(checkpoint["model_state"], strict=True)
        rows, diagnostics = evaluate_refiner(
            extractor,
            refiner,
            loader,
            names,
            (threshold,),
            amp=use_amp,
            identity_tolerance=args.identity_tolerance,
        )
        lookup = {row["variant"]: row for row in rows}
        coarse = lookup["coarse"]
        refined = lookup[f"quality_{threshold:.3f}"]
        baseline_error = abs(coarse["map50_95"] - args.expected_ca_map50_95)
        audit = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "ca_weights": str(ca_path),
            "ca_sha256": ca_hash,
            "data": args.data,
            "split": args.split,
            "imgsz": args.imgsz,
            "quality_threshold": threshold,
            "proposal_conf": proposal_conf,
            "nms_iou": nms_iou,
            "max_det": max_det,
            "expected_ca_map50_95": args.expected_ca_map50_95,
            "baseline_abs_error": baseline_error,
            "baseline_pass": baseline_error <= args.baseline_tolerance,
            "identity_pass": diagnostics["roundtrip_identity_abs_delta"] <= args.identity_tolerance,
            "delta_map50_95": refined["map50_95"] - coarse["map50_95"],
            "delta_ap75": refined["ap75"] - coarse["ap75"],
            "delta_ap90": refined["ap90"] - coarse["ap90"],
            "gate_ratio": refined["gate_ratio"],
            "short_residual_std": diagnostics["short_residual_std"],
            "long_residual_std": diagnostics["long_residual_std"],
            "test_used": False,
        }
        write_csv(output_dir / "val_metrics.csv", rows)
        write_json(output_dir / "val_diagnostics.json", diagnostics)
        write_json(output_dir / "validation_audit.json", audit)
        if not audit["baseline_pass"]:
            raise RuntimeError(
                f"CA baseline mismatch: observed={coarse['map50_95']:.6f}, "
                f"expected={args.expected_ca_map50_95:.6f}, tolerance={args.baseline_tolerance:.6f}"
            )
        print(output_dir / "validation_audit.json")
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
