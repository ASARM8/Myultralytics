"""Train the proposal-level Refine V3 head while keeping the CA detector frozen.

The module intentionally keeps all heavy imports inside ``main`` so ``--help``
and argument tests remain usable on a local machine without PyTorch.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Any


CANONICAL_CA_WEIGHTS = "/root/autodl-tmp/work-dirs/yolo11_obb_640_811_ca/weights/best.pt"


def parse_thresholds(value: str) -> tuple[float, ...]:
    """Parse and validate a deterministic ascending quality-threshold list."""
    try:
        values = tuple(sorted({round(float(item.strip()), 3) for item in value.split(",") if item.strip()}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("quality thresholds must be comma-separated numbers") from error
    if not values or any(item <= 0.0 or item >= 1.0 for item in values):
        raise argparse.ArgumentTypeError("quality thresholds must contain values strictly between 0 and 1")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Refine V3 on frozen pure-CA post-NMS proposals (train-fit/holdout, then one val run)."
    )
    parser.add_argument("--ca-weights", default=CANONICAL_CA_WEIGHTS)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640, choices=(640,))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument(
        "--group-regex",
        default="",
        help="Optional regex whose first capture group identifies one scene; default groups by image path.",
    )
    parser.add_argument("--proposal-conf", type=float, default=0.01)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--match-iou", type=float, default=0.30)
    parser.add_argument("--quality-min-gain", type=float, default=0.002)
    parser.add_argument("--quality-thresholds", type=parse_thresholds, default=parse_thresholds("0.3,0.5,0.7,0.9"))
    parser.add_argument("--roi-height", type=int, default=5)
    parser.add_argument("--roi-width", type=int, default=24)
    parser.add_argument("--roi-channels", type=int, default=32)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--long-context", type=float, default=1.2)
    parser.add_argument("--short-context", type=float, default=4.0)
    parser.add_argument("--min-short-context-px", type=float, default=16.0)
    parser.add_argument("--short-negative-limit", type=float, default=1.5)
    parser.add_argument("--short-positive-limit", type=float, default=0.25)
    parser.add_argument("--long-negative-limit", type=float, default=0.15)
    parser.add_argument("--long-positive-limit", type=float, default=0.15)
    parser.add_argument("--target-margin", type=float, default=0.99)
    parser.add_argument("--tiny-reference-px", type=float, default=8.0)
    parser.add_argument("--tiny-weight-floor", type=float, default=0.25)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.05)
    parser.add_argument("--geometry-gain", type=float, default=1.0)
    parser.add_argument("--quality-gain", type=float, default=0.5)
    parser.add_argument("--identity-gain", type=float, default=0.02)
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected-ca-map50-95", type=float, default=0.45413)
    parser.add_argument("--baseline-tolerance", type=float, default=0.002)
    parser.add_argument("--identity-tolerance", type=float, default=5e-4)
    parser.add_argument("--output-dir", required=True)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if Path(args.ca_weights).as_posix() != Path(CANONICAL_CA_WEIGHTS).as_posix():
        parser.error(f"V3 seed0 is locked to the canonical CA checkpoint: {CANONICAL_CA_WEIGHTS}")
    if args.batch <= 0 or args.workers < 0 or args.epochs <= 0 or args.eval_interval <= 0:
        parser.error("batch/epochs/eval-interval must be positive and workers must be non-negative")
    if not 0.0 < args.holdout_fraction < 1.0:
        parser.error("--holdout-fraction must be in (0, 1)")
    if args.eval_interval > args.epochs:
        parser.error("--eval-interval cannot exceed --epochs")
    unit_interval = ("proposal_conf", "nms_iou", "match_iou", "focal_alpha")
    if any(not 0.0 < getattr(args, name) < 1.0 for name in unit_interval):
        parser.error("proposal-conf, nms-iou, match-iou and focal-alpha must be in (0, 1)")
    if args.quality_min_gain < 0.0:
        parser.error("--quality-min-gain must be non-negative")
    positive = (
        "max_det",
        "roi_height",
        "roi_width",
        "roi_channels",
        "hidden_channels",
        "long_context",
        "short_context",
        "min_short_context_px",
        "short_negative_limit",
        "short_positive_limit",
        "long_negative_limit",
        "long_positive_limit",
        "tiny_reference_px",
        "smooth_l1_beta",
        "geometry_gain",
        "quality_gain",
        "lr",
        "grad_clip",
        "baseline_tolerance",
        "identity_tolerance",
    )
    if any(getattr(args, name) <= 0 for name in positive):
        parser.error("all sizes, residual limits, gains, lr and tolerances must be positive")
    if not 0.0 < args.tiny_weight_floor <= 1.0:
        parser.error("--tiny-weight-floor must be in (0, 1]")
    if not 0.0 < args.target_margin < 1.0:
        parser.error("--target-margin must be strictly between 0 and 1")
    if (
        args.identity_gain < 0.0
        or args.weight_decay < 0.0
        or args.focal_gamma < 0.0
        or args.warmup_epochs < 0.0
    ):
        parser.error("identity-gain, weight-decay, focal-gamma and warmup-epochs must be non-negative")
    if not 0.0 <= args.expected_ca_map50_95 <= 1.0:
        parser.error("--expected-ca-map50-95 must be in [0, 1]")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def focal_binary_loss(torch, logits, targets, alpha: float, gamma: float):
    functional = torch.nn.functional
    cross_entropy = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = logits.sigmoid()
    pt = targets * probability + (1.0 - targets) * (1.0 - probability)
    alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)
    return (alpha_t * (1.0 - pt).pow(gamma) * cross_entropy).mean()


def select_holdout_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in rows if row["variant"].startswith("quality_")]
    if not candidates:
        raise RuntimeError("holdout evaluation did not produce a quality-gated candidate")
    non_degenerate = [row for row in candidates if 0.001 < float(row.get("gate_ratio", 0.5)) < 0.999]
    if non_degenerate:
        candidates = non_degenerate
    # mAP is the primary criterion. AP75/AP90 break exact ties; a higher
    # threshold is the final tie-breaker because it changes fewer proposals.
    return max(
        candidates,
        key=lambda row: (
            row["map50_95"],
            row["ap75"],
            row["ap90"],
            float(row["variant"].split("_", 1)[1]),
        ),
    )


def model_config(args: argparse.Namespace, p2_channels: int, p3_channels: int) -> dict[str, Any]:
    return {
        "p2_channels": p2_channels,
        "p3_channels": p3_channels,
        "roi_channels": args.roi_channels,
        "roi_size": (args.roi_height, args.roi_width),
        "hidden_channels": args.hidden_channels,
        "long_context": args.long_context,
        "short_context": args.short_context,
        "min_short_context_px": args.min_short_context_px,
        "short_negative_limit": args.short_negative_limit,
        "short_positive_limit": args.short_positive_limit,
        "long_negative_limit": args.long_negative_limit,
        "long_positive_limit": args.long_positive_limit,
        "target_margin": args.target_margin,
        "enable_center": False,
        "center_limit": 1.0,
    }


def save_checkpoint(
    torch, path: Path, refiner, optimizer, epoch: int, config: dict[str, Any], metadata: dict[str, Any]
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "architecture": "OBBProposalRefinerV3",
            "epoch": epoch,
            "model_config": config,
            "model_state": refiner.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            **metadata,
        },
        path,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    omp_threads = os.environ.get("OMP_NUM_THREADS", "")
    if not omp_threads.isdigit() or int(omp_threads) <= 0:
        os.environ["OMP_NUM_THREADS"] = "1"

    import numpy as np
    import torch
    import torch.nn.functional as F

    from ultralytics import YOLO
    from ultralytics.nn.modules.refine_v3 import OBBProposalRefinerV3
    from ultralytics.utils.torch_utils import select_device

    from .runtime import (
        FrozenCAExtractor,
        build_dataset,
        build_supervision,
        evaluate_refiner,
        full_loader,
        pad_detections,
        sha256_file,
        split_dataset_indices,
        subset_loader,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    output_dir = Path(args.output_dir)
    if (output_dir / "run_manifest.json").exists():
        raise FileExistsError(
            f"output directory already contains a V3 run: {output_dir}; use a new directory to avoid mixed evidence"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    device = select_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    ca_path = Path(args.ca_weights)
    if not ca_path.is_file():
        raise FileNotFoundError(f"canonical CA checkpoint not found: {ca_path}")
    ca_hash = sha256_file(ca_path)
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

    train_dataset, data = build_dataset(args.data, "train", args.imgsz, args.batch, args.workers, rect=False)
    fit_indices, holdout_indices, fit_groups, holdout_groups = split_dataset_indices(
        train_dataset.im_files, args.holdout_fraction, args.seed, args.group_regex
    )
    fit_loader = subset_loader(train_dataset, fit_indices, args.batch, args.workers, shuffle=True)
    holdout_loader = subset_loader(train_dataset, holdout_indices, args.batch, args.workers, shuffle=False)
    names = getattr(ca_model, "names", data["names"])
    nc = len(names)

    extractor = FrozenCAExtractor(
        ca_model,
        device=device,
        nc=nc,
        conf=args.proposal_conf,
        nms_iou=args.nms_iou,
        max_det=args.max_det,
        amp=use_amp,
    )
    p2_channels, p3_channels = extractor.infer_channels(args.imgsz)
    config = model_config(args, p2_channels, p3_channels)
    refiner = OBBProposalRefinerV3(**config).to(device).float()
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    manifest = {
        "stage": "V3 scale-only seed0",
        "ca_weights": str(ca_path),
        "ca_sha256": ca_hash,
        "data": args.data,
        "imgsz": args.imgsz,
        "train_split": "train",
        "selection_split": "train-holdout",
        "final_eval_split": "val",
        "test_used": False,
        "fit_images": len(fit_indices),
        "holdout_images": len(holdout_indices),
        "fit_groups": len(fit_groups),
        "holdout_groups": len(holdout_groups),
        "group_overlap": len(fit_groups & holdout_groups),
        "arguments": vars(args),
        "model_config": config,
    }
    write_json(output_dir / "run_manifest.json", manifest)

    history: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    global_step = 0
    warmup_steps = max(int(round(args.warmup_epochs * len(fit_loader))), 1) if args.warmup_epochs else 0

    try:
        for epoch in range(1, args.epochs + 1):
            refiner.train()
            totals = {
                "loss": 0.0,
                "geometry_loss": 0.0,
                "quality_loss": 0.0,
                "identity_loss": 0.0,
                "batches": 0,
                "valid_proposals": 0,
                "matched_proposals": 0,
                "quality_positives": 0,
                "short_target_clipped": 0,
                "long_target_clipped": 0,
            }
            for batch in fit_loader:
                images, p2, p3, detections = extractor.infer(batch)
                boxes, scores, classes, valid = pad_detections(detections)
                if not valid.any():
                    continue
                supervision = build_supervision(
                    refiner,
                    boxes.float(),
                    classes,
                    valid,
                    batch,
                    images.shape[2:],
                    match_iou=args.match_iou,
                    quality_min_gain=args.quality_min_gain,
                    tiny_reference_px=args.tiny_reference_px,
                    tiny_weight_floor=args.tiny_weight_floor,
                )
                optimizer.zero_grad(set_to_none=True)
                if warmup_steps:
                    learning_rate = args.lr * min((global_step + 1) / warmup_steps, 1.0)
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    output = refiner(p2, p3, boxes, scores, images.shape[2:], valid)
                    residual = output["residual"]
                    quality_logit = output["quality_logit"].squeeze(-1)
                    matched = supervision["matched"]
                    if matched.any():
                        geometry_raw = F.smooth_l1_loss(
                            residual[matched][..., :2],
                            supervision["clipped_target"][matched][..., :2],
                            beta=args.smooth_l1_beta,
                            reduction="none",
                        ).mean(dim=-1)
                        geometry_weight = supervision["geometry_weight"][matched]
                        geometry_loss = (geometry_raw * geometry_weight).sum() / geometry_weight.sum().clamp_min(1.0)
                    else:
                        geometry_loss = residual.sum() * 0.0
                    quality_loss = focal_binary_loss(
                        torch,
                        quality_logit[valid],
                        supervision["quality_target"][valid],
                        args.focal_alpha,
                        args.focal_gamma,
                    )
                    identity_mask = valid & ~supervision["quality_target"].bool()
                    if identity_mask.any():
                        identity_loss = F.smooth_l1_loss(
                            residual[identity_mask][..., :2],
                            torch.zeros_like(residual[identity_mask][..., :2]),
                            beta=args.smooth_l1_beta,
                        )
                    else:
                        identity_loss = residual.sum() * 0.0
                    loss = (
                        args.geometry_gain * geometry_loss
                        + args.quality_gain * quality_loss
                        + args.identity_gain * identity_loss
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite training loss at epoch {epoch}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(refiner.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                global_step += 1

                totals["loss"] += float(loss.detach())
                totals["geometry_loss"] += float(geometry_loss.detach())
                totals["quality_loss"] += float(quality_loss.detach())
                totals["identity_loss"] += float(identity_loss.detach())
                totals["batches"] += 1
                totals["valid_proposals"] += int(valid.sum())
                totals["matched_proposals"] += int(supervision["matched"].sum())
                totals["quality_positives"] += int(supervision["quality_target"].sum())
                if supervision["matched"].any():
                    exact = supervision["exact_target"][supervision["matched"]]
                    totals["short_target_clipped"] += int(
                        (
                            (exact[:, 0] < -args.short_negative_limit * args.target_margin)
                            | (exact[:, 0] > args.short_positive_limit * args.target_margin)
                        ).sum()
                    )
                    totals["long_target_clipped"] += int(
                        (
                            (exact[:, 1] < -args.long_negative_limit * args.target_margin)
                            | (exact[:, 1] > args.long_positive_limit * args.target_margin)
                        ).sum()
                    )

            if not totals["batches"]:
                raise RuntimeError("no train-fit batch contained a valid CA proposal")
            record = {
                "epoch": epoch,
                "loss": totals["loss"] / totals["batches"],
                "geometry_loss": totals["geometry_loss"] / totals["batches"],
                "quality_loss": totals["quality_loss"] / totals["batches"],
                "identity_loss": totals["identity_loss"] / totals["batches"],
                "valid_proposals": totals["valid_proposals"],
                "matched_proposals": totals["matched_proposals"],
                "quality_positives": totals["quality_positives"],
                "matched_ratio": totals["matched_proposals"] / max(totals["valid_proposals"], 1),
                "quality_positive_ratio": totals["quality_positives"] / max(totals["valid_proposals"], 1),
                "short_target_clip_ratio": totals["short_target_clipped"] / max(totals["matched_proposals"], 1),
                "long_target_clip_ratio": totals["long_target_clipped"] / max(totals["matched_proposals"], 1),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(record)
            write_csv(output_dir / "train_history.csv", history)
            print(
                f"epoch={epoch:02d}/{args.epochs} loss={record['loss']:.6f} "
                f"geometry={record['geometry_loss']:.6f} quality={record['quality_loss']:.6f} "
                f"identity={record['identity_loss']:.6f}"
            )

            metadata = {"ca_weights": str(ca_path), "ca_sha256": ca_hash, "arguments": vars(args)}
            save_checkpoint(torch, checkpoint_dir / "last.pt", refiner, optimizer, epoch, config, metadata)
            should_evaluate = epoch % args.eval_interval == 0 or epoch == args.epochs
            if not should_evaluate:
                continue
            evaluation, diagnostics = evaluate_refiner(
                extractor,
                refiner,
                holdout_loader,
                names,
                args.quality_thresholds,
                amp=use_amp,
                identity_tolerance=args.identity_tolerance,
            )
            for row in evaluation:
                holdout_rows.append({"epoch": epoch, **row})
            write_csv(output_dir / "holdout_metrics.csv", holdout_rows)
            write_json(output_dir / f"holdout_diagnostics_epoch{epoch:03d}.json", diagnostics)
            selected = select_holdout_row(evaluation)
            candidate = {
                "epoch": epoch,
                "threshold": float(selected["variant"].split("_", 1)[1]),
                "row": selected,
                "diagnostics": diagnostics,
            }
            checkpoint_path = checkpoint_dir / f"epoch{epoch:03d}.pt"
            save_checkpoint(
                torch,
                checkpoint_path,
                refiner,
                optimizer,
                epoch,
                config,
                {**metadata, "holdout_selection": candidate},
            )
            if best is None or (
                selected["map50_95"], selected["ap75"], selected["ap90"], candidate["threshold"]
            ) > (
                best["row"]["map50_95"],
                best["row"]["ap75"],
                best["row"]["ap90"],
                best["threshold"],
            ):
                best = {**candidate, "checkpoint": str(checkpoint_path)}
                save_checkpoint(
                    torch,
                    checkpoint_dir / "best.pt",
                    refiner,
                    optimizer,
                    epoch,
                    config,
                    {**metadata, "holdout_selection": best},
                )
            print(
                f"  holdout selected threshold={candidate['threshold']:.3f}, "
                f"mAP50-95={selected['map50_95']:.6f}, delta={selected['delta_map50_95_vs_coarse']:+.6f}"
            )

        if best is None:
            raise RuntimeError("training finished without a holdout checkpoint")
        best_payload = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
        refiner.load_state_dict(best_payload["model_state"], strict=True)
        write_json(output_dir / "selection.json", best)

        # Validation is deliberately run exactly once after epoch and threshold
        # selection have been frozen on train-holdout.
        val_dataset, _ = build_dataset(args.data, "val", args.imgsz, args.batch, args.workers, rect=True)
        val_loader = full_loader(val_dataset, args.batch, args.workers)
        val_rows, val_diagnostics = evaluate_refiner(
            extractor,
            refiner,
            val_loader,
            names,
            (best["threshold"],),
            amp=use_amp,
            identity_tolerance=args.identity_tolerance,
        )
        write_csv(output_dir / "val_metrics.csv", val_rows)
        write_json(output_dir / "val_diagnostics.json", val_diagnostics)
        val_lookup = {row["variant"]: row for row in val_rows}
        coarse = val_lookup["coarse"]
        refined = val_lookup[f"quality_{best['threshold']:.3f}"]
        baseline_error = abs(coarse["map50_95"] - args.expected_ca_map50_95)
        val_delta = refined["map50_95"] - coarse["map50_95"]
        holdout_delta = float(best["row"]["delta_map50_95_vs_coarse"])
        gate_ratio = float(refined["gate_ratio"])
        residual_non_constant = max(
            float(val_diagnostics["short_residual_std"]), float(val_diagnostics["long_residual_std"])
        ) > 1e-4
        acceptance = {
            "expected_ca_map50_95": args.expected_ca_map50_95,
            "observed_ca_map50_95": coarse["map50_95"],
            "baseline_abs_error": baseline_error,
            "baseline_pass": baseline_error <= args.baseline_tolerance,
            "identity_pass": val_diagnostics["roundtrip_identity_abs_delta"] <= args.identity_tolerance,
            "holdout_delta_map50_95": holdout_delta,
            "delta_map50_95": val_delta,
            "delta_ap75": refined["ap75"] - coarse["ap75"],
            "delta_ap90": refined["ap90"] - coarse["ap90"],
            "gate_ratio": gate_ratio,
            "gate_non_degenerate": 0.001 < gate_ratio < 0.999,
            "residual_non_constant": residual_non_constant,
            "holdout_val_direction_consistent": holdout_delta > 0.0 and val_delta > 0.0,
            "screening_pass": (
                holdout_delta > 0.0
                and val_delta >= 0.002
                and refined["ap75"] >= coarse["ap75"]
                and refined["ap90"] - coarse["ap90"] >= -0.002
                and 0.001 < gate_ratio < 0.999
                and residual_non_constant
            ),
            "test_used": False,
        }
        write_json(output_dir / "acceptance.json", acceptance)
        report = [
            "# Refine V3 seed0 训练结果",
            "",
            f"- CA SHA256：`{ca_hash}`",
            f"- train-fit / train-holdout：{len(fit_indices)} / {len(holdout_indices)} 张；组重叠：0。",
            f"- 选中 epoch / 质量阈值：{best['epoch']} / {best['threshold']:.3f}。",
            f"- val coarse mAP50-95：{coarse['map50_95']:.6f}。",
            f"- val refined mAP50-95：{refined['map50_95']:.6f}（Δ={acceptance['delta_map50_95']:+.6f}）。",
            f"- AP75 Δ：{acceptance['delta_ap75']:+.6f}；AP90 Δ：{acceptance['delta_ap90']:+.6f}。",
            f"- holdout Δ / val gate ratio：{holdout_delta:+.6f} / {gate_ratio:.6f}。",
            f"- baseline / identity / seed0 screening：{acceptance['baseline_pass']} / "
            f"{acceptance['identity_pass']} / {acceptance['screening_pass']}。",
            "- test split：未使用。",
            "",
            "seed0 screening 只决定是否进入后续结构或种子复核，不构成正式论文结论。",
        ]
        (output_dir / "training_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        if not acceptance["baseline_pass"]:
            raise RuntimeError(
                f"CA baseline mismatch: observed={coarse['map50_95']:.6f}, "
                f"expected={args.expected_ca_map50_95:.6f}, tolerance={args.baseline_tolerance:.6f}"
            )
        print(output_dir / "training_report.md")
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
