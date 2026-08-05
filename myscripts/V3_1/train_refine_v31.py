"""Train Refine V3.1 on the original image-level split with a frozen CA detector."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any

from myscripts.V3.train_refine_v3 import focal_binary_loss, write_csv, write_json


CANONICAL_CA_WEIGHTS = "/root/autodl-tmp/work-dirs/yolo11_obb_640_811_ca/weights/best.pt"
EXPERIMENTS = ("geometry_only", "quality_aux")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train conservative Refine V3.1 on frozen CA post-NMS proposals. "
            "Inference always refines all valid proposals and never reruns NMS."
        )
    )
    parser.add_argument("--experiment", required=True, choices=EXPERIMENTS)
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
    parser.add_argument("--proposal-conf", type=float, default=0.01)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--match-iou", type=float, default=0.30)
    parser.add_argument("--quality-min-gain", type=float, default=0.002)
    parser.add_argument("--roi-height", type=int, default=5)
    parser.add_argument("--roi-width", type=int, default=24)
    parser.add_argument("--roi-channels", type=int, default=32)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--long-context", type=float, default=1.2)
    parser.add_argument("--short-context", type=float, default=4.0)
    parser.add_argument("--min-short-context-px", type=float, default=16.0)
    parser.add_argument("--short-negative-limit", type=float, default=0.50)
    parser.add_argument("--short-positive-limit", type=float, default=0.20)
    parser.add_argument("--long-negative-limit", type=float, default=0.08)
    parser.add_argument("--long-positive-limit", type=float, default=0.08)
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
    parser.add_argument("--minimum-map-gain", type=float, default=0.03)
    parser.add_argument("--max-ap90-drop", type=float, default=0.002)
    parser.add_argument("--max-boundary-ratio", type=float, default=0.10)
    parser.add_argument("--output-dir", required=True)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if Path(args.ca_weights).as_posix() != Path(CANONICAL_CA_WEIGHTS).as_posix():
        parser.error(f"V3.1 is locked to the canonical CA checkpoint: {CANONICAL_CA_WEIGHTS}")
    if args.batch <= 0 or args.workers < 0 or args.epochs <= 0 or args.eval_interval <= 0:
        parser.error("batch/epochs/eval-interval must be positive and workers must be non-negative")
    if args.eval_interval > args.epochs:
        parser.error("--eval-interval cannot exceed --epochs")
    for name in ("holdout_fraction", "proposal_conf", "nms_iou", "match_iou", "focal_alpha"):
        if not 0.0 < getattr(args, name) < 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in (0, 1)")
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
        "minimum_map_gain",
        "max_ap90_drop",
        "max_boundary_ratio",
    )
    if any(getattr(args, name) <= 0 for name in positive):
        parser.error("sizes, limits, gains, learning rate, tolerances and acceptance thresholds must be positive")
    if not 0.0 < args.target_margin < 1.0 or not 0.0 < args.tiny_weight_floor <= 1.0:
        parser.error("target-margin must be in (0,1) and tiny-weight-floor in (0,1]")
    if args.quality_min_gain < 0 or args.identity_gain < 0 or args.weight_decay < 0 or args.focal_gamma < 0:
        parser.error("quality-min-gain, identity-gain, weight-decay and focal-gamma must be non-negative")
    if args.warmup_epochs < 0 or not 0 <= args.expected_ca_map50_95 <= 1:
        parser.error("warmup-epochs must be non-negative and expected CA mAP must be in [0,1]")


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
        "use_quality_aux": args.experiment == "quality_aux",
    }


def save_checkpoint(torch, path: Path, refiner, optimizer, epoch: int, config: dict[str, Any], metadata: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "architecture": "OBBProposalRefinerV31",
            "experiment": metadata["arguments"]["experiment"],
            "epoch": epoch,
            "model_config": config,
            "model_state": refiner.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            **metadata,
        },
        path,
    )


def _candidate_key(candidate: dict[str, Any]):
    row = candidate["row"]
    return row["map50_95"], row["ap75"], row["ap90"], -candidate["epoch"]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    if not os.environ.get("OMP_NUM_THREADS", "").isdigit() or int(os.environ.get("OMP_NUM_THREADS", "0")) <= 0:
        os.environ["OMP_NUM_THREADS"] = "1"

    import numpy as np
    import torch
    import torch.nn.functional as F

    from ultralytics import YOLO
    from ultralytics.nn.modules.refine_v31 import OBBProposalRefinerV31
    from ultralytics.utils.torch_utils import select_device

    from .runtime import (
        FrozenCAExtractor,
        build_dataset,
        build_supervision,
        evaluate_refiner_v31,
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

    output_dir = Path(args.output_dir)
    if (output_dir / "run_manifest.json").exists():
        raise FileExistsError(f"output directory already contains a V3.1 run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    ca_path = Path(args.ca_weights)
    if not ca_path.is_file():
        raise FileNotFoundError(f"canonical CA checkpoint not found: {ca_path}")
    ca_hash_before = sha256_file(ca_path)
    device = select_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")

    yolo = YOLO(str(ca_path), task="obb")
    ca_model = yolo.model.to(device).float().eval()
    head = ca_model.model[-1]
    if type(head).__name__ != "OBB" or int(getattr(head, "reg_max", -1)) != 32:
        raise RuntimeError(
            f"V3.1 requires pure CA OBB(reg_max=32); received head={type(head).__name__}, "
            f"reg_max={getattr(head, 'reg_max', None)}"
        )
    for parameter in ca_model.parameters():
        parameter.requires_grad_(False)

    train_dataset, data = build_dataset(args.data, "train", args.imgsz, args.batch, args.workers, rect=False)
    fit_indices, holdout_indices, fit_groups, holdout_groups = split_dataset_indices(
        train_dataset.im_files, args.holdout_fraction, args.seed, ""
    )
    fit_loader = subset_loader(train_dataset, fit_indices, args.batch, args.workers, shuffle=True)
    holdout_loader = subset_loader(train_dataset, holdout_indices, args.batch, args.workers, shuffle=False)
    names = getattr(ca_model, "names", data["names"])
    extractor = FrozenCAExtractor(
        ca_model,
        device=device,
        nc=len(names),
        conf=args.proposal_conf,
        nms_iou=args.nms_iou,
        max_det=args.max_det,
        amp=use_amp,
    )

    try:
        p2_channels, p3_channels = extractor.infer_channels(args.imgsz)
        config = model_config(args, p2_channels, p3_channels)
        refiner = OBBProposalRefinerV31(**config).to(device).float()
        optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        manifest = {
            "stage": "Refine V3.1 original-split seed run",
            "experiment": args.experiment,
            "architecture": "OBBProposalRefinerV31",
            "ca_weights": str(ca_path),
            "ca_sha256_before": ca_hash_before,
            "data": args.data,
            "imgsz": args.imgsz,
            "split_policy": "original train; deterministic image-path holdout",
            "fit_images": len(fit_indices),
            "holdout_images": len(holdout_indices),
            "fit_groups": len(fit_groups),
            "holdout_groups": len(holdout_groups),
            "group_overlap": len(fit_groups & holdout_groups),
            "proposal_policy": "all",
            "rerun_nms": False,
            "quality_used_at_inference": False,
            "test_used": False,
            "model_config": config,
            "arguments": vars(args),
        }
        write_json(output_dir / "run_manifest.json", manifest)

        history: list[dict[str, Any]] = []
        holdout_rows: list[dict[str, Any]] = []
        best = None
        global_step = 0
        warmup_steps = max(int(args.warmup_epochs * len(fit_loader)), 1)
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
                supervision = build_supervision(
                    refiner,
                    boxes,
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
                global_step += 1
                learning_rate = args.lr * min(global_step / warmup_steps, 1.0)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    output = refiner(p2, p3, boxes, scores, images.shape[2:], valid)
                    residual = output["residual"]
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

                    if args.experiment == "quality_aux" and valid.any():
                        quality_logit = output["quality_logit"]
                        if quality_logit is None:
                            raise RuntimeError("quality_aux experiment constructed a model without a quality head")
                        quality_loss = focal_binary_loss(
                            torch,
                            quality_logit.squeeze(-1)[valid],
                            supervision["quality_target"][valid],
                            args.focal_alpha,
                            args.focal_gamma,
                        )
                    else:
                        quality_loss = residual.sum() * 0.0

                    identity_mask = valid & ~supervision["quality_target"].bool()
                    if identity_mask.any():
                        identity_loss = F.smooth_l1_loss(
                            residual[identity_mask][..., :2],
                            torch.zeros_like(residual[identity_mask][..., :2]),
                            beta=args.smooth_l1_beta,
                        )
                    else:
                        identity_loss = residual.sum() * 0.0
                    loss = args.geometry_gain * geometry_loss + args.identity_gain * identity_loss
                    if args.experiment == "quality_aux":
                        loss = loss + args.quality_gain * quality_loss

                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite training loss at epoch {epoch}")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(refiner.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

                totals["loss"] += float(loss.detach())
                totals["geometry_loss"] += float(geometry_loss.detach())
                totals["quality_loss"] += float(quality_loss.detach())
                totals["identity_loss"] += float(identity_loss.detach())
                totals["batches"] += 1
                totals["valid_proposals"] += int(valid.sum())
                totals["matched_proposals"] += int(matched.sum())
                totals["quality_positives"] += int(supervision["quality_target"].sum())
                if matched.any():
                    exact = supervision["exact_target"][matched]
                    totals["short_target_clipped"] += int(
                        ((exact[:, 0] < -args.short_negative_limit * args.target_margin)
                         | (exact[:, 0] > args.short_positive_limit * args.target_margin)).sum()
                    )
                    totals["long_target_clipped"] += int(
                        ((exact[:, 1] < -args.long_negative_limit * args.target_margin)
                         | (exact[:, 1] > args.long_positive_limit * args.target_margin)).sum()
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

            metadata = {"ca_weights": str(ca_path), "ca_sha256": ca_hash_before, "arguments": vars(args)}
            save_checkpoint(torch, checkpoint_dir / "last.pt", refiner, optimizer, epoch, config, metadata)
            if epoch % args.eval_interval != 0 and epoch != args.epochs:
                continue
            rows, diagnostics = evaluate_refiner_v31(
                extractor,
                refiner,
                holdout_loader,
                names,
                amp=use_amp,
                match_iou=args.match_iou,
                quality_min_gain=args.quality_min_gain,
                tiny_reference_px=args.tiny_reference_px,
                tiny_weight_floor=args.tiny_weight_floor,
                identity_tolerance=args.identity_tolerance,
            )
            for row in rows:
                holdout_rows.append({"epoch": epoch, **row})
            write_csv(output_dir / "holdout_metrics.csv", holdout_rows)
            write_json(output_dir / f"holdout_diagnostics_epoch{epoch:03d}.json", diagnostics)
            refined = next(row for row in rows if row["variant"] == "refined")
            candidate = {"epoch": epoch, "row": refined, "diagnostics": diagnostics}
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
            if best is None or _candidate_key(candidate) > _candidate_key(best):
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
                f"  holdout mAP50-95={refined['map50_95']:.6f}, "
                f"delta={refined['delta_map50_95_vs_coarse']:+.6f}"
            )

        if best is None:
            raise RuntimeError("training finished without a holdout checkpoint")
        best_payload = torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=False)
        refiner.load_state_dict(best_payload["model_state"], strict=True)
        write_json(output_dir / "selection.json", best)

        val_dataset, _ = build_dataset(args.data, "val", args.imgsz, args.batch, args.workers, rect=True)
        val_loader = full_loader(val_dataset, args.batch, args.workers)
        val_rows, val_diagnostics = evaluate_refiner_v31(
            extractor,
            refiner,
            val_loader,
            names,
            amp=use_amp,
            match_iou=args.match_iou,
            quality_min_gain=args.quality_min_gain,
            tiny_reference_px=args.tiny_reference_px,
            tiny_weight_floor=args.tiny_weight_floor,
            identity_tolerance=args.identity_tolerance,
        )
        write_csv(output_dir / "val_metrics.csv", val_rows)
        write_json(output_dir / "val_diagnostics.json", val_diagnostics)
        lookup = {row["variant"]: row for row in val_rows}
        coarse, refined = lookup["coarse"], lookup["refined"]
        ca_hash_after = sha256_file(ca_path)
        delta_map = refined["map50_95"] - coarse["map50_95"]
        delta_ap75 = refined["ap75"] - coarse["ap75"]
        delta_ap90 = refined["ap90"] - coarse["ap90"]
        residual_non_constant = max(
            val_diagnostics["short_residual_std"], val_diagnostics["long_residual_std"]
        ) > 1e-4
        acceptance = {
            "experiment": args.experiment,
            "selected_epoch": best["epoch"],
            "expected_ca_map50_95": args.expected_ca_map50_95,
            "observed_ca_map50_95": coarse["map50_95"],
            "baseline_abs_error": abs(coarse["map50_95"] - args.expected_ca_map50_95),
            "baseline_pass": abs(coarse["map50_95"] - args.expected_ca_map50_95) <= args.baseline_tolerance,
            "identity_pass": val_diagnostics["identity_max_abs_metric_delta"] <= args.identity_tolerance,
            "ca_hash_before": ca_hash_before,
            "ca_hash_after": ca_hash_after,
            "ca_hash_pass": ca_hash_before == ca_hash_after,
            "delta_map50_95": delta_map,
            "delta_ap75": delta_ap75,
            "delta_ap90": delta_ap90,
            "minimum_map_gain": args.minimum_map_gain,
            "matched_delta_iou_mean": val_diagnostics["matched_delta_iou_mean"],
            "matched_improved_ratio": val_diagnostics["matched_improved_ratio"],
            "matched_worsened_ratio": val_diagnostics["matched_worsened_ratio"],
            "short_boundary_ratio": val_diagnostics["short_boundary_ratio"],
            "long_boundary_ratio": val_diagnostics["long_boundary_ratio"],
            "residual_non_constant": residual_non_constant,
            "proposal_policy": "all",
            "rerun_nms": False,
            "test_used": False,
        }
        acceptance["screening_pass"] = all(
            (
                acceptance["baseline_pass"],
                acceptance["identity_pass"],
                acceptance["ca_hash_pass"],
                delta_map >= args.minimum_map_gain,
                delta_ap75 >= 0.0,
                delta_ap90 >= -args.max_ap90_drop,
                val_diagnostics["matched_delta_iou_mean"] >= 0.0,
                val_diagnostics["matched_improved_ratio"] >= val_diagnostics["matched_worsened_ratio"],
                val_diagnostics["short_boundary_ratio"] <= args.max_boundary_ratio,
                val_diagnostics["long_boundary_ratio"] <= args.max_boundary_ratio,
                residual_non_constant,
            )
        )
        write_json(output_dir / "acceptance.json", acceptance)
        report = [
            "# Refine V3.1 seed0 训练结果",
            "",
            f"- 实验：`{args.experiment}`",
            f"- 选中 epoch：{best['epoch']}",
            f"- CA mAP50-95：{coarse['map50_95']:.6f}",
            f"- V3.1 mAP50-95：{refined['map50_95']:.6f}（Δ={delta_map:+.6f}）",
            f"- AP75 Δ：{delta_ap75:+.6f}；AP90 Δ：{delta_ap90:+.6f}",
            f"- 匹配 proposal 平均 IoU Δ：{val_diagnostics['matched_delta_iou_mean']:+.6f}",
            f"- 短边/长边边界比例：{val_diagnostics['short_boundary_ratio']:.6f} / "
            f"{val_diagnostics['long_boundary_ratio']:.6f}",
            f"- CA 哈希一致：{acceptance['ca_hash_pass']}",
            f"- seed0 筛选：{acceptance['screening_pass']}",
            "- 推理策略：全部 proposal；不使用质量门控；不执行第二次 NMS。",
            "- test：未使用。",
            "",
        ]
        (output_dir / "RESULTS.md").write_text("\n".join(report), encoding="utf-8")
        print(output_dir / "acceptance.json")
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
