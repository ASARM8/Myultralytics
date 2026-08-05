"""Shared runtime for Refine V3.1 training and validation.

V3.1 consumes the frozen CA detector's post-NMS proposals, refines every valid
proposal once, and deliberately does not run a second NMS.
"""

from __future__ import annotations

import math

import torch

from ultralytics.utils.metrics import OBBMetrics, probiou

from myscripts.V3.runtime import (
    FrozenCAExtractor,
    build_dataset,
    build_supervision,
    full_loader,
    metric_summary,
    pad_detections,
    sha256_file,
    split_dataset_indices,
    subset_loader,
    update_metric,
)

__all__ = (
    "FrozenCAExtractor",
    "build_dataset",
    "build_supervision",
    "evaluate_refiner_v31",
    "full_loader",
    "pad_detections",
    "sha256_file",
    "split_dataset_indices",
    "subset_loader",
)


def _direct_prediction(boxes, scores, classes):
    """Create a metric prediction without score changes or another NMS pass."""
    return {"bboxes": boxes, "conf": scores, "cls": classes}


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else math.nan


def evaluate_refiner_v31(
    extractor: FrozenCAExtractor,
    refiner,
    loader,
    names,
    *,
    amp: bool,
    match_iou: float,
    quality_min_gain: float,
    tiny_reference_px: float,
    tiny_weight_floor: float,
    identity_tolerance: float = 5e-4,
):
    """Evaluate coarse, zero-residual identity, and all-proposal refinement."""
    variants = ("coarse", "identity", "refined")
    metrics = {name: OBBMetrics(names=names) for name in variants}
    residual_values = []
    quality_values = []
    matched_delta_iou = []
    valid_count = 0

    refiner.eval()
    with torch.inference_mode():
        for batch in loader:
            images, p2, p3, detections = extractor.infer(batch)
            boxes, scores, classes, valid = pad_detections(detections)
            with torch.autocast(
                device_type=extractor.device.type,
                dtype=torch.float16,
                enabled=bool(amp and extractor.device.type == "cuda"),
            ):
                output = refiner(p2, p3, boxes, scores, images.shape[2:], valid)
            residual = output["residual"].float()
            quality_logit = output.get("quality_logit")
            valid_count += int(valid.sum().item())
            if valid.any():
                residual_values.append(residual[valid][..., :2].detach().cpu())
                if quality_logit is not None:
                    quality_values.append(quality_logit.float().sigmoid().squeeze(-1)[valid].detach().cpu())

            supervision = build_supervision(
                refiner,
                boxes,
                classes,
                valid,
                batch,
                images.shape[2:],
                match_iou=match_iou,
                quality_min_gain=quality_min_gain,
                tiny_reference_px=tiny_reference_px,
                tiny_weight_floor=tiny_weight_floor,
            )
            matched = supervision["matched"]
            if matched.any():
                coarse_matched = boxes[matched].float()
                target_matched = supervision["target_boxes"][matched].float()
                refined_matched = refiner.apply_residual(coarse_matched, residual[matched])
                coarse_iou = probiou(coarse_matched, target_matched).reshape(-1)
                refined_iou = probiou(refined_matched, target_matched).reshape(-1)
                matched_delta_iou.append((refined_iou - coarse_iou).detach().cpu())

            for image_index, detection in enumerate(detections):
                update_metric(metrics["coarse"], detection, batch, image_index)
                count = int(valid[image_index].sum().item())
                proposal_boxes = boxes[image_index, :count].float()
                proposal_scores = scores[image_index, :count]
                proposal_classes = classes[image_index, :count]
                identity_boxes = refiner.apply_residual(proposal_boxes, torch.zeros_like(residual[image_index, :count]))
                refined_boxes = refiner.apply_residual(proposal_boxes, residual[image_index, :count])
                update_metric(
                    metrics["identity"],
                    _direct_prediction(identity_boxes, proposal_scores, proposal_classes),
                    batch,
                    image_index,
                )
                update_metric(
                    metrics["refined"],
                    _direct_prediction(refined_boxes, proposal_scores, proposal_classes),
                    batch,
                    image_index,
                )

    rows = []
    for variant, metric in metrics.items():
        metric.process(plot=False)
        rows.append(metric_summary(metric, variant))
        metric.clear_stats()
    lookup = {row["variant"]: row for row in rows}
    identity_error = max(
        abs(lookup["identity"][key] - lookup["coarse"][key])
        for key in ("map50_95", "ap75", "ap90", "ap95")
    )
    if identity_error > identity_tolerance:
        raise RuntimeError(
            f"zero-residual identity failed: max metric delta={identity_error:.6f} > {identity_tolerance:.6f}"
        )
    for row in rows:
        row["delta_map50_95_vs_coarse"] = row["map50_95"] - lookup["coarse"]["map50_95"]
        row["rerun_nms"] = False
        row["proposal_policy"] = "all"

    residual = torch.cat(residual_values) if residual_values else torch.empty(0, 2)
    quality = torch.cat(quality_values) if quality_values else torch.empty(0)
    delta_iou = torch.cat(matched_delta_iou) if matched_delta_iou else torch.empty(0)
    short_boundary = (
        (residual[:, 0] <= -0.98 * refiner.short_negative_limit)
        | (residual[:, 0] >= 0.98 * refiner.short_positive_limit)
    ) if residual.numel() else torch.empty(0, dtype=torch.bool)
    long_boundary = (
        (residual[:, 1] <= -0.98 * refiner.long_negative_limit)
        | (residual[:, 1] >= 0.98 * refiner.long_positive_limit)
    ) if residual.numel() else torch.empty(0, dtype=torch.bool)
    diagnostics = {
        "valid_proposals": valid_count,
        "matched_proposals": int(delta_iou.numel()),
        "identity_max_abs_metric_delta": identity_error,
        "short_residual_mean": float(residual[:, 0].mean()) if residual.numel() else math.nan,
        "short_residual_std": float(residual[:, 0].std(unbiased=False)) if residual.numel() else math.nan,
        "short_residual_p05": float(torch.quantile(residual[:, 0], 0.05)) if residual.numel() else math.nan,
        "short_residual_p95": float(torch.quantile(residual[:, 0], 0.95)) if residual.numel() else math.nan,
        "long_residual_mean": float(residual[:, 1].mean()) if residual.numel() else math.nan,
        "long_residual_std": float(residual[:, 1].std(unbiased=False)) if residual.numel() else math.nan,
        "long_residual_p05": float(torch.quantile(residual[:, 1], 0.05)) if residual.numel() else math.nan,
        "long_residual_p95": float(torch.quantile(residual[:, 1], 0.95)) if residual.numel() else math.nan,
        "short_boundary_ratio": float(short_boundary.float().mean()) if short_boundary.numel() else math.nan,
        "long_boundary_ratio": float(long_boundary.float().mean()) if long_boundary.numel() else math.nan,
        "matched_delta_iou_mean": float(delta_iou.mean()) if delta_iou.numel() else math.nan,
        "matched_improved_ratio": _safe_ratio(int((delta_iou > 0).sum()), int(delta_iou.numel())),
        "matched_worsened_ratio": _safe_ratio(int((delta_iou < 0).sum()), int(delta_iou.numel())),
        "quality_probability_mean": float(quality.mean()) if quality.numel() else None,
        "quality_probability_std": float(quality.std(unbiased=False)) if quality.numel() else None,
        "proposal_policy": "all",
        "rerun_nms": False,
    }
    return rows, diagnostics
