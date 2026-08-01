"""Run the Refine V3 pre-implementation checks without touching the test split.

The script answers three questions before a new head is implemented:

1. Which geometric degrees of freedom still have oracle room on matched positives?
2. Can a train-fitted quality probe predict when a scale correction is beneficial?
3. Can frozen CA/FPN features predict instance-specific short/long residuals?

The pure CA checkpoint is used as the immutable reference.  A V2.2 checkpoint is
used only after every shared state tensor has been proved bit-identical to that CA
checkpoint.  Probes are fitted on ``train`` and evaluated once on ``val``.

Example:
    python -m myscripts.refine_v3_precheck \
      --ca-weights /root/autodl-tmp/work-dirs/yolo11_obb_640_811_ca/weights/best.pt \
      --refine-weights /root/autodl-tmp/work-dirs/REFINE_V22/weights/epoch0.pt \
      --data /root/autodl-tmp/datasets/TTPLA-640-811/dataset.yaml \
      --imgsz 640 --batch 8 --device 0 --workers 8 \
      --train-split train --eval-split val \
      --output-dir /root/autodl-tmp/paper_exports/refine_v3_precheck
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


STATE_FEATURE_NAMES = (
    "confidence",
    "dfl_entropy",
    "log_short_norm",
    "log_aspect_ratio",
    "anchor_offset_x",
    "anchor_offset_y",
    "anchor_offset_norm",
    "sin_2theta",
    "cos_2theta",
    "log_stride_norm",
)
ORACLE_VARIANTS = (
    "coarse",
    "v22_current",
    "v22_perfect_quality",
    "scale_all",
    "scale_gt_gate",
    "center_all",
    "angle_all",
    "scale_center",
    "scale_angle",
    "full_target",
)


def binary_roc_auc(labels: Sequence[int | bool], scores: Sequence[float]) -> float:
    """Compute tie-aware binary ROC AUC using only the Python standard library."""
    pairs = sorted((float(score), int(bool(label))) for label, score in zip(labels, scores))
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return math.nan
    concordant = 0.0
    negatives_before = 0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        group = pairs[index:end]
        group_positives = sum(label for _, label in group)
        group_negatives = len(group) - group_positives
        concordant += group_positives * negatives_before + 0.5 * group_positives * group_negatives
        negatives_before += group_negatives
        index = end
    return concordant / (positives * negatives)


def binary_average_precision(labels: Sequence[int | bool], scores: Sequence[float]) -> float:
    """Compute step-integrated average precision at distinct score thresholds."""
    pairs = sorted(
        ((float(score), int(bool(label))) for label, score in zip(labels, scores)),
        key=lambda item: item[0],
        reverse=True,
    )
    positives = sum(label for _, label in pairs)
    if positives == 0:
        return math.nan
    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    average_precision = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        group = pairs[index:end]
        group_positives = sum(label for _, label in group)
        true_positives += group_positives
        false_positives += len(group) - group_positives
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
        index = end
    return average_precision


def precheck_decision(
    quality_pass: bool,
    residual_pass: bool,
    center_gain: float,
    angle_gain: float,
    extra_dof_min_gain: float,
) -> dict[str, Any]:
    """Return the frozen V3 go/no-go decision in a unit-testable form."""
    return {
        "quality_pass": bool(quality_pass),
        "residual_pass": bool(residual_pass),
        "recommend_v3": bool(quality_pass and residual_pass),
        "include_center": bool(center_gain >= extra_dof_min_gain),
        "include_angle": bool(angle_gain >= extra_dof_min_gain),
    }


@dataclass
class PriorityReservoir:
    """Deterministic random-priority reservoir for aligned CPU tensors."""

    capacity: int
    generator: Any
    data: dict[str, Any] | None = None
    priorities: Any | None = None
    seen: int = 0

    def update(self, values: dict[str, Any]) -> None:
        if not values:
            return
        length = int(next(iter(values.values())).shape[0])
        if length == 0:
            return
        if any(int(value.shape[0]) != length for value in values.values()):
            raise ValueError("reservoir fields are not row-aligned")
        cpu_values = {name: value.detach().cpu() for name, value in values.items()}
        keys = torch.rand(length, generator=self.generator)
        self.seen += length
        if self.data is None:
            combined = cpu_values
            combined_keys = keys
        else:
            combined = {name: torch.cat((self.data[name], value), dim=0) for name, value in cpu_values.items()}
            combined_keys = torch.cat((self.priorities, keys), dim=0)
        if combined_keys.numel() > self.capacity:
            keep = torch.topk(combined_keys, self.capacity, largest=True, sorted=False).indices
            combined = {name: value[keep] for name, value in combined.items()}
            combined_keys = combined_keys[keep]
        self.data = combined
        self.priorities = combined_keys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ca-weights", type=Path, required=True, help="纯 CA best.pt；作为共享参数审计基准")
    parser.add_argument("--refine-weights", type=Path, required=True, help="已通过验证的 V2.2 checkpoint")
    parser.add_argument(
        "--expect-refine-profile",
        default="stable_raw_short_long",
        help="防止误把 V2.3 或其他语义的 checkpoint 用作 V2.2 诊断权重",
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--train-split", choices=("train",), default="train")
    parser.add_argument("--eval-split", choices=("val",), default="val")
    parser.add_argument("--max-train-batches", type=int, default=0, help="0 表示完整 train")
    parser.add_argument("--max-eval-batches", type=int, default=0, help="0 表示完整 val")
    parser.add_argument("--max-probe-samples-per-level", type=int, default=15000)
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--probe-batch-size", type=int, default=1024)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ar-threshold", type=float, default=30.0)
    parser.add_argument("--short-threshold", type=float, default=16.0)
    parser.add_argument("--benefit-epsilon", type=float, default=1e-7)
    parser.add_argument("--quality-min-roc-auc", type=float, default=0.65)
    parser.add_argument("--residual-min-mae-improvement", type=float, default=0.05)
    parser.add_argument("--residual-min-direction", type=float, default=0.55)
    parser.add_argument("--extra-dof-min-gain", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.imgsz != 640:
        parser.error("创新点一固定使用 imgsz=640")
    if args.eval_split != "val":
        parser.error("V3 方案冻结前禁止读取 test split")
    if args.max_train_batches < 0 or args.max_eval_batches < 0:
        parser.error("--max-*-batches 不能为负数")
    if args.max_probe_samples_per_level <= 0:
        parser.error("--max-probe-samples-per-level 必须大于 0")
    if args.probe_epochs <= 0 or args.probe_batch_size <= 0 or args.probe_lr <= 0:
        parser.error("probe epochs、batch size 和 learning rate 必须大于 0")
    if not 0.5 <= args.quality_min_roc_auc <= 1.0:
        parser.error("--quality-min-roc-auc 必须位于 [0.5, 1]")
    if not 0.0 <= args.residual_min_mae_improvement <= 1.0:
        parser.error("--residual-min-mae-improvement 必须位于 [0, 1]")
    if not 0.0 <= args.residual_min_direction <= 1.0:
        parser.error("--residual-min-direction 必须位于 [0, 1]")
    if args.extra_dof_min_gain < 0 or args.benefit_epsilon < 0:
        parser.error("增益阈值不能为负数")
    for label, path in (("--ca-weights", args.ca_weights), ("--refine-weights", args.refine_weights)):
        if not path.is_file():
            parser.error(f"{label} 不是有效文件: {path}")


def bind_runtime_dependencies():
    """Import cloud-only dependencies lazily and bind refine_diag's helper globals."""
    global np, pd, torch, probiou
    import numpy as np
    import pandas as pd
    import torch

    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_KEYS
    from ultralytics.utils.metrics import probiou
    from ultralytics.utils.tal import make_anchors
    from ultralytics.utils.torch_utils import select_device

    import myscripts.refine_diag as rd

    rd.DEFAULT_CFG = DEFAULT_CFG
    rd.DEFAULT_CFG_KEYS = DEFAULT_CFG_KEYS
    rd.build_dataloader = build_dataloader
    rd.build_yolo_dataset = build_yolo_dataset
    rd.check_det_dataset = check_det_dataset
    rd.get_cfg = get_cfg
    rd.make_anchors = make_anchors
    rd.np = np
    rd.pd = pd
    rd.probiou = probiou
    rd.select_device = select_device
    rd.torch = torch
    return rd, YOLO, check_det_dataset, make_anchors, select_device


def allowed_refine_extra(name: str) -> bool:
    return ".cv5." in name or ".one2one_cv5." in name or name.endswith("._refine_v2_marker")


def audit_shared_checkpoint_state(ca_model, refine_model):
    """Require exact equality for every CA state tensor and reject unexplained extras."""
    ca_state = ca_model.state_dict()
    refine_state = refine_model.state_dict()
    rows = []
    failures = []
    for name, ca_value in ca_state.items():
        if name not in refine_state:
            rows.append({"tensor": name, "kind": "shared", "status": "missing", "max_abs_diff": math.nan})
            failures.append(name)
            continue
        refine_value = refine_state[name]
        if ca_value.shape != refine_value.shape or ca_value.dtype != refine_value.dtype:
            rows.append({"tensor": name, "kind": "shared", "status": "shape_or_dtype", "max_abs_diff": math.nan})
            failures.append(name)
            continue
        equal = torch.equal(ca_value.detach().cpu(), refine_value.detach().cpu())
        max_diff = (
            float((ca_value.detach().float().cpu() - refine_value.detach().float().cpu()).abs().max().item())
            if ca_value.numel()
            else 0.0
        )
        rows.append({"tensor": name, "kind": "shared", "status": "equal" if equal else "different", "max_abs_diff": max_diff})
        if not equal:
            failures.append(name)
    for name in sorted(set(refine_state).difference(ca_state)):
        allowed = allowed_refine_extra(name)
        rows.append(
            {
                "tensor": name,
                "kind": "refine_only" if allowed else "unexpected_extra",
                "status": "allowed" if allowed else "rejected",
                "max_abs_diff": math.nan,
            }
        )
        if not allowed:
            failures.append(name)
    frame = pd.DataFrame(rows, columns=["tensor", "kind", "status", "max_abs_diff"])
    return frame, failures


def flatten_fpn_feature(feature):
    return feature.permute(0, 2, 3, 1).reshape(feature.shape[0], -1, feature.shape[1])


def paired_iou(boxes, targets):
    return probiou(boxes, targets).reshape(-1).float()


def collect_split(
    *,
    split: str,
    loader,
    core_model,
    criterion,
    refine_head,
    rd,
    make_anchors,
    args: argparse.Namespace,
    reservoirs: dict[str, PriorityReservoir],
    keep_eval_rows: bool,
):
    """Collect matched-positive probe rows and optional complete validation summaries."""
    eval_chunks: dict[str, list[Any]] = {name: [] for name in (
        "fpn_level", "coarse_iou", "v22_current", "v22_perfect_quality", "scale_all",
        "scale_gt_gate", "center_all", "angle_all", "scale_center", "scale_angle", "full_target", "target_short",
        "target_ar", "anchor_offset_norm", "current_gate",
    )}
    seen_images = 0
    processed_batches = 0
    target_limit = float(getattr(refine_head, "refine_target_limit", 0.1))
    delta_max = float(getattr(refine_head, "refine_delta_max", 0.1))

    for batch_index, raw_batch in enumerate(loader):
        maximum = args.max_train_batches if split == args.train_split else args.max_eval_batches
        if maximum and batch_index >= maximum:
            break
        batch = rd.prepare_batch(raw_batch, criterion.device)
        with torch.inference_mode():
            raw = rd.extract_raw_predictions(core_model(batch["img"]))
            assigned = rd.build_assignments(criterion, raw, batch)
        fg_mask = assigned["fg_mask"].bool()
        if not fg_mask.any():
            continue
        batch_size, anchor_count = fg_mask.shape
        coarse_all = assigned["coarse_px"]
        coarse = coarse_all[fg_mask]
        target = assigned["target_bboxes"][fg_mask]
        raw_residual = assigned["pred_refine"][fg_mask]
        predicted_gates = rd.build_predicted_gates(coarse_all, args.ar_threshold, args.short_threshold)
        current_gate = predicted_gates["current"][fg_mask]
        gt_gate, target_short, target_ar = rd.build_gt_gate(target, args.ar_threshold, args.short_threshold)

        coarse_short = coarse[:, 2:4].amin(dim=-1).clamp_min(1e-6)
        coarse_long = coarse[:, 2:4].amax(dim=-1).clamp_min(1e-6)
        target_long = target[:, 2:4].amax(dim=-1).clamp_min(1e-6)
        exact_target_delta = torch.stack(
            (torch.log(target_short.clamp_min(1e-6) / coarse_short), torch.log(target_long / coarse_long)), dim=-1
        )
        clipped_target_delta = exact_target_delta.clamp(-target_limit, target_limit)
        positive_stride = assigned["stride_tensor"].reshape(1, -1).expand(batch_size, -1)[fg_mask].float()
        coarse_long_angle = coarse[:, 4] + torch.where(
            coarse[:, 3] > coarse[:, 2],
            torch.full_like(coarse[:, 4], math.pi / 2.0),
            torch.zeros_like(coarse[:, 4]),
        )
        target_long_angle = target[:, 4] + torch.where(
            target[:, 3] > target[:, 2],
            torch.full_like(target[:, 4], math.pi / 2.0),
            torch.zeros_like(target[:, 4]),
        )
        center_difference = target[:, 0:2] - coarse[:, 0:2]
        cos_long = torch.cos(coarse_long_angle)
        sin_long = torch.sin(coarse_long_angle)
        center_long = (center_difference[:, 0] * cos_long + center_difference[:, 1] * sin_long) / positive_stride
        center_short = (-center_difference[:, 0] * sin_long + center_difference[:, 1] * cos_long) / positive_stride
        center_target = torch.stack((center_long, center_short), dim=-1)
        long_angle_difference = target_long_angle - coarse_long_angle
        angle_target = (
            0.5 * torch.atan2(torch.sin(2.0 * long_angle_difference), torch.cos(2.0 * long_angle_difference))
            / (math.pi / 2.0)
        ).unsqueeze(-1)
        all_gate = torch.ones_like(current_gate)
        scale_all_box = rd.apply_short_long_target(coarse, exact_target_delta, all_gate, refine_head)
        scale_gt_box = rd.apply_short_long_target(coarse, exact_target_delta, gt_gate, refine_head)
        scale_center_box = scale_all_box.clone()
        scale_center_box[:, 0:2] = target[:, 0:2]
        center_all_box = coarse.clone()
        center_all_box[:, 0:2] = target[:, 0:2]
        aligned_target_theta = coarse[:, 4] + angle_target.squeeze(-1) * (math.pi / 2.0)
        angle_all_box = coarse.clone()
        angle_all_box[:, 4] = aligned_target_theta
        scale_angle_box = scale_all_box.clone()
        scale_angle_box[:, 4] = aligned_target_theta
        v22_ungated_box = rd.apply_refine(coarse, raw_residual, all_gate, 1.0, delta_max, refine_head)
        v22_current_box = rd.apply_refine(coarse, raw_residual, current_gate, 1.0, delta_max, refine_head)

        coarse_iou = paired_iou(coarse, target)
        v22_ungated_iou = paired_iou(v22_ungated_box, target)
        perfect_quality_gate = current_gate & (v22_ungated_iou > coarse_iou + args.benefit_epsilon)
        v22_perfect_box = rd.apply_refine(coarse, raw_residual, perfect_quality_gate, 1.0, delta_max, refine_head)
        oracle_ious = {
            "coarse_iou": coarse_iou,
            "v22_current": paired_iou(v22_current_box, target),
            "v22_perfect_quality": paired_iou(v22_perfect_box, target),
            "scale_all": paired_iou(scale_all_box, target),
            "scale_gt_gate": paired_iou(scale_gt_box, target),
            "center_all": paired_iou(center_all_box, target),
            "angle_all": paired_iou(angle_all_box, target),
            "scale_center": paired_iou(scale_center_box, target),
            "scale_angle": paired_iou(scale_angle_box, target),
            "full_target": paired_iou(target, target),
        }

        pred_scores = assigned["pred_scores"].sigmoid().amax(dim=-1)
        reg_max = raw["boxes"].shape[1] // 4
        distribution = raw["boxes"].reshape(batch_size, 4, reg_max, anchor_count).float()
        probabilities = distribution.softmax(dim=2)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=2).mean(dim=1)
        entropy = entropy / math.log(reg_max)
        anchor_points, stride_tensor = make_anchors(raw["feats"], criterion.stride, 0.5)
        anchor_offset = assigned["coarse_grid"][..., 0:2] - anchor_points.unsqueeze(0)
        offset_norm = anchor_offset.square().sum(dim=-1).sqrt()
        short_all = coarse_all[..., 2:4].amin(dim=-1).clamp_min(1e-6)
        long_all = coarse_all[..., 2:4].amax(dim=-1).clamp_min(1e-6)
        aspect_all = long_all / short_all
        theta_all = coarse_all[..., 4]
        stride_all = stride_tensor.reshape(1, -1).expand(batch_size, -1)
        state_all = torch.stack(
            (
                pred_scores,
                entropy,
                torch.log(short_all / float(args.imgsz)),
                torch.log(aspect_all.clamp_min(1.0)),
                anchor_offset[..., 0],
                anchor_offset[..., 1],
                offset_norm,
                torch.sin(2.0 * theta_all),
                torch.cos(2.0 * theta_all),
                torch.log(stride_all / float(args.imgsz)),
            ),
            dim=-1,
        ).float()

        target_delta_all = torch.zeros((batch_size, anchor_count, 2), device=coarse.device, dtype=torch.float32)
        exact_scale_target_all = torch.zeros_like(target_delta_all)
        center_target_all = torch.zeros_like(target_delta_all)
        angle_target_all = torch.zeros((batch_size, anchor_count, 1), device=coarse.device, dtype=torch.float32)
        coarse_box_all = torch.zeros((batch_size, anchor_count, 5), device=coarse.device, dtype=torch.float32)
        target_box_all = torch.zeros_like(coarse_box_all)
        positive_stride_all = torch.zeros((batch_size, anchor_count, 1), device=coarse.device, dtype=torch.float32)
        oracle_gain_all = torch.zeros((batch_size, anchor_count), device=coarse.device, dtype=torch.float32)
        v22_gain_all = torch.zeros_like(oracle_gain_all)
        target_delta_all[fg_mask] = clipped_target_delta.float()
        exact_scale_target_all[fg_mask] = exact_target_delta.float()
        center_target_all[fg_mask] = center_target.float()
        angle_target_all[fg_mask] = angle_target.float()
        coarse_box_all[fg_mask] = coarse.float()
        target_box_all[fg_mask] = target.float()
        positive_stride_all[fg_mask] = positive_stride.unsqueeze(-1)
        oracle_gain_all[fg_mask] = (oracle_ious["scale_all"] - coarse_iou).float()
        v22_gain_all[fg_mask] = (v22_ungated_iou - coarse_iou).float()

        start = 0
        level_ids, _ = rd.fpn_level_ids(raw["feats"], batch_size, coarse.device)
        positive_level_ids = level_ids[fg_mask]
        positive_offset_norm = offset_norm[fg_mask]
        for level_index, feature in enumerate(raw["feats"]):
            level_count = int(feature.shape[2] * feature.shape[3])
            end = start + level_count
            level_mask = fg_mask[:, start:end]
            level_positive_count = int(level_mask.sum().item())
            if level_positive_count:
                fpn = flatten_fpn_feature(feature)[level_mask].float()
                values = {
                    "fpn": fpn,
                    "state": state_all[:, start:end, :][level_mask],
                    "target": target_delta_all[:, start:end, :][level_mask],
                    "scale_target_exact": exact_scale_target_all[:, start:end, :][level_mask],
                    "center_target": center_target_all[:, start:end, :][level_mask],
                    "angle_target": angle_target_all[:, start:end, :][level_mask],
                    "coarse_box": coarse_box_all[:, start:end, :][level_mask],
                    "target_box": target_box_all[:, start:end, :][level_mask],
                    "stride": positive_stride_all[:, start:end, :][level_mask],
                    "oracle_gain": oracle_gain_all[:, start:end][level_mask],
                    "v22_gain": v22_gain_all[:, start:end][level_mask],
                    "current_gate": predicted_gates["current"][:, start:end][level_mask],
                }
                reservoirs[f"P{level_index + 3}"].update(values)
            start = end

        if keep_eval_rows:
            eval_chunks["fpn_level"].append(positive_level_ids.detach().cpu())
            eval_chunks["target_short"].append(target_short.detach().float().cpu())
            eval_chunks["target_ar"].append(target_ar.detach().float().cpu())
            eval_chunks["anchor_offset_norm"].append(positive_offset_norm.detach().float().cpu())
            eval_chunks["current_gate"].append(current_gate.detach().cpu())
            for name, values in oracle_ious.items():
                eval_chunks[name].append(values.detach().float().cpu())

        processed_batches += 1
        seen_images += int(batch["img"].shape[0])
        if processed_batches % 100 == 0:
            print(f"  [{split}] batches={processed_batches}, images={seen_images}")

    if not processed_batches:
        raise RuntimeError(f"split={split} 没有产生可用 batch")
    merged = {
        name: torch.cat(chunks, dim=0).numpy() if chunks else np.asarray([])
        for name, chunks in eval_chunks.items()
    }
    return merged, {"split": split, "batches": processed_batches, "images": seen_images}


def build_probe(kind: str, input_dim: int, output_dim: int, hidden: int):
    if kind == "linear":
        return torch.nn.Linear(input_dim, output_dim)
    if kind == "mlp":
        return torch.nn.Sequential(torch.nn.Linear(input_dim, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, output_dim))
    raise ValueError(kind)


def select_features(data: dict[str, Any], feature_set: str):
    if feature_set == "state":
        return data["state"].float()
    if feature_set == "fpn_state":
        return torch.cat((data["fpn"].float(), data["state"].float()), dim=1)
    raise ValueError(feature_set)


def fit_probe(
    x_train,
    y_train,
    *,
    task: str,
    kind: str,
    device,
    args: argparse.Namespace,
    seed: int,
):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mean = x_train.mean(dim=0)
    std = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    model = build_probe(kind, x_train.shape[1], y_train.shape[1], args.probe_hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.probe_lr, weight_decay=1e-4)
    if task == "classification":
        positives = float(y_train.sum().item())
        negatives = float(y_train.numel() - positives)
        pos_weight = torch.tensor([negatives / max(positives, 1.0)], device=device)
    model.train()
    for _ in range(args.probe_epochs):
        order = torch.randperm(x_train.shape[0])
        for start in range(0, x_train.shape[0], args.probe_batch_size):
            indices = order[start : start + args.probe_batch_size]
            xb = ((x_train[indices] - mean) / std).to(device)
            yb = y_train[indices].to(device)
            prediction = model(xb)
            if task == "classification":
                loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction, yb, pos_weight=pos_weight)
            else:
                loss = torch.nn.functional.smooth_l1_loss(prediction, yb, beta=0.02)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model.eval(), mean, std


def predict_probe(model, mean, std, x, *, task: str, device, batch_size: int):
    outputs = []
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            xb = ((x[start : start + batch_size] - mean) / std).to(device)
            output = model(xb)
            if task == "classification":
                output = output.sigmoid()
            outputs.append(output.detach().cpu())
    return torch.cat(outputs, dim=0)


def quality_metrics_row(
    *,
    target: str,
    feature_set: str,
    model_kind: str,
    level: str,
    labels,
    scores,
    min_auc: float,
):
    label_list = labels.reshape(-1).to(torch.int64).tolist()
    score_list = scores.reshape(-1).to(torch.float64).tolist()
    prevalence = sum(label_list) / len(label_list) if label_list else math.nan
    auc = binary_roc_auc(label_list, score_list)
    ap = binary_average_precision(label_list, score_list)
    brier = sum((score - label) ** 2 for label, score in zip(label_list, score_list)) / len(label_list)
    passed = math.isfinite(auc) and math.isfinite(ap) and auc >= min_auc and ap > prevalence
    return {
        "target": target,
        "features": feature_set,
        "model": model_kind,
        "level": level,
        "n": len(label_list),
        "positives": sum(label_list),
        "prevalence": prevalence,
        "roc_auc": auc,
        "pr_auc": ap,
        "pr_auc_lift": ap - prevalence,
        "brier": brier,
        "pass": passed,
    }


def run_quality_probes(train_data, eval_data, args, device):
    rows = []
    seed_cursor = args.seed * 1000 + 100
    for target_name, gain_name, gated in (
        ("oracle_scale_benefit", "oracle_gain", False),
        ("v22_residual_benefit", "v22_gain", True),
    ):
        for feature_set in ("state", "fpn_state"):
            for kind in ("linear", "mlp"):
                aggregate_labels = []
                aggregate_scores = []
                for level in sorted(train_data):
                    train_level = train_data[level]
                    eval_level = eval_data[level]
                    train_mask = train_level["current_gate"].bool() if gated else torch.ones_like(train_level["current_gate"], dtype=torch.bool)
                    eval_mask = eval_level["current_gate"].bool() if gated else torch.ones_like(eval_level["current_gate"], dtype=torch.bool)
                    x_train = select_features(train_level, feature_set)[train_mask]
                    x_eval = select_features(eval_level, feature_set)[eval_mask]
                    y_train = (train_level[gain_name][train_mask] > args.benefit_epsilon).float().reshape(-1, 1)
                    y_eval = (eval_level[gain_name][eval_mask] > args.benefit_epsilon).float().reshape(-1, 1)
                    if x_train.shape[0] < 32 or x_eval.shape[0] < 2 or y_train.unique().numel() < 2:
                        rows.append({
                            "target": target_name, "features": feature_set, "model": kind, "level": level,
                            "n": int(y_eval.numel()), "positives": int(y_eval.sum().item()), "prevalence": float(y_eval.mean().item()) if y_eval.numel() else math.nan,
                            "roc_auc": math.nan, "pr_auc": math.nan, "pr_auc_lift": math.nan, "brier": math.nan, "pass": False,
                        })
                        continue
                    model, mean, std = fit_probe(
                        x_train, y_train, task="classification", kind=kind, device=device, args=args, seed=seed_cursor
                    )
                    seed_cursor += 1
                    scores = predict_probe(model, mean, std, x_eval, task="classification", device=device, batch_size=args.probe_batch_size)
                    rows.append(quality_metrics_row(
                        target=target_name, feature_set=feature_set, model_kind=kind, level=level,
                        labels=y_eval, scores=scores, min_auc=args.quality_min_roc_auc,
                    ))
                    aggregate_labels.append(y_eval)
                    aggregate_scores.append(scores)
                    del model
                if aggregate_labels:
                    rows.append(quality_metrics_row(
                        target=target_name, feature_set=feature_set, model_kind=kind, level="all",
                        labels=torch.cat(aggregate_labels), scores=torch.cat(aggregate_scores), min_auc=args.quality_min_roc_auc,
                    ))
    return pd.DataFrame(
        rows,
        columns=[
            "target", "features", "model", "level", "n", "positives", "prevalence",
            "roc_auc", "pr_auc", "pr_auc_lift", "brier", "pass",
        ],
    )


def pearson_correlation(first, second) -> float:
    first = first.float().reshape(-1)
    second = second.float().reshape(-1)
    if first.numel() < 2 or float(first.std(unbiased=False)) < 1e-12 or float(second.std(unbiased=False)) < 1e-12:
        return math.nan
    return float(torch.corrcoef(torch.stack((first, second)))[0, 1].item())


def regression_metric_rows(
    *,
    feature_set: str,
    model_kind: str,
    level: str,
    target,
    prediction,
    mean_baseline,
    args: argparse.Namespace,
):
    rows = []
    channel_specs = (("dshort", 0), ("dlong", 1), ("both", None))
    for channel, index in channel_specs:
        actual = target if index is None else target[:, index : index + 1]
        predicted = prediction if index is None else prediction[:, index : index + 1]
        baseline_mean = mean_baseline if index is None else mean_baseline[:, index : index + 1]
        mae = float((predicted - actual).abs().mean().item())
        zero_mae = float(actual.abs().mean().item())
        mean_mae = float((baseline_mean - actual).abs().mean().item())
        best_baseline = min(zero_mae, mean_mae)
        relative = (best_baseline - mae) / max(best_baseline, 1e-12)
        eligible = actual.abs() > args.benefit_epsilon
        direction = float(((predicted[eligible] * actual[eligible]) > 0).float().mean().item()) if eligible.any() else math.nan
        rows.append({
            "features": feature_set,
            "model": model_kind,
            "level": level,
            "channel": channel,
            "n": int(actual.numel()),
            "mae": mae,
            "zero_baseline_mae": zero_mae,
            "mean_baseline_mae": mean_mae,
            "relative_mae_improvement": relative,
            "direction_agreement": direction,
            "pearson_r": pearson_correlation(predicted, actual),
            "pass": relative >= args.residual_min_mae_improvement and direction >= args.residual_min_direction,
        })
    return rows


def run_residual_probes(train_data, eval_data, args, device):
    rows = []
    seed_cursor = args.seed * 1000 + 500
    for feature_set in ("state", "fpn_state"):
        for kind in ("linear", "mlp"):
            aggregate_target = []
            aggregate_prediction = []
            aggregate_mean_baseline = []
            for level in sorted(train_data):
                train_level = train_data[level]
                eval_level = eval_data[level]
                train_mask = train_level["current_gate"].bool()
                eval_mask = eval_level["current_gate"].bool()
                x_train = select_features(train_level, feature_set)[train_mask]
                x_eval = select_features(eval_level, feature_set)[eval_mask]
                y_train = train_level["target"][train_mask].float()
                y_eval = eval_level["target"][eval_mask].float()
                if x_train.shape[0] < 32 or x_eval.shape[0] < 2:
                    continue
                model, mean, std = fit_probe(
                    x_train, y_train, task="regression", kind=kind, device=device, args=args, seed=seed_cursor
                )
                seed_cursor += 1
                prediction = predict_probe(model, mean, std, x_eval, task="regression", device=device, batch_size=args.probe_batch_size)
                mean_baseline = y_train.mean(dim=0, keepdim=True).expand_as(y_eval)
                rows.extend(regression_metric_rows(
                    feature_set=feature_set, model_kind=kind, level=level, target=y_eval,
                    prediction=prediction, mean_baseline=mean_baseline, args=args,
                ))
                aggregate_target.append(y_eval)
                aggregate_prediction.append(prediction)
                aggregate_mean_baseline.append(mean_baseline)
                del model
            if aggregate_target:
                rows.extend(regression_metric_rows(
                    feature_set=feature_set, model_kind=kind, level="all",
                    target=torch.cat(aggregate_target), prediction=torch.cat(aggregate_prediction),
                    mean_baseline=torch.cat(aggregate_mean_baseline), args=args,
                ))
    return pd.DataFrame(
        rows,
        columns=[
            "features", "model", "level", "channel", "n", "mae", "zero_baseline_mae",
            "mean_baseline_mae", "relative_mae_improvement", "direction_agreement", "pearson_r", "pass",
        ],
    )


def summarize_oracles(eval_rows: dict[str, Any], epsilon: float):
    coarse = eval_rows["coarse_iou"]
    rows = []
    for variant in ORACLE_VARIANTS:
        refined = coarse if variant == "coarse" else eval_rows[variant]
        delta = refined - coarse
        rows.append({
            "variant": variant,
            "n": int(delta.size),
            "coarse_iou_mean": float(coarse.mean()),
            "refined_iou_mean": float(refined.mean()),
            "delta_iou_mean": float(delta.mean()),
            "delta_iou_p25": float(np.percentile(delta, 25)),
            "delta_iou_p50": float(np.percentile(delta, 50)),
            "delta_iou_p75": float(np.percentile(delta, 75)),
            "improved_ratio": float((delta > epsilon).mean()),
            "worsened_ratio": float((delta < -epsilon).mean()),
        })
    return pd.DataFrame(rows)


def add_subgroup_rows(rows, *, dimension, group, mask, coarse, signals, epsilon):
    if not mask.any():
        return
    for signal_name, refined in signals.items():
        delta = refined[mask] - coarse[mask]
        rows.append({
            "dimension": dimension,
            "group": group,
            "signal": signal_name,
            "n": int(mask.sum()),
            "coarse_iou_mean": float(coarse[mask].mean()),
            "delta_iou_mean": float(delta.mean()),
            "improved_ratio": float((delta > epsilon).mean()),
            "worsened_ratio": float((delta < -epsilon).mean()),
        })


def summarize_subgroups(eval_rows, epsilon: float):
    coarse = eval_rows["coarse_iou"]
    signals = {
        "oracle_scale": eval_rows["scale_all"],
        "v22_current": eval_rows["v22_current"],
        "v22_perfect_quality": eval_rows["v22_perfect_quality"],
    }
    rows = []
    level = eval_rows["fpn_level"]
    for index, name in enumerate(("P3", "P4", "P5")):
        add_subgroup_rows(rows, dimension="fpn_level", group=name, mask=level == index, coarse=coarse, signals=signals, epsilon=epsilon)
    specifications = (
        ("coarse_iou", coarse, ((-np.inf, 0.5, "<0.50"), (0.5, 0.7, "0.50-0.70"), (0.7, 0.8, "0.70-0.80"), (0.8, 0.9, "0.80-0.90"), (0.9, 0.95, "0.90-0.95"), (0.95, np.inf, ">=0.95"))),
        ("gt_short_side", eval_rows["target_short"], ((-np.inf, 16, "<16"), (16, 32, "16-32"), (32, 64, "32-64"), (64, np.inf, ">=64"))),
        ("gt_aspect_ratio", eval_rows["target_ar"], ((-np.inf, 10, "<=10"), (10, 30, "10-30"), (30, np.inf, ">30"))),
        ("anchor_offset_norm", eval_rows["anchor_offset_norm"], ((-np.inf, 0.5, "<0.5"), (0.5, 1, "0.5-1"), (1, 2, "1-2"), (2, np.inf, ">=2"))),
    )
    for dimension, values, bins in specifications:
        for lower, upper, name in bins:
            mask = (values >= lower) & (values < upper)
            add_subgroup_rows(rows, dimension=dimension, group=name, mask=mask, coarse=coarse, signals=signals, epsilon=epsilon)
    return pd.DataFrame(rows)


def dataframe_to_markdown(frame, columns: Iterable[str], max_rows: int = 50) -> str:
    columns = list(columns)
    if frame.empty:
        return "无数据。"
    view = frame.loc[:, columns].head(max_rows)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for values in view.itertuples(index=False, name=None):
        rendered = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                rendered.append("" if not math.isfinite(float(value)) else f"{float(value):.6f}")
            else:
                rendered.append(str(value))
        body.append("| " + " | ".join(rendered) + " |")
    return "\n".join((header, divider, *body))


def write_report(
    path: Path,
    *,
    args,
    audit_frame,
    split_stats,
    oracle_frame,
    quality_frame,
    residual_frame,
    subgroup_frame,
    decision,
    oracle_quality_pass,
    deployed_quality_pass,
    center_gain,
    angle_gain,
):
    best_quality = quality_frame[
        (quality_frame["target"] == "oracle_scale_benefit") & (quality_frame["level"] == "all")
    ].sort_values(["pass", "roc_auc", "pr_auc"], ascending=False).head(4)
    best_deployed_quality = quality_frame[
        (quality_frame["target"] == "v22_residual_benefit") & (quality_frame["level"] == "all")
    ].sort_values(["pass", "roc_auc", "pr_auc"], ascending=False).head(4)
    best_residual = residual_frame[
        (residual_frame["level"] == "all") & (residual_frame["channel"] == "both")
    ].sort_values(["pass", "relative_mae_improvement", "direction_agreement"], ascending=False).head(4)
    audit_failures = audit_frame[~audit_frame["status"].isin(("equal", "allowed"))]
    lines = [
        "# Refine V3 实现前检查报告",
        "",
        "## 1. 口径与数据边界",
        "",
        f"- 纯 CA 基准：`{args.ca_weights}`",
        f"- Refine 诊断权重：`{args.refine_weights}`",
        f"- 数据：`{args.data}`",
        f"- 输入尺寸：`{args.imgsz}`",
        f"- Probe 拟合/评估：`{args.train_split}` / `{args.eval_split}`",
        "- 本脚本未读取 test split。所有 IoU 均为匹配正样本上的 ProbIoU 机制指标，不等价于完整 mAP。",
        f"- train batches/images：`{split_stats[0]['batches']}` / `{split_stats[0]['images']}`",
        f"- val batches/images：`{split_stats[1]['batches']}` / `{split_stats[1]['images']}`",
        "",
        "## 2. CA 共享参数审计",
        "",
        f"共享及额外张量共 `{len(audit_frame)}` 项；失败 `{len(audit_failures)}` 项。审计必须为零失败，后续分析才成立。",
        "",
        "## 3. 几何自由度 oracle",
        "",
        dataframe_to_markdown(oracle_frame, ["variant", "n", "coarse_iou_mean", "refined_iou_mean", "delta_iou_mean", "improved_ratio", "worsened_ratio"]),
        "",
        f"相对 scale-only 的中心增量：`{center_gain:.6f}`；角度增量：`{angle_gain:.6f}`。预声明门槛为 `{args.extra_dof_min_gain:.6f}`。",
        "",
        "## 4. 质量可预测性",
        "",
        "通过条件：验证集 ROC AUC 不低于预声明阈值，且 PR-AUC 高于正例率基线。下表为 oracle scale 标签的聚合最佳结果。",
        "",
        dataframe_to_markdown(best_quality, ["target", "features", "model", "level", "n", "prevalence", "roc_auc", "pr_auc", "pr_auc_lift", "brier", "pass"]),
        "",
        "oracle 标签只回答理想尺度修正空间是否可识别；真正可部署的门控还必须能预测实际残差是否有益。下表为 V2.2 实际残差收益标签。",
        "",
        dataframe_to_markdown(best_deployed_quality, ["target", "features", "model", "level", "n", "prevalence", "roc_auc", "pr_auc", "pr_auc_lift", "brier", "pass"]),
        "",
        "## 5. 残差可学习性",
        "",
        "通过条件：相对零/训练均值中更强基线的 MAE 至少改善预声明比例，且方向一致率达到门槛。下表为双通道聚合结果。",
        "",
        dataframe_to_markdown(best_residual, ["features", "model", "level", "channel", "n", "mae", "zero_baseline_mae", "mean_baseline_mae", "relative_mae_improvement", "direction_agreement", "pearson_r", "pass"]),
        "",
        "## 6. 冻结的判断",
        "",
        f"- oracle scale 标签可预测：`{oracle_quality_pass}`",
        f"- V2.2 实际残差收益可预测：`{deployed_quality_pass}`",
        f"- 联合质量门槛通过：`{decision['quality_pass']}`",
        f"- 残差预测通过：`{decision['residual_pass']}`",
        f"- 建议进入 V3 Head 实现：`{decision['recommend_v3']}`",
        f"- V3 是否纳入中心残差：`{decision['include_center']}`",
        f"- V3 是否纳入角度残差：`{decision['include_angle']}`",
        "",
    ]
    if decision["recommend_v3"]:
        lines.append("质量门控和实例残差均显示出独立验证集可预测性，可以按最小自由度原则实现 V3。")
    else:
        lines.append("至少一项核心可预测性检查未通过；不建议仅凭 oracle 上界继续包装 Refine，应保留 CA 主线并停止结构扩张。")
    lines.extend([
        "",
        "## 7. 输出文件",
        "",
        "- `checkpoint_audit.csv`：逐张量 CA/Refine 一致性审计。",
        "- `oracle_dof.csv`：几何自由度及完美质量门控上界。",
        "- `quality_probe.csv`：train 拟合、val 评估的质量分类 Probe。",
        "- `residual_probe.csv`：train 拟合、val 评估的短/长边残差 Probe。",
        "- `subgroup_breakdown.csv`：按 FPN、IoU、目标尺度、长宽比和中心偏移分组。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rd, YOLO, check_det_dataset, make_anchors, select_device = bind_runtime_dependencies()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)

    print("=" * 80)
    print("Refine V3 precheck: CA audit -> train probes -> val-only evaluation")
    print(f"CA={args.ca_weights}")
    print(f"Refine={args.refine_weights}")
    print(f"data={args.data}, imgsz={args.imgsz}, train={args.train_split}, eval={args.eval_split}")
    print("=" * 80)

    ca_yolo = YOLO(str(args.ca_weights), task="obb")
    refine_yolo = YOLO(str(args.refine_weights), task="obb")
    audit_frame, audit_failures = audit_shared_checkpoint_state(ca_yolo.model, refine_yolo.model)
    audit_path = args.output_dir / "checkpoint_audit.csv"
    audit_frame.to_csv(audit_path, index=False)
    print(f"共享参数审计: failures={len(audit_failures)}, output={audit_path}")
    if audit_failures:
        preview = "\n  ".join(audit_failures[:20])
        raise RuntimeError(f"CA/Refine 共享状态不一致，停止 V3 precheck：\n  {preview}")
    del ca_yolo

    core_model = refine_yolo.model.to(device).float().eval()
    for parameter in core_model.parameters():
        parameter.requires_grad_(False)
    refine_head = rd.find_refine_head(core_model)
    if int(getattr(refine_head, "refine_version", 0)) != 2:
        raise RuntimeError("--refine-weights 必须是带 V2 marker 的 Refine V2 checkpoint")
    runtime_args = rd.read_refine_runtime_args(refine_head)
    actual_profile = str(runtime_args["refine_experiment"])
    if actual_profile != args.expect_refine_profile:
        raise RuntimeError(
            f"Refine profile 不匹配: checkpoint={actual_profile!r}, "
            f"expected={args.expect_refine_profile!r}；请检查 --refine-weights"
        )
    cfg = rd.configure_model_args(core_model, args, runtime_args)
    criterion = core_model.init_criterion()
    rd.assert_refine_runtime_args(refine_head, runtime_args)
    data_dict = check_det_dataset(args.data)
    stride = max(int(core_model.stride.max().item()), 32)

    generators = {
        split: {level: torch.Generator().manual_seed(args.seed + split_index * 100 + level_index) for level_index, level in enumerate(("P3", "P4", "P5"))}
        for split_index, split in enumerate((args.train_split, args.eval_split))
    }
    split_reservoirs = {
        split: {
            level: PriorityReservoir(args.max_probe_samples_per_level, generators[split][level])
            for level in ("P3", "P4", "P5")
        }
        for split in (args.train_split, args.eval_split)
    }
    train_loader = rd.build_split_loader(cfg, data_dict, args.train_split, args.batch, args.workers, stride)
    train_rows, train_stats = collect_split(
        split=args.train_split, loader=train_loader, core_model=core_model, criterion=criterion,
        refine_head=refine_head, rd=rd, make_anchors=make_anchors, args=args,
        reservoirs=split_reservoirs[args.train_split], keep_eval_rows=False,
    )
    del train_loader, train_rows
    eval_loader = rd.build_split_loader(cfg, data_dict, args.eval_split, args.batch, args.workers, stride)
    eval_rows, eval_stats = collect_split(
        split=args.eval_split, loader=eval_loader, core_model=core_model, criterion=criterion,
        refine_head=refine_head, rd=rd, make_anchors=make_anchors, args=args,
        reservoirs=split_reservoirs[args.eval_split], keep_eval_rows=True,
    )
    del eval_loader

    train_data = {level: reservoir.data for level, reservoir in split_reservoirs[args.train_split].items() if reservoir.data}
    eval_data = {level: reservoir.data for level, reservoir in split_reservoirs[args.eval_split].items() if reservoir.data}
    common_levels = sorted(set(train_data).intersection(eval_data))
    train_data = {level: train_data[level] for level in common_levels}
    eval_data = {level: eval_data[level] for level in common_levels}
    if not common_levels:
        raise RuntimeError("train/val 没有共同的正样本 FPN 层，无法训练 Probe")
    for split, reservoirs in split_reservoirs.items():
        summary = ", ".join(f"{level}: kept={len(res.data['target']) if res.data else 0}/seen={res.seen}" for level, res in reservoirs.items())
        print(f"  reservoir[{split}] {summary}")

    oracle_frame = summarize_oracles(eval_rows, args.benefit_epsilon)
    subgroup_frame = summarize_subgroups(eval_rows, args.benefit_epsilon)
    print("训练质量 Probe...")
    quality_frame = run_quality_probes(train_data, eval_data, args, device)
    print("训练残差 Probe...")
    residual_frame = run_residual_probes(train_data, eval_data, args, device)

    oracle_frame.to_csv(args.output_dir / "oracle_dof.csv", index=False)
    quality_frame.to_csv(args.output_dir / "quality_probe.csv", index=False)
    residual_frame.to_csv(args.output_dir / "residual_probe.csv", index=False)
    subgroup_frame.to_csv(args.output_dir / "subgroup_breakdown.csv", index=False)

    oracle_lookup = oracle_frame.set_index("variant")["delta_iou_mean"]
    center_gain = float(oracle_lookup["scale_center"] - oracle_lookup["scale_all"])
    angle_gain = float(oracle_lookup["scale_angle"] - oracle_lookup["scale_all"])
    oracle_quality_pass = bool(quality_frame[
        (quality_frame["target"] == "oracle_scale_benefit") & (quality_frame["level"] == "all")
    ]["pass"].any())
    deployed_quality_pass = bool(quality_frame[
        (quality_frame["target"] == "v22_residual_benefit") & (quality_frame["level"] == "all")
    ]["pass"].any())
    quality_pass = oracle_quality_pass and deployed_quality_pass
    residual_pass = bool(residual_frame[
        (residual_frame["level"] == "all") & (residual_frame["channel"] == "both")
    ]["pass"].any())
    decision = precheck_decision(
        quality_pass, residual_pass, center_gain, angle_gain, args.extra_dof_min_gain
    )
    report_path = args.output_dir / "refine_v3_precheck.md"
    write_report(
        report_path, args=args, audit_frame=audit_frame, split_stats=(train_stats, eval_stats),
        oracle_frame=oracle_frame, quality_frame=quality_frame, residual_frame=residual_frame,
        subgroup_frame=subgroup_frame, decision=decision, oracle_quality_pass=oracle_quality_pass,
        deployed_quality_pass=deployed_quality_pass, center_gain=center_gain, angle_gain=angle_gain,
    )
    print("=" * 80)
    print(f"quality_pass={decision['quality_pass']}, residual_pass={decision['residual_pass']}")
    print(f"recommend_v3={decision['recommend_v3']}, center={decision['include_center']}, angle={decision['include_angle']}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
