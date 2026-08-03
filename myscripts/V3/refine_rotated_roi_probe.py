"""Compare point-state and rotated-ROI probes for proposal-level OBB refinement.

The pure Coverage-Aware checkpoint supplies immutable post-NMS proposals and
frozen feature maps.  Probe hyperparameters are selected on an image-grouped
holdout subset of ``train``; ``val`` is evaluated once afterwards.  The test
split is deliberately unavailable.

This is an architecture diagnostic, not a production Refine head.  A positive
ROI result shows that spatially aligned features contain usable instance-level
geometry information that the previous pointwise head could not access.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CANONICAL_CA_WEIGHTS = Path("/root/autodl-tmp/work-dirs/yolo11_obb_640_811_ca/weights/best.pt")
TARGET_NAMES = ("dshort", "dlong", "dcenter_long", "dcenter_short")
STATE_NAMES = (
    "confidence",
    "log_short_norm",
    "log_long_norm",
    "log_aspect_ratio",
    "center_x_norm",
    "center_y_norm",
    "sin_2theta",
    "cos_2theta",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ca-weights", type=Path, default=CANONICAL_CA_WEIGHTS)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--train-split", choices=("train",), default="train")
    parser.add_argument("--eval-split", choices=("val",), default="val")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    parser.add_argument("--max-eval-samples", type=int, default=12000)
    parser.add_argument("--match-iou", type=float, default=0.30)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--roi-height", type=int, default=5)
    parser.add_argument("--roi-width", type=int, default=24)
    parser.add_argument("--projection-channels", type=int, default=16)
    parser.add_argument("--long-context", type=float, default=1.20)
    parser.add_argument("--short-context", type=float, default=4.0)
    parser.add_argument("--min-short-context-px", type=float, default=16.0)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument(
        "--group-regex",
        default="",
        help="Optional regex; capture group 1 defines a scene group. Default grouping is one image per group.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-epochs", type=int, default=12)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--scale-target-limit", type=float, default=0.50)
    parser.add_argument("--center-target-limit", type=float, default=1.00)
    parser.add_argument("--direction-deadzone", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.imgsz != 640:
        parser.error("Innovation-one experiments are fixed at imgsz=640")
    if args.eval_split != "val":
        parser.error("The Refine design must be frozen before test is used")
    if not args.ca_weights.is_file():
        parser.error(f"CA weights not found: {args.ca_weights}")
    positive = (
        "batch",
        "max_train_samples",
        "max_eval_samples",
        "max_det",
        "roi_height",
        "roi_width",
        "projection_channels",
        "epochs",
        "early_stop_patience",
        "early_stop_min_epochs",
        "probe_batch_size",
        "hidden",
    )
    if any(int(getattr(args, name)) <= 0 for name in positive):
        parser.error(f"These arguments must be positive: {positive}")
    if args.workers < 0 or args.max_train_batches < 0 or args.max_eval_batches < 0:
        parser.error("workers and max-*-batches must be non-negative")
    if not 0.05 <= args.holdout_fraction <= 0.5:
        parser.error("--holdout-fraction must be in [0.05, 0.5]")
    if not 0.0 <= args.match_iou <= 1.0 or not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.nms_iou <= 1.0:
        parser.error("IoU and confidence thresholds must be in [0, 1]")
    if args.long_context <= 0 or args.short_context <= 0 or args.min_short_context_px <= 0:
        parser.error("ROI context dimensions must be positive")
    if args.scale_target_limit <= 0 or args.center_target_limit <= 0 or args.lr <= 0:
        parser.error("target limits and learning rate must be positive")
    if args.early_stop_min_epochs > args.epochs:
        parser.error("--early-stop-min-epochs cannot exceed --epochs")
    if args.group_regex:
        try:
            pattern = re.compile(args.group_regex)
        except re.error as error:
            parser.error(f"invalid --group-regex: {error}")
        if pattern.groups < 1:
            parser.error("--group-regex must contain capture group 1")


def group_key(path: str, pattern: str = "") -> str:
    """Map an image path to a leakage-safe image or scene group."""
    normalized = str(Path(path).as_posix())
    if not pattern:
        return normalized
    match = re.search(pattern, Path(path).name)
    if not match:
        raise ValueError(f"group regex did not match image name: {path}")
    return match.group(1)


def grouped_fit_holdout_indices(groups: list[str], holdout_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    """Split complete groups, never rows, into train-fit and train-holdout."""
    unique = sorted(set(groups))
    if len(unique) < 2:
        raise ValueError("at least two image/scene groups are required")
    random.Random(seed).shuffle(unique)
    holdout_count = max(1, min(len(unique) - 1, round(len(unique) * holdout_fraction)))
    holdout_groups = set(unique[:holdout_count])
    fit = [index for index, group in enumerate(groups) if group not in holdout_groups]
    holdout = [index for index, group in enumerate(groups) if group in holdout_groups]
    if not fit or not holdout:
        raise RuntimeError("grouped split produced an empty partition")
    return fit, holdout


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class TensorReservoir:
    capacity: int
    torch: Any
    generator: Any
    tensors: dict[str, Any] | None = None
    groups: list[str] | None = None
    priorities: Any | None = None
    seen: int = 0

    def update(self, values: dict[str, Any], groups: list[str]) -> None:
        if not values:
            return
        length = int(next(iter(values.values())).shape[0])
        if length == 0:
            return
        if len(groups) != length or any(int(value.shape[0]) != length for value in values.values()):
            raise ValueError("reservoir fields and groups must be row-aligned")
        cpu = {name: value.detach().cpu() for name, value in values.items()}
        priorities = self.torch.rand(length, generator=self.generator)
        self.seen += length
        if self.tensors is None:
            combined = cpu
            combined_groups = list(groups)
            combined_priorities = priorities
        else:
            combined = {name: self.torch.cat((self.tensors[name], value), dim=0) for name, value in cpu.items()}
            combined_groups = self.groups + list(groups)
            combined_priorities = self.torch.cat((self.priorities, priorities), dim=0)
        if combined_priorities.numel() > self.capacity:
            keep = self.torch.topk(combined_priorities, self.capacity, largest=True, sorted=False).indices
            combined = {name: value[keep] for name, value in combined.items()}
            combined_groups = [combined_groups[index] for index in keep.tolist()]
            combined_priorities = combined_priorities[keep]
        self.tensors = combined
        self.groups = combined_groups
        self.priorities = combined_priorities


def align_equivalent_targets(torch, proposals, targets):
    alternative = targets.clone()
    alternative[:, 2] = targets[:, 3]
    alternative[:, 3] = targets[:, 2]
    alternative[:, 4] = targets[:, 4] + math.pi / 2.0

    def distance(first, second):
        difference = first - second
        return 0.5 * torch.atan2(torch.sin(2.0 * difference), torch.cos(2.0 * difference)).abs()

    use_alternative = distance(alternative[:, 4], proposals[:, 4]) < distance(targets[:, 4], proposals[:, 4])
    return torch.where(use_alternative[:, None], alternative, targets)


def greedy_class_match(torch, batch_probiou, proposals, proposal_cls, targets, target_cls, minimum_iou: float):
    """Return a one-to-one, class-aware greedy proposal/GT assignment."""
    if not proposals.shape[0] or not targets.shape[0]:
        empty = torch.empty(0, dtype=torch.long, device=proposals.device)
        return empty, empty
    iou = batch_probiou(targets, proposals)
    valid = target_cls[:, None].long() == proposal_cls[None, :].long()
    pairs = torch.nonzero(valid & (iou >= minimum_iou), as_tuple=False)
    if not pairs.shape[0]:
        empty = torch.empty(0, dtype=torch.long, device=proposals.device)
        return empty, empty
    scores = iou[pairs[:, 0], pairs[:, 1]]
    order = scores.argsort(descending=True)
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    chosen_gt: list[int] = []
    chosen_pred: list[int] = []
    for index in order.tolist():
        gt_index = int(pairs[index, 0])
        pred_index = int(pairs[index, 1])
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        chosen_gt.append(gt_index)
        chosen_pred.append(pred_index)
    return (
        torch.tensor(chosen_pred, dtype=torch.long, device=proposals.device),
        torch.tensor(chosen_gt, dtype=torch.long, device=proposals.device),
    )


def fixed_projection(torch, channels: int, output_channels: int, seed: int, device, dtype):
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(output_channels, channels, generator=generator, dtype=torch.float32)
    matrix /= math.sqrt(max(channels, 1))
    return matrix.to(device=device, dtype=dtype)


def rotated_roi_sample(torch, functional, feature, image_indices, boxes, *, output_channels: int, height: int, width: int,
                       image_height: int, image_width: int, long_context: float, short_context: float,
                       min_short_context_px: float, seed: int):
    """Sample proposal-aligned strips and apply a deterministic channel projection."""
    selected = feature[image_indices]
    box_width, box_height, angle = boxes[:, 2], boxes[:, 3], boxes[:, 4]
    short = torch.minimum(box_width, box_height)
    long = torch.maximum(box_width, box_height)
    short_is_width = box_width <= box_height
    long_angle = torch.where(short_is_width, angle + math.pi / 2.0, angle)
    long_extent = long * long_context
    short_extent = torch.maximum(short * short_context, torch.full_like(short, min_short_context_px))
    u = torch.linspace(-0.5, 0.5, width, device=boxes.device, dtype=boxes.dtype)
    v = torch.linspace(-0.5, 0.5, height, device=boxes.device, dtype=boxes.dtype)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    local_long = uu[None] * long_extent[:, None, None]
    local_short = vv[None] * short_extent[:, None, None]
    cos = long_angle.cos()[:, None, None]
    sin = long_angle.sin()[:, None, None]
    x = boxes[:, 0, None, None] + local_long * cos - local_short * sin
    y = boxes[:, 1, None, None] + local_long * sin + local_short * cos
    grid = torch.stack((2.0 * x / image_width - 1.0, 2.0 * y / image_height - 1.0), dim=-1)
    crop = functional.grid_sample(selected, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    projection = fixed_projection(torch, crop.shape[1], output_channels, seed, crop.device, crop.dtype)
    return torch.einsum("oc,nchw->nohw", projection, crop)


def proposal_state(torch, boxes, confidence, image_height: int, image_width: int):
    width, height, angle = boxes[:, 2], boxes[:, 3], boxes[:, 4]
    short = torch.minimum(width, height).clamp_min(1e-3)
    long = torch.maximum(width, height).clamp_min(1e-3)
    return torch.stack(
        (
            confidence,
            torch.log(short / 640.0),
            torch.log(long / 640.0),
            torch.log(long / short),
            boxes[:, 0] / image_width,
            boxes[:, 1] / image_height,
            torch.sin(2.0 * angle),
            torch.cos(2.0 * angle),
        ),
        dim=1,
    )


def geometry_targets(torch, proposals, targets):
    proposal_short = proposals[:, 2:4].amin(dim=1).clamp_min(1e-3)
    proposal_long = proposals[:, 2:4].amax(dim=1).clamp_min(1e-3)
    target_short = targets[:, 2:4].amin(dim=1).clamp_min(1e-3)
    target_long = targets[:, 2:4].amax(dim=1).clamp_min(1e-3)
    scale = torch.stack((torch.log(target_short / proposal_short), torch.log(target_long / proposal_long)), dim=1)
    short_is_width = proposals[:, 2] <= proposals[:, 3]
    long_angle = torch.where(short_is_width, proposals[:, 4] + math.pi / 2.0, proposals[:, 4])
    difference = targets[:, :2] - proposals[:, :2]
    long_offset = difference[:, 0] * long_angle.cos() + difference[:, 1] * long_angle.sin()
    short_offset = -difference[:, 0] * long_angle.sin() + difference[:, 1] * long_angle.cos()
    center = torch.stack((long_offset / proposal_long, short_offset / proposal_short.clamp_min(4.0)), dim=1)
    return torch.cat((scale, center), dim=1)


def collect_split(*, torch, functional, nms_module, batch_probiou, loader, core_model, hooks, args, split: str,
                  reservoir: TensorReservoir, max_batches: int, device):
    processed_images = 0
    processed_batches = 0
    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        images = batch["img"].to(device, non_blocking=True).float() / 255.0
        hooks.clear()
        with torch.inference_mode():
            outputs = core_model(images)
        inference = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if "p2" not in hooks or "p3" not in hooks:
            raise RuntimeError("feature hooks did not capture P2/P3")
        detections = nms_module.non_max_suppression(
            inference,
            args.conf,
            args.nms_iou,
            nc=len(core_model.names),
            multi_label=True,
            agnostic=False,
            max_det=args.max_det,
            rotated=True,
        )
        image_height, image_width = int(images.shape[2]), int(images.shape[3])
        all_boxes = []
        all_targets = []
        all_confidence = []
        all_image_indices = []
        all_groups: list[str] = []
        for image_index, detection in enumerate(detections):
            mask = batch["batch_idx"] == image_index
            target = batch["bboxes"][mask].to(images.device).float()
            target_cls = batch["cls"][mask].reshape(-1).to(images.device)
            if target.shape[0]:
                target[:, :4] *= target.new_tensor((image_width, image_height, image_width, image_height))
            boxes = torch.cat((detection[:, :4], detection[:, -1:]), dim=1)
            pred_index, gt_index = greedy_class_match(
                torch, batch_probiou, boxes, detection[:, 5], target, target_cls, args.match_iou
            )
            if not pred_index.numel():
                continue
            matched_boxes = boxes[pred_index]
            matched_targets = align_equivalent_targets(torch, matched_boxes, target[gt_index])
            all_boxes.append(matched_boxes)
            all_targets.append(matched_targets)
            all_confidence.append(detection[pred_index, 4])
            all_image_indices.append(torch.full_like(pred_index, image_index))
            key = group_key(str(batch["im_file"][image_index]), args.group_regex)
            all_groups.extend([key] * pred_index.numel())
        if all_boxes:
            boxes = torch.cat(all_boxes)
            targets = torch.cat(all_targets)
            confidence = torch.cat(all_confidence)
            image_indices = torch.cat(all_image_indices)
            p2 = rotated_roi_sample(
                torch,
                functional,
                hooks["p2"],
                image_indices,
                boxes,
                output_channels=args.projection_channels,
                height=args.roi_height,
                width=args.roi_width,
                image_height=image_height,
                image_width=image_width,
                long_context=args.long_context,
                short_context=args.short_context,
                min_short_context_px=args.min_short_context_px,
                seed=args.seed + 1001,
            )
            p3 = rotated_roi_sample(
                torch,
                functional,
                hooks["p3"],
                image_indices,
                boxes,
                output_channels=args.projection_channels,
                height=args.roi_height,
                width=args.roi_width,
                image_height=image_height,
                image_width=image_width,
                long_context=args.long_context,
                short_context=args.short_context,
                min_short_context_px=args.min_short_context_px,
                seed=args.seed + 2003,
            )
            reservoir.update(
                {
                    "state": proposal_state(torch, boxes, confidence, image_height, image_width).float(),
                    "roi": torch.cat((p2, p3), dim=1).half(),
                    "target_exact": geometry_targets(torch, boxes, targets).float(),
                    "coarse": boxes.float(),
                    "gt": targets.float(),
                },
                all_groups,
            )
        processed_images += len(batch["im_file"])
        processed_batches += 1
    if reservoir.tensors is None:
        raise RuntimeError(f"no matched proposal samples collected from split={split}")
    return {"split": split, "batches": processed_batches, "images": processed_images, "seen": reservoir.seen,
            "kept": len(reservoir.groups)}


def standardize(tensor, dimensions):
    mean = tensor.mean(dim=dimensions, keepdim=True)
    std = tensor.std(dim=dimensions, unbiased=False, keepdim=True).clamp_min(1e-5)
    return mean, std


def build_model(torch, kind: str, state_dim: int, roi_channels: int, output_dim: int, hidden: int):
    if kind == "state":
        return torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, output_dim),
        )

    class ROIProbe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(roi_channels, 64, 3, padding=1),
                torch.nn.SiLU(),
                torch.nn.Conv2d(64, 64, 3, stride=(1, 2), padding=1),
                torch.nn.SiLU(),
                torch.nn.AdaptiveAvgPool2d((2, 6)),
            )
            self.regressor = torch.nn.Sequential(
                torch.nn.Linear(64 * 2 * 6 + state_dim, hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden, output_dim),
            )

        def forward(self, state, roi):
            encoded = self.encoder(roi).flatten(1)
            return self.regressor(torch.cat((encoded, state), dim=1))

    return ROIProbe()


def fit_probe(torch, functional, kind: str, train, fit_indices, holdout_indices, args, device):
    fit_indices = torch.as_tensor(fit_indices, dtype=torch.long)
    holdout_indices = torch.as_tensor(holdout_indices, dtype=torch.long)
    state_mean, state_std = standardize(train["state"][fit_indices].float(), (0,))
    roi_mean, roi_std = standardize(train["roi"][fit_indices].float(), (0, 2, 3))
    exact = train["target_exact"].float()
    limits = exact.new_tensor((args.scale_target_limit, args.scale_target_limit, args.center_target_limit,
                               args.center_target_limit))
    target = torch.maximum(torch.minimum(exact, limits), -limits)
    target_mean, target_std = standardize(target[fit_indices], (0,))
    target_std = target_std.clamp_min(1e-4)
    torch.manual_seed(args.seed + (0 if kind == "state" else 10000))
    model = build_model(torch, kind, train["state"].shape[1], train["roi"].shape[1], target.shape[1], args.hidden).to(device)
    final_layer = model[-1] if kind == "state" else model.regressor[-1]
    torch.nn.init.zeros_(final_layer.weight)
    torch.nn.init.zeros_(final_layer.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed + 29)
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    stale = 0

    def forward_indices(indices):
        state = ((train["state"][indices].float() - state_mean) / state_std).to(device)
        if kind == "state":
            return model(state)
        roi = ((train["roi"][indices].float() - roi_mean) / roi_std).to(device)
        return model(state, roi)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = fit_indices[torch.randperm(fit_indices.numel(), generator=generator)]
        for start in range(0, order.numel(), args.probe_batch_size):
            indices = order[start : start + args.probe_batch_size]
            prediction = forward_indices(indices)
            normalized_target = ((target[indices] - target_mean) / target_std).to(device)
            loss = functional.smooth_l1_loss(prediction, normalized_target, beta=0.2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        losses = []
        with torch.inference_mode():
            for start in range(0, holdout_indices.numel(), args.probe_batch_size):
                indices = holdout_indices[start : start + args.probe_batch_size]
                prediction = forward_indices(indices)
                normalized_target = ((target[indices] - target_mean) / target_std).to(device)
                losses.append(functional.l1_loss(prediction, normalized_target).item() * indices.numel())
        holdout_loss = sum(losses) / holdout_indices.numel()
        if holdout_loss < best_loss - 1e-6:
            best_loss = holdout_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch >= args.early_stop_min_epochs and stale >= args.early_stop_patience:
            break
    if best_state is None:
        raise RuntimeError(f"{kind} probe did not produce a checkpoint")
    model.load_state_dict(best_state)
    return {
        "kind": kind,
        "model": model.eval(),
        "state_mean": state_mean,
        "state_std": state_std,
        "roi_mean": roi_mean,
        "roi_std": roi_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "limits": limits,
        "best_epoch": best_epoch,
        "holdout_normalized_mae": best_loss,
    }


def predict_probe(torch, fitted, data, device, batch_size: int):
    outputs = []
    with torch.inference_mode():
        for start in range(0, data["state"].shape[0], batch_size):
            state = ((data["state"][start : start + batch_size].float() - fitted["state_mean"]) /
                     fitted["state_std"]).to(device)
            if fitted["kind"] == "state":
                normalized = fitted["model"](state)
            else:
                roi = ((data["roi"][start : start + batch_size].float() - fitted["roi_mean"]) /
                       fitted["roi_std"]).to(device)
                normalized = fitted["model"](state, roi)
            prediction = normalized.cpu() * fitted["target_std"] + fitted["target_mean"]
            outputs.append(torch.maximum(torch.minimum(prediction, fitted["limits"]), -fitted["limits"]))
    return torch.cat(outputs)


def apply_geometry(torch, coarse, prediction, family: str):
    output = coarse.clone()
    if family in {"scale", "combined"}:
        short_is_width = coarse[:, 2] <= coarse[:, 3]
        delta_width = torch.where(short_is_width, prediction[:, 0], prediction[:, 1])
        delta_height = torch.where(short_is_width, prediction[:, 1], prediction[:, 0])
        output[:, 2] *= torch.exp(delta_width)
        output[:, 3] *= torch.exp(delta_height)
    if family in {"center", "combined"}:
        short = coarse[:, 2:4].amin(dim=1).clamp_min(4.0)
        long = coarse[:, 2:4].amax(dim=1).clamp_min(1e-3)
        short_is_width = coarse[:, 2] <= coarse[:, 3]
        long_angle = torch.where(short_is_width, coarse[:, 4] + math.pi / 2.0, coarse[:, 4])
        long_offset = prediction[:, 2] * long
        short_offset = prediction[:, 3] * short
        output[:, 0] += long_offset * long_angle.cos() - short_offset * long_angle.sin()
        output[:, 1] += long_offset * long_angle.sin() + short_offset * long_angle.cos()
    return output


def evaluation_rows(torch, probiou, fitted, data, split: str, device, args):
    prediction = predict_probe(torch, fitted, data, device, args.probe_batch_size)
    exact = data["target_exact"].float()
    coarse = data["coarse"].float()
    gt = data["gt"].float()
    coarse_iou = probiou(coarse, gt).reshape(-1)
    rows = []
    for family, indices in (("scale", (0, 1)), ("center", (2, 3)), ("combined", (0, 1, 2, 3))):
        chosen = torch.as_tensor(indices, dtype=torch.long)
        pred_family = prediction[:, chosen]
        target_family = exact[:, chosen]
        active = target_family.abs() > args.direction_deadzone
        direction = ((pred_family.sign() == target_family.sign()) & active).sum() / active.sum().clamp_min(1)
        refined = apply_geometry(torch, coarse, prediction, family)
        refined_iou = probiou(refined, gt).reshape(-1)
        delta = refined_iou - coarse_iou
        zero_mae = float(target_family.abs().mean().item())
        mae = float((pred_family - target_family).abs().mean().item())
        rows.append(
            {
                "model": fitted["kind"],
                "split": split,
                "family": family,
                "samples": int(coarse.shape[0]),
                "target_zero_mae": zero_mae,
                "prediction_mae": mae,
                "mae_improvement": (zero_mae - mae) / zero_mae if zero_mae else math.nan,
                "direction_accuracy": float(direction.item()),
                "coarse_probiou": float(coarse_iou.mean().item()),
                "refined_probiou": float(refined_iou.mean().item()),
                "delta_probiou": float(delta.mean().item()),
                "improved_ratio": float((delta > 1e-7).float().mean().item()),
                "worsened_ratio": float((delta < -1e-7).float().mean().item()),
                "best_epoch": fitted["best_epoch"],
                "holdout_normalized_mae": fitted["holdout_normalized_mae"],
            }
        )
    oracle = apply_geometry(torch, coarse, exact, "combined")
    oracle_iou = probiou(oracle, gt).reshape(-1)
    rows.append(
        {
            "model": "exact_scale_center_oracle",
            "split": split,
            "family": "combined",
            "samples": int(coarse.shape[0]),
            "target_zero_mae": math.nan,
            "prediction_mae": 0.0,
            "mae_improvement": math.nan,
            "direction_accuracy": 1.0,
            "coarse_probiou": float(coarse_iou.mean().item()),
            "refined_probiou": float(oracle_iou.mean().item()),
            "delta_probiou": float((oracle_iou - coarse_iou).mean().item()),
            "improved_ratio": float((oracle_iou - coarse_iou > 1e-7).float().mean().item()),
            "worsened_ratio": float((oracle_iou - coarse_iou < -1e-7).float().mean().item()),
            "best_epoch": 0,
            "holdout_normalized_mae": math.nan,
        }
    )
    return rows


def subset(torch, data: dict[str, Any], indices: list[int]):
    selected = torch.as_tensor(indices, dtype=torch.long)
    return {name: value[selected] for name, value in data.items()}


def write_report(path: Path, args, collection_rows, metrics, train_groups, fit_groups, holdout_groups) -> None:
    val_rows = [row for row in metrics if row["split"] == "val" and row["family"] == "combined"]
    lines = [
        "# Rotated ROI Refine Probe 报告",
        "",
        "该诊断使用纯 CA 的 post-NMS proposal。Probe 只在 train-fit 上训练、在按图像/场景分组的 train-holdout 上选择轮次，最后才评价 val。",
        "",
        f"- CA 权重：`{args.ca_weights}`",
        f"- imgsz：{args.imgsz}",
        f"- proposal 匹配阈值：{args.match_iou}",
        f"- 分组规则：`{args.group_regex or '完整图像路径'}`",
        f"- train groups / fit / holdout：{train_groups} / {fit_groups} / {holdout_groups}",
        "",
        "## 样本采集",
        "",
        "| split | images | matched seen | reservoir kept |",
        "|---|---:|---:|---:|",
    ]
    for row in collection_rows:
        lines.append(f"| {row['split']} | {row['images']} | {row['seen']} | {row['kept']} |")
    lines.extend(
        [
            "",
            "## Val 上 scale+center 结果",
            "",
            "| model | MAE improvement | direction | ΔProbIoU | improved | worsened |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in val_rows:
        lines.append(
            f"| {row['model']} | {row['mae_improvement']:.6f} | {row['direction_accuracy']:.6f} | "
            f"{row['delta_probiou']:+.6f} | {row['improved_ratio']:.6f} | {row['worsened_ratio']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 解释规则",
            "",
            "- `state` 是不含空间纹理的 proposal 状态基线；`roi` 额外读取 P2/P3 旋转对齐条带。",
            "- 若 ROI 在 train-holdout 与 val 都稳定优于 state，说明重新设计空间对齐 Refine 有依据。",
            "- 若两者都失败，只能说明当前 proposal、ROI 尺度和监督组合不足，不能推出所有 Refine 都不可行。",
            "- exact oracle 是上限，不代表网络能自动达到；本脚本不生成停止 Refine 的决定。",
            "- 本脚本不读取 test，避免用测试集选择结构、阈值或训练轮次。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn.functional as functional

    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG, get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import nms
    from ultralytics.utils.metrics import batch_probiou, probiou
    from ultralytics.utils.torch_utils import select_device

    device = select_device(args.device)
    wrapper = YOLO(str(args.ca_weights), task="obb")
    core_model = wrapper.model.to(device).float().eval()
    if any(hasattr(module, "cv5") for module in core_model.modules()):
        raise RuntimeError("--ca-weights must be the pure CA checkpoint, not CA+Refine")

    hooks: dict[str, Any] = {}

    def capture_p2(_module, _inputs, output):
        hooks["p2"] = output.detach()

    def capture_head_inputs(_module, inputs):
        features = inputs[0]
        hooks["p3"] = features[0].detach()

    handles = [core_model.model[2].register_forward_hook(capture_p2), core_model.model[-1].register_forward_pre_hook(capture_head_inputs)]
    try:
        data = check_det_dataset(args.data)
        cfg = get_cfg(
            DEFAULT_CFG,
            overrides={
                "task": "obb",
                "data": args.data,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "device": args.device,
                "workers": args.workers,
                "rect": True,
                "cache": False,
                "augment": False,
                "plots": False,
            },
        )

        def loader(split):
            path = data.get(split)
            if not path:
                raise ValueError(f"dataset YAML does not define split={split}")
            dataset = build_yolo_dataset(cfg, path, args.batch, data, mode="val", rect=True, stride=32)
            return build_dataloader(dataset, args.batch, args.workers, shuffle=False, rank=-1, drop_last=False,
                                    pin_memory=True)

        train_reservoir = TensorReservoir(args.max_train_samples, torch, torch.Generator().manual_seed(args.seed + 41))
        val_reservoir = TensorReservoir(args.max_eval_samples, torch, torch.Generator().manual_seed(args.seed + 43))
        collection = [
            collect_split(
                torch=torch,
                functional=functional,
                nms_module=nms,
                batch_probiou=batch_probiou,
                loader=loader(args.train_split),
                core_model=core_model,
                hooks=hooks,
                args=args,
                split=args.train_split,
                reservoir=train_reservoir,
                max_batches=args.max_train_batches,
                device=device,
            ),
            collect_split(
                torch=torch,
                functional=functional,
                nms_module=nms,
                batch_probiou=batch_probiou,
                loader=loader(args.eval_split),
                core_model=core_model,
                hooks=hooks,
                args=args,
                split=args.eval_split,
                reservoir=val_reservoir,
                max_batches=args.max_eval_batches,
                device=device,
            ),
        ]
    finally:
        for handle in handles:
            handle.remove()

    fit_indices, holdout_indices = grouped_fit_holdout_indices(
        train_reservoir.groups, args.holdout_fraction, args.seed
    )
    train = train_reservoir.tensors
    validation = val_reservoir.tensors
    metrics = []
    for kind in ("state", "roi"):
        fitted = fit_probe(torch, functional, kind, train, fit_indices, holdout_indices, args, device)
        metrics.extend(evaluation_rows(torch, probiou, fitted, subset(torch, train, fit_indices), "train_fit", device, args))
        metrics.extend(
            evaluation_rows(torch, probiou, fitted, subset(torch, train, holdout_indices), "train_holdout", device, args)
        )
        metrics.extend(evaluation_rows(torch, probiou, fitted, validation, "val", device, args))
        checkpoint = {
            "kind": kind,
            "model": fitted["model"].state_dict(),
            "state_mean": fitted["state_mean"],
            "state_std": fitted["state_std"],
            "roi_mean": fitted["roi_mean"],
            "roi_std": fitted["roi_std"],
            "target_mean": fitted["target_mean"],
            "target_std": fitted["target_std"],
            "limits": fitted["limits"],
            "best_epoch": fitted["best_epoch"],
            "config": vars(args),
        }
        torch.save(checkpoint, args.output_dir / f"{kind}_probe.pt")

    exact = train["target_exact"].float()
    limits = exact.new_tensor((args.scale_target_limit, args.scale_target_limit, args.center_target_limit,
                               args.center_target_limit))
    clipped = exact.abs() > limits
    target_rows = []
    for index, name in enumerate(TARGET_NAMES):
        values = exact[:, index]
        target_rows.append(
            {
                "target": name,
                "mean": float(values.mean().item()),
                "std": float(values.std(unbiased=False).item()),
                "p05": float(torch.quantile(values, 0.05).item()),
                "p50": float(torch.quantile(values, 0.50).item()),
                "p95": float(torch.quantile(values, 0.95).item()),
                "clipped_ratio": float(clipped[:, index].float().mean().item()),
            }
        )
    write_csv(args.output_dir / "probe_metrics.csv", metrics)
    write_csv(args.output_dir / "target_distribution.csv", target_rows)
    train_groups = set(train_reservoir.groups)
    fit_groups = {train_reservoir.groups[index] for index in fit_indices}
    holdout_groups = {train_reservoir.groups[index] for index in holdout_indices}
    if fit_groups & holdout_groups:
        raise RuntimeError("group leakage detected between train-fit and train-holdout")
    manifest = {
        "ca_weights": str(args.ca_weights),
        "data": args.data,
        "imgsz": args.imgsz,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "test_used": False,
        "group_regex": args.group_regex,
        "train_group_count": len(train_groups),
        "fit_group_count": len(fit_groups),
        "holdout_group_count": len(holdout_groups),
        "group_overlap": len(fit_groups & holdout_groups),
        "collection": collection,
        "state_features": STATE_NAMES,
        "target_channels": TARGET_NAMES,
        "roi_features": "P2/P3 rotated-aligned strips with deterministic channel projection",
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    write_report(
        args.output_dir / "rotated_roi_probe_report.md",
        args,
        collection,
        metrics,
        len(train_groups),
        len(fit_groups),
        len(holdout_groups),
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
