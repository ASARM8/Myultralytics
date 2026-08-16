"""Shared cloud runtime for proposal-level Refine V3 training and validation."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from ultralytics.cfg import DEFAULT_CFG, get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import nms
from ultralytics.utils.metrics import OBBMetrics, batch_probiou, probiou


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_key(path: str, pattern: str = "") -> str:
    normalized = str(Path(path).as_posix())
    if not pattern:
        return normalized
    match = re.search(pattern, Path(path).name)
    if not match:
        raise ValueError(f"group regex did not match image name: {path}")
    if match.lastindex is None:
        raise ValueError("group regex must contain at least one capture group")
    return match.group(1)


def stable_holdout(group: str, fraction: float, seed: int) -> bool:
    value = hashlib.sha256(f"{seed}:{group}".encode()).digest()[:8]
    unit = int.from_bytes(value, "big") / float(2**64)
    return unit < fraction


def split_dataset_indices(image_paths: Iterable[str], holdout_fraction: float, seed: int, pattern: str = ""):
    fit, holdout = [], []
    fit_groups, holdout_groups = set(), set()
    for index, path in enumerate(image_paths):
        group = group_key(str(path), pattern)
        if stable_holdout(group, holdout_fraction, seed):
            holdout.append(index)
            holdout_groups.add(group)
        else:
            fit.append(index)
            fit_groups.add(group)
    if not fit or not holdout:
        raise RuntimeError("deterministic group split produced an empty partition")
    if fit_groups & holdout_groups:
        raise RuntimeError("group leakage detected between train-fit and train-holdout")
    return fit, holdout, fit_groups, holdout_groups


def build_dataset(data_yaml: str, split: str, imgsz: int, batch: int, workers: int, *, rect: bool):
    data = check_det_dataset(data_yaml)
    path = data.get(split)
    if not path:
        raise ValueError(f"dataset YAML does not define split={split}")
    cfg = get_cfg(
        DEFAULT_CFG,
        overrides={
            "task": "obb",
            "data": data_yaml,
            "imgsz": imgsz,
            "batch": batch,
            "workers": workers,
            "rect": rect,
            "cache": False,
            "augment": False,
            "plots": False,
        },
    )
    dataset = build_yolo_dataset(cfg, path, batch, data, mode="val", rect=rect, stride=32)
    return dataset, data


def subset_loader(dataset, indices: list[int], batch: int, workers: int, *, shuffle: bool):
    subset = torch.utils.data.Subset(dataset, indices)
    subset.collate_fn = dataset.collate_fn
    return build_dataloader(
        subset,
        batch=batch,
        workers=workers,
        shuffle=shuffle,
        rank=-1,
        drop_last=False,
        pin_memory=True,
    )


def full_loader(dataset, batch: int, workers: int, *, shuffle: bool = False):
    return build_dataloader(
        dataset,
        batch=batch,
        workers=workers,
        shuffle=shuffle,
        rank=-1,
        drop_last=False,
        pin_memory=True,
    )


class FrozenCAExtractor:
    """Run a frozen CA model and capture P2/P3 features used by V3."""

    def __init__(self, core_model, *, device, nc: int, conf: float, nms_iou: float, max_det: int, amp: bool):
        self.core_model = core_model
        self.device = device
        self.nc = int(nc)
        self.conf = float(conf)
        self.nms_iou = float(nms_iou)
        self.max_det = int(max_det)
        self.amp = bool(amp and device.type == "cuda")
        self.cache: dict[str, torch.Tensor] = {}
        self.handles = [
            core_model.model[2].register_forward_hook(self._capture_p2),
            core_model.model[-1].register_forward_pre_hook(self._capture_head_inputs),
        ]

    def _capture_p2(self, _module, _inputs, output):
        self.cache["p2"] = output.detach()

    def _capture_head_inputs(self, _module, inputs):
        features = inputs[0]
        if not isinstance(features, (tuple, list)) or len(features) < 1:
            raise TypeError("OBB head pre-hook did not receive an FPN feature list")
        self.cache["p3"] = features[0].detach()

    def infer(self, batch: dict[str, Any]):
        images = batch["img"].to(self.device, non_blocking=True).float() / 255.0
        self.cache.clear()
        # V3 is trained on frozen CA features. ``no_grad`` prevents a CA
        # autograd graph while still returning ordinary tensors that a
        # downstream trainable convolution may save for its own backward.
        # ``inference_mode`` is invalid here because its tensors cannot be
        # saved for backward by the V3 projection layers.
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.amp,
        ):
            outputs = self.core_model(images)
        inference = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if not isinstance(inference, torch.Tensor) or inference.ndim != 3:
            raise TypeError(f"expected CA inference tensor [B,C,N], received {type(inference)}")
        if "p2" not in self.cache or "p3" not in self.cache:
            raise RuntimeError("P2/P3 hooks did not fire")
        detections = nms.non_max_suppression(
            inference,
            self.conf,
            self.nms_iou,
            nc=self.nc,
            multi_label=True,
            agnostic=False,
            max_det=self.max_det,
            rotated=True,
        )
        formatted = [
            {
                "bboxes": torch.cat((item[:, :4], item[:, -1:]), dim=1),
                "conf": item[:, 4],
                "cls": item[:, 5],
            }
            for item in detections
        ]
        return images, self.cache["p2"], self.cache["p3"], formatted

    def infer_channels(self, imgsz: int) -> tuple[int, int]:
        dummy = torch.zeros(1, 3, imgsz, imgsz, device=self.device)
        self.cache.clear()
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.amp,
        ):
            self.core_model(dummy)
        if "p2" not in self.cache or "p3" not in self.cache:
            raise RuntimeError("unable to infer P2/P3 channels")
        return int(self.cache["p2"].shape[1]), int(self.cache["p3"].shape[1])

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def pad_detections(detections: list[dict[str, torch.Tensor]]):
    batch = len(detections)
    maximum = max((item["bboxes"].shape[0] for item in detections), default=0)
    device = detections[0]["bboxes"].device if detections else torch.device("cpu")
    dtype = detections[0]["bboxes"].dtype if detections else torch.float32
    boxes = torch.zeros(batch, maximum, 5, device=device, dtype=dtype)
    scores = torch.zeros(batch, maximum, device=device, dtype=dtype)
    classes = torch.zeros(batch, maximum, device=device, dtype=torch.long)
    valid = torch.zeros(batch, maximum, device=device, dtype=torch.bool)
    for image_index, item in enumerate(detections):
        count = item["bboxes"].shape[0]
        if not count:
            continue
        boxes[image_index, :count] = item["bboxes"]
        scores[image_index, :count] = item["conf"]
        classes[image_index, :count] = item["cls"].long()
        valid[image_index, :count] = True
    return boxes, scores, classes, valid


def periodic_angle_distance(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    difference = first - second
    return 0.5 * torch.atan2(torch.sin(2.0 * difference), torch.cos(2.0 * difference)).abs()


def align_equivalent_targets(proposals: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    alternative = targets.clone()
    alternative[..., 2] = targets[..., 3]
    alternative[..., 3] = targets[..., 2]
    alternative[..., 4] = targets[..., 4] + math.pi / 2.0
    use_alternative = periodic_angle_distance(alternative[..., 4], proposals[..., 4]) < periodic_angle_distance(
        targets[..., 4], proposals[..., 4]
    )
    return torch.where(use_alternative[..., None], alternative, targets)


def greedy_class_match(proposals, proposal_cls, targets, target_cls, minimum_iou: float):
    if not proposals.shape[0] or not targets.shape[0]:
        empty = torch.empty(0, dtype=torch.long, device=proposals.device)
        return empty, empty
    iou = batch_probiou(targets, proposals)
    valid = target_cls[:, None].long() == proposal_cls[None, :].long()
    pairs = torch.nonzero(valid & (iou >= minimum_iou), as_tuple=False)
    if not pairs.shape[0]:
        empty = torch.empty(0, dtype=torch.long, device=proposals.device)
        return empty, empty
    pair_iou = iou[pairs[:, 0], pairs[:, 1]]
    order = pair_iou.argsort(descending=True)
    used_gt, used_proposal = set(), set()
    selected_gt, selected_proposal = [], []
    for index in order.tolist():
        gt_index = int(pairs[index, 0])
        proposal_index = int(pairs[index, 1])
        if gt_index in used_gt or proposal_index in used_proposal:
            continue
        used_gt.add(gt_index)
        used_proposal.add(proposal_index)
        selected_gt.append(gt_index)
        selected_proposal.append(proposal_index)
    return (
        torch.tensor(selected_proposal, dtype=torch.long, device=proposals.device),
        torch.tensor(selected_gt, dtype=torch.long, device=proposals.device),
    )


def build_supervision(
    refiner,
    boxes: torch.Tensor,
    classes: torch.Tensor,
    valid: torch.Tensor,
    batch: dict[str, Any],
    image_size: tuple[int, int],
    *,
    match_iou: float,
    quality_min_gain: float,
    tiny_reference_px: float,
    tiny_weight_floor: float,
):
    image_height, image_width = image_size
    matched = torch.zeros_like(valid)
    quality_target = torch.zeros_like(valid, dtype=boxes.dtype)
    exact_target = boxes.new_zeros((*boxes.shape[:2], 4))
    clipped_target = boxes.new_zeros((*boxes.shape[:2], 4))
    geometry_weight = boxes.new_zeros(boxes.shape[:2])
    oracle_gain = boxes.new_zeros(boxes.shape[:2])
    matched_target_boxes = boxes.new_zeros(boxes.shape)
    for image_index in range(boxes.shape[0]):
        count = int(valid[image_index].sum().item())
        mask = batch["batch_idx"] == image_index
        target_boxes = batch["bboxes"][mask].to(boxes.device).float()
        target_cls = batch["cls"][mask].reshape(-1).to(boxes.device)
        if target_boxes.shape[0]:
            target_boxes[:, :4] *= target_boxes.new_tensor((image_width, image_height, image_width, image_height))
        proposal_index, gt_index = greedy_class_match(
            boxes[image_index, :count], classes[image_index, :count], target_boxes, target_cls, match_iou
        )
        if not proposal_index.numel():
            continue
        proposal = boxes[image_index, proposal_index]
        target = align_equivalent_targets(proposal, target_boxes[gt_index])
        encoded = refiner.encode_targets(proposal, target)
        clipped = refiner.clip_target(encoded)
        oracle = refiner.apply_residual(proposal, clipped)
        coarse_iou = probiou(proposal, target).reshape(-1)
        bounded_iou = probiou(oracle, target).reshape(-1)
        gain = bounded_iou - coarse_iou
        gt_short = target[:, 2:4].amin(dim=1)
        resolution_weight = (gt_short / float(tiny_reference_px)).clamp(
            min=float(tiny_weight_floor), max=1.0
        )
        matched[image_index, proposal_index] = True
        exact_target[image_index, proposal_index] = encoded
        clipped_target[image_index, proposal_index] = clipped
        geometry_weight[image_index, proposal_index] = resolution_weight
        oracle_gain[image_index, proposal_index] = gain
        quality_target[image_index, proposal_index] = (gain >= quality_min_gain).to(boxes.dtype)
        matched_target_boxes[image_index, proposal_index] = target
    return {
        "matched": matched,
        "quality_target": quality_target,
        "exact_target": exact_target,
        "clipped_target": clipped_target,
        "geometry_weight": geometry_weight,
        "oracle_gain": oracle_gain,
        "target_boxes": matched_target_boxes,
    }


def prediction_to_raw(boxes, scores, classes, nc: int):
    count = boxes.shape[0]
    # ``non_max_suppression`` treats any tensor whose final dimension is 6 as
    # an end-to-end BNC result. A one-class OBB raw tensor is BCN with exactly
    # six channels, so K=6 would otherwise create the ambiguous shape [1,6,6].
    # Add one zero-confidence sentinel column; it is filtered before NMS and
    # leaves every real proposal unchanged.
    safe_count = count + int(count == 6)
    raw = boxes.new_zeros((4 + nc + 1, safe_count))
    if count:
        raw[:4, :count] = boxes[:, :4].T
        index = torch.arange(count, device=boxes.device)
        raw[4 + classes.long(), index] = scores
        raw[4 + nc, :count] = boxes[:, 4]
    return raw


def rerun_rotated_nms(boxes, scores, classes, nc: int, conf: float, nms_iou: float, max_det: int):
    raw = prediction_to_raw(boxes, scores, classes, nc)
    output = nms.non_max_suppression(
        raw.unsqueeze(0),
        conf,
        nms_iou,
        nc=nc,
        multi_label=True,
        agnostic=False,
        max_det=max_det,
        rotated=True,
    )[0]
    return {
        "bboxes": torch.cat((output[:, :4], output[:, -1:]), dim=1),
        "conf": output[:, 4],
        "cls": output[:, 5],
    }


def match_predictions(pred_classes, true_classes, iou, thresholds):
    correct = np.zeros((pred_classes.shape[0], len(thresholds)), dtype=bool)
    correct_class = (true_classes[:, None] == pred_classes).cpu().numpy()
    iou = iou.cpu().numpy() * correct_class
    for threshold_index, threshold in enumerate(thresholds):
        matches = np.array(np.nonzero(iou >= threshold)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), threshold_index] = True
    return correct


def update_metric(metric: OBBMetrics, prediction: dict[str, torch.Tensor], batch: dict[str, Any], image_index: int):
    mask = batch["batch_idx"] == image_index
    target_cls = batch["cls"][mask].reshape(-1).to(prediction["bboxes"].device)
    target_boxes = batch["bboxes"][mask].to(prediction["bboxes"].device).float()
    image_height, image_width = int(batch["img"].shape[2]), int(batch["img"].shape[3])
    if target_boxes.shape[0]:
        target_boxes[:, :4] *= target_boxes.new_tensor((image_width, image_height, image_width, image_height))
    pred_cls = prediction["cls"]
    if target_boxes.shape[0] and pred_cls.shape[0]:
        correct = match_predictions(
            pred_cls,
            target_cls,
            batch_probiou(target_boxes, prediction["bboxes"]),
            torch.linspace(0.5, 0.95, 10, device=pred_cls.device).tolist(),
        )
    else:
        correct = np.zeros((pred_cls.shape[0], 10), dtype=bool)
    no_prediction = pred_cls.shape[0] == 0
    cls_numpy = target_cls.cpu().numpy()
    metric.update_stats(
        {
            "tp": correct,
            "target_cls": cls_numpy,
            "target_img": np.unique(cls_numpy),
            "conf": np.zeros(0) if no_prediction else prediction["conf"].float().cpu().numpy(),
            "pred_cls": np.zeros(0) if no_prediction else pred_cls.float().cpu().numpy(),
        }
    )


def metric_summary(metric: OBBMetrics, variant: str) -> dict[str, Any]:
    values = metric.results_dict
    all_ap = metric.box.all_ap
    row = {
        "variant": variant,
        "precision": float(values["metrics/precision(B)"]),
        "recall": float(values["metrics/recall(B)"]),
        "map50": float(values["metrics/mAP50(B)"]),
        "map50_95": float(values["metrics/mAP50-95(B)"]),
    }
    row.update(
        {
            f"ap{threshold}": float(all_ap[:, index].mean()) if len(all_ap) else math.nan
            for index, threshold in enumerate(range(50, 100, 5))
        }
    )
    return row


def evaluate_refiner(
    extractor: FrozenCAExtractor,
    refiner,
    loader,
    names,
    thresholds: tuple[float, ...],
    *,
    amp: bool,
    identity_tolerance: float = 5e-4,
):
    variants = ("coarse", "roundtrip") + tuple(f"quality_{threshold:.3f}" for threshold in thresholds)
    metrics = {name: OBBMetrics(names=names) for name in variants}
    gate_counts = {threshold: 0 for threshold in thresholds}
    valid_count = 0
    residual_values = []
    quality_values = []
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
            quality = output["quality_logit"].float().sigmoid().squeeze(-1)
            residual = output["residual"].float()
            valid_count += int(valid.sum().item())
            if valid.any():
                # Boolean indexing consumes the first two dimensions. Slice the
                # residual channels afterwards to avoid ambiguous mixed indexing.
                residual_values.append(residual[valid][..., :2].detach().cpu())
                quality_values.append(quality[valid].detach().cpu())
            for image_index, detection in enumerate(detections):
                update_metric(metrics["coarse"], detection, batch, image_index)
                count = int(valid[image_index].sum().item())
                roundtrip = rerun_rotated_nms(
                    boxes[image_index, :count],
                    scores[image_index, :count],
                    classes[image_index, :count],
                    extractor.nc,
                    extractor.conf,
                    extractor.nms_iou,
                    extractor.max_det,
                )
                update_metric(metrics["roundtrip"], roundtrip, batch, image_index)
                for threshold in thresholds:
                    gate = quality[image_index, :count] >= threshold
                    gate_counts[threshold] += int(gate.sum().item())
                    gated_residual = residual[image_index, :count] * gate[:, None].to(residual.dtype)
                    refined_boxes = refiner.apply_residual(boxes[image_index, :count].float(), gated_residual)
                    refined = rerun_rotated_nms(
                        refined_boxes,
                        scores[image_index, :count],
                        classes[image_index, :count],
                        extractor.nc,
                        extractor.conf,
                        extractor.nms_iou,
                        extractor.max_det,
                    )
                    update_metric(metrics[f"quality_{threshold:.3f}"], refined, batch, image_index)
    rows = []
    for variant, metric in metrics.items():
        metric.process(plot=False)
        rows.append(metric_summary(metric, variant))
        metric.clear_stats()
    lookup = {row["variant"]: row for row in rows}
    identity_error = abs(lookup["roundtrip"]["map50_95"] - lookup["coarse"]["map50_95"])
    if identity_error > identity_tolerance:
        raise RuntimeError(
            f"post-NMS roundtrip identity failed: abs delta mAP50-95={identity_error:.6f} > {identity_tolerance:.6f}"
        )
    for row in rows:
        row["delta_map50_95_vs_coarse"] = row["map50_95"] - lookup["coarse"]["map50_95"]
        if row["variant"].startswith("quality_"):
            threshold = float(row["variant"].split("_", 1)[1])
            row["gate_ratio"] = gate_counts[threshold] / valid_count if valid_count else math.nan
        else:
            row["gate_ratio"] = 0.0
    residual = torch.cat(residual_values) if residual_values else torch.empty(0, 2)
    quality = torch.cat(quality_values) if quality_values else torch.empty(0)
    diagnostics = {
        "valid_proposals": valid_count,
        "roundtrip_identity_abs_delta": identity_error,
        "short_residual_mean": float(residual[:, 0].mean().item()) if residual.numel() else math.nan,
        "short_residual_std": float(residual[:, 0].std(unbiased=False).item()) if residual.numel() else math.nan,
        "short_residual_p05": float(torch.quantile(residual[:, 0], 0.05).item()) if residual.numel() else math.nan,
        "short_residual_p95": float(torch.quantile(residual[:, 0], 0.95).item()) if residual.numel() else math.nan,
        "long_residual_mean": float(residual[:, 1].mean().item()) if residual.numel() else math.nan,
        "long_residual_std": float(residual[:, 1].std(unbiased=False).item()) if residual.numel() else math.nan,
        "long_residual_p05": float(torch.quantile(residual[:, 1], 0.05).item()) if residual.numel() else math.nan,
        "long_residual_p95": float(torch.quantile(residual[:, 1], 0.95).item()) if residual.numel() else math.nan,
        "quality_probability_mean": float(quality.mean().item()) if quality.numel() else math.nan,
        "quality_probability_std": float(quality.std(unbiased=False).item()) if quality.numel() else math.nan,
        "quality_probability_p05": float(torch.quantile(quality, 0.05).item()) if quality.numel() else math.nan,
        "quality_probability_p95": float(torch.quantile(quality, 0.95).item()) if quality.numel() else math.nan,
    }
    return rows, diagnostics
