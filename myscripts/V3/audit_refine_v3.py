"""Audit whether a frozen Refine V3 gain is real, aligned, and mechanism-specific.

This entry point is intentionally diagnostic. It never opens ``test`` and it
never selects a checkpoint or quality threshold on ``val``. The checkpoint and
the train-holdout-selected threshold are frozen inputs. The script evaluates
counterfactual controls that preserve scores/classes while changing only the
geometry residual or its assignment to proposals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable


CANONICAL_CA_WEIGHTS = "/root/autodl-tmp/work-dirs/yolo11_obb_640_811_ca/weights/best.pt"
VARIANTS = (
    "coarse",
    "roundtrip",
    "gate_off",
    "selected_gate",
    "selected_no_renms",
    "all_refine",
    "short_only",
    "long_only",
    "mean_residual_selected",
    "mean_residual_all",
    "residual_shuffle",
    "quality_shuffle",
    "spatial_shuffle",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ca-weights", default=CANONICAL_CA_WEIGHTS)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--imgsz", type=int, choices=(640,), default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use autocast. The canonical truth audit defaults to FP32; use --amp only for a precision sensitivity run.",
    )
    parser.add_argument("--shuffle-seed", type=int, default=20250301)
    parser.add_argument("--expected-ca-map50-95", type=float, default=0.45413)
    parser.add_argument("--baseline-tolerance", type=float, default=0.002)
    parser.add_argument("--identity-tolerance", type=float, default=5e-4)
    parser.add_argument("--expected-refined-map50-95", type=float, default=None)
    parser.add_argument("--refined-tolerance", type=float, default=0.002)
    parser.add_argument("--output-dir", required=True)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if Path(args.ca_weights).as_posix() != Path(CANONICAL_CA_WEIGHTS).as_posix():
        parser.error(f"V3 truth audit is locked to the canonical CA checkpoint: {CANONICAL_CA_WEIGHTS}")
    if args.batch <= 1:
        parser.error("--batch must be at least 2 because the spatial-shuffle control operates across proposals in a batch")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if not 0.0 <= args.expected_ca_map50_95 <= 1.0:
        parser.error("--expected-ca-map50-95 must be in [0, 1]")
    if args.expected_refined_map50_95 is not None and not 0.0 <= args.expected_refined_map50_95 <= 1.0:
        parser.error("--expected-refined-map50-95 must be in [0, 1]")
    if min(args.baseline_tolerance, args.identity_tolerance, args.refined_tolerance) <= 0.0:
        parser.error("all tolerances must be positive")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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


def stable_seed(text: str, seed: int) -> int:
    """Return a process-independent 63-bit seed."""
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()[:8]
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def binary_auc(scores: Iterable[float], labels: Iterable[int]) -> float:
    """Compute tie-aware ROC AUC without requiring scipy or sklearn."""
    pairs = sorted((float(score), int(label)) for score, label in zip(scores, labels))
    positives = sum(label == 1 for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return math.nan
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        rank_sum += average_rank * sum(label == 1 for _, label in pairs[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def pearson(values_a: Iterable[float], values_b: Iterable[float]) -> float:
    pairs = [(float(a), float(b)) for a, b in zip(values_a, values_b)]
    if len(pairs) < 2:
        return math.nan
    mean_a = sum(a for a, _ in pairs) / len(pairs)
    mean_b = sum(b for _, b in pairs) / len(pairs)
    covariance = sum((a - mean_a) * (b - mean_b) for a, b in pairs)
    variance_a = sum((a - mean_a) ** 2 for a, _ in pairs)
    variance_b = sum((b - mean_b) ** 2 for _, b in pairs)
    denominator = math.sqrt(variance_a * variance_b)
    return covariance / denominator if denominator else math.nan


def binary_summary(scores: list[float], labels: list[int], threshold: float) -> dict[str, Any]:
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    predicted = [score >= threshold for score in scores]
    tp = sum(prediction and label == 1 for prediction, label in zip(predicted, labels))
    fp = sum(prediction and label == 0 for prediction, label in zip(predicted, labels))
    fn = sum(not prediction and label == 1 for prediction, label in zip(predicted, labels))
    tn = sum(not prediction and label == 0 for prediction, label in zip(predicted, labels))
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else math.nan
    brier = sum((score - label) ** 2 for score, label in zip(scores, labels)) / len(scores) if scores else math.nan
    return {
        "count": len(scores),
        "positive_count": sum(labels),
        "positive_ratio": sum(labels) / len(labels) if labels else math.nan,
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / len(labels) if labels else math.nan,
        "brier": brier,
        "roc_auc": binary_auc(scores, labels),
    }


def assign_bin(value: float, edges: tuple[float, ...]) -> str:
    for edge in edges:
        if value < edge:
            return f"<{edge:g}"
    return f">={edges[-1]:g}"


def summarize_rows(rows: list[dict[str, Any]], dimension: str, bins: tuple[float, ...]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(assign_bin(float(row[dimension]), bins), []).append(row)
    ordered_labels = [f"<{edge:g}" for edge in bins] + [f">={bins[-1]:g}"]
    result = []
    for label in ordered_labels:
        selected = grouped.get(label, [])
        if not selected:
            continue
        deltas = [float(row["selected_delta_iou"]) for row in selected]
        result.append(
            {
                "dimension": dimension,
                "bin": label,
                "count": len(selected),
                "coarse_iou_mean": sum(float(row["coarse_iou"]) for row in selected) / len(selected),
                "selected_iou_mean": sum(float(row["selected_iou"]) for row in selected) / len(selected),
                "delta_iou_mean": sum(deltas) / len(deltas),
                "improved_ratio": sum(delta > 1e-6 for delta in deltas) / len(deltas),
                "worsened_ratio": sum(delta < -1e-6 for delta in deltas) / len(deltas),
                "gain_ge_0_002_ratio": sum(delta >= 0.002 for delta in deltas) / len(deltas),
                "loss_le_minus_0_002_ratio": sum(delta <= -0.002 for delta in deltas) / len(deltas),
                "gate_ratio": sum(int(row["gate"]) for row in selected) / len(selected),
            }
        )
    return result


def _metric_delta(lookup: dict[str, dict[str, Any]], variant: str, reference: str = "coarse") -> float:
    return float(lookup[variant]["map50_95"]) - float(lookup[reference]["map50_95"])


def write_report(path: Path, audit: dict[str, Any], metrics: list[dict[str, Any]]) -> None:
    lookup = {row["variant"]: row for row in metrics}
    lines = [
        "# Refine V3 完整真实性审计",
        "",
        f"- checkpoint epoch：{audit['checkpoint_epoch']}",
        f"- train-holdout 预选 quality threshold：{audit['quality_threshold']:.3f}",
        f"- 精度模式：{'AMP' if audit['amp'] else 'FP32'}；batch={audit['batch']}",
        f"- hard integrity：{audit['hard_integrity_pass']}",
        "- test：未读取。",
        "",
        "## 完整检测指标",
        "",
        "| variant | mAP50-95 | Δ vs coarse | AP75 | AP90 | AP95 | gate ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['variant']} | {row['map50_95']:.6f} | {row['delta_map50_95_vs_coarse']:+.6f} | "
            f"{row['ap75']:.6f} | {row['ap90']:.6f} | {row['ap95']:.6f} | {row['gate_ratio']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 机制差值",
            "",
            f"- selected − all-refine：{_metric_delta(lookup, 'selected_gate', 'all_refine'):+.6f}",
            f"- selected − mean-residual-selected：{_metric_delta(lookup, 'selected_gate', 'mean_residual_selected'):+.6f}",
            f"- selected − residual-shuffle：{_metric_delta(lookup, 'selected_gate', 'residual_shuffle'):+.6f}",
            f"- selected − quality-shuffle：{_metric_delta(lookup, 'selected_gate', 'quality_shuffle'):+.6f}",
            f"- selected − spatial-shuffle：{_metric_delta(lookup, 'selected_gate', 'spatial_shuffle'):+.6f}",
            f"- selected − no-reNMS：{_metric_delta(lookup, 'selected_gate', 'selected_no_renms'):+.6f}",
            "",
            "## 解释边界",
            "",
            "- oracle、匹配 IoU 和标签只用于离线诊断与计分，不参与任何 variant 的推理门控。",
            "- mean residual 来自 checkpoint 内冻结的 train-holdout 诊断，不从 val 标签估计。",
            "- shuffle 对照保持边际分布，破坏 proposal 与残差、quality 或空间特征的对应关系。",
            "- 本报告验证单 checkpoint 的真实性和机制，不替代多种子配对试验。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    omp_threads = os.environ.get("OMP_NUM_THREADS", "")
    if not omp_threads.isdigit() or int(omp_threads) <= 0:
        os.environ["OMP_NUM_THREADS"] = "1"

    import numpy as np
    import torch

    from ultralytics import YOLO
    from ultralytics.nn.modules.refine_v3 import OBBProposalRefinerV3
    from ultralytics.utils.metrics import OBBMetrics, probiou
    from ultralytics.utils.torch_utils import select_device

    from .runtime import (
        FrozenCAExtractor,
        align_equivalent_targets,
        build_dataset,
        full_loader,
        greedy_class_match,
        metric_summary,
        pad_detections,
        rerun_rotated_nms,
        sha256_file,
        update_metric,
    )

    random.seed(args.shuffle_seed)
    np.random.seed(args.shuffle_seed)
    torch.manual_seed(args.shuffle_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.shuffle_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    output_dir = Path(args.output_dir)
    if (output_dir / "run_manifest.json").exists():
        raise FileExistsError(f"audit output already exists: {output_dir}; use a new directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint)
    ca_path = Path(args.ca_weights)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"V3 checkpoint not found: {checkpoint_path}")
    if not ca_path.is_file():
        raise FileNotFoundError(f"canonical CA checkpoint not found: {ca_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 1 or checkpoint.get("architecture") != "OBBProposalRefinerV3":
        raise RuntimeError("unsupported or non-V3 checkpoint format")
    selection = checkpoint.get("holdout_selection") or {}
    threshold = selection.get("threshold")
    if threshold is None:
        raise RuntimeError("checkpoint does not contain a train-holdout-selected quality threshold")
    threshold = float(threshold)
    training_args = checkpoint.get("arguments") or {}
    if training_args.get("output_dir") is None:
        raise RuntimeError("checkpoint is missing frozen training arguments")
    ca_hash = sha256_file(ca_path)
    if checkpoint.get("ca_sha256") != ca_hash:
        raise RuntimeError("CA SHA256 differs from the checkpoint training source")

    device = select_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
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

    proposal_conf = float(training_args.get("proposal_conf", 0.01))
    nms_iou = float(training_args.get("nms_iou", 0.70))
    max_det = int(training_args.get("max_det", 300))
    match_iou = float(training_args.get("match_iou", 0.30))
    quality_min_gain = float(training_args.get("quality_min_gain", 0.002))
    dataset, data = build_dataset(args.data, args.split, args.imgsz, args.batch, args.workers, rect=True)
    loader = full_loader(dataset, args.batch, args.workers)
    names = getattr(ca_model, "names", data["names"])
    nc = len(names)
    extractor = FrozenCAExtractor(
        ca_model,
        device=device,
        nc=nc,
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
            raise RuntimeError(f"feature channels differ: checkpoint={expected_channels}, observed={observed_channels}")
        refiner = OBBProposalRefinerV3(**model_config).to(device).float().eval()
        refiner.load_state_dict(checkpoint["model_state"], strict=True)

        diagnostics = selection.get("diagnostics") or {}
        if "short_residual_mean" not in diagnostics or "long_residual_mean" not in diagnostics:
            raise RuntimeError("checkpoint selection is missing frozen holdout residual means")
        holdout_mean = torch.tensor(
            [float(diagnostics["short_residual_mean"]), float(diagnostics["long_residual_mean"]), 0.0, 0.0],
            device=device,
            dtype=torch.float32,
        )

        metrics = {variant: OBBMetrics(names=names) for variant in VARIANTS}
        gate_counts = {variant: 0 for variant in VARIANTS}
        valid_count = 0
        proposal_rows: list[dict[str, Any]] = []

        def permutation(count: int, key: str, tensor_device):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(stable_seed(key, args.shuffle_seed))
            return torch.randperm(count, generator=generator).to(tensor_device)

        def prediction(box_values, score_values, class_values, *, rerun: bool):
            if not rerun:
                return {"bboxes": box_values, "conf": score_values, "cls": class_values}
            return rerun_rotated_nms(box_values, score_values, class_values, nc, proposal_conf, nms_iou, max_det)

        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                images, p2, p3, detections = extractor.infer(batch)
                boxes, scores, classes, valid = pad_detections(detections)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    output = refiner(p2, p3, boxes, scores, images.shape[2:], valid)

                flat_valid = valid.reshape(-1)
                spatial_key = "|".join(str(item) for item in batch.get("im_file", ()))

                def shuffle_spatial(_module, _inputs, encoded):
                    indices = torch.where(flat_valid.to(encoded.device))[0]
                    if indices.numel() <= 1:
                        return encoded
                    order = permutation(indices.numel(), f"spatial:{batch_index}:{spatial_key}", encoded.device)
                    shuffled = encoded.clone()
                    shuffled[indices] = encoded[indices[order]]
                    return shuffled

                handle = refiner.roi_encoder.register_forward_hook(shuffle_spatial)
                try:
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        spatial_output = refiner(p2, p3, boxes, scores, images.shape[2:], valid)
                finally:
                    handle.remove()

                residual = output["residual"].float()
                quality = output["quality_logit"].float().sigmoid().squeeze(-1)
                spatial_residual = spatial_output["residual"].float()
                spatial_quality = spatial_output["quality_logit"].float().sigmoid().squeeze(-1)
                valid_count += int(valid.sum().item())

                for image_index, coarse_detection in enumerate(detections):
                    count = int(valid[image_index].sum().item())
                    image_boxes = boxes[image_index, :count].float()
                    image_scores = scores[image_index, :count].float()
                    image_classes = classes[image_index, :count]
                    image_residual = residual[image_index, :count]
                    image_quality = quality[image_index, :count]
                    selected_gate = image_quality >= threshold
                    all_gate = torch.ones_like(selected_gate)
                    zero_gate = torch.zeros_like(selected_gate)
                    image_name = str(batch.get("im_file", [f"batch{batch_index}_image{image_index}"])[image_index])
                    order = permutation(count, f"proposal:{image_name}", image_boxes.device) if count else None
                    shuffled_residual = image_residual[order] if count else image_residual
                    shuffled_quality = image_quality[order] if count else image_quality
                    shuffled_gate = shuffled_quality >= threshold
                    spatial_gate = spatial_quality[image_index, :count] >= threshold

                    mean_residual = holdout_mean.expand(count, -1)
                    short_residual = image_residual.clone()
                    short_residual[:, 1:] = 0.0
                    long_residual = image_residual.clone()
                    long_residual[:, 0] = 0.0
                    long_residual[:, 2:] = 0.0

                    variant_specs = {
                        "roundtrip": (image_residual, zero_gate, True),
                        "gate_off": (image_residual, zero_gate, True),
                        "selected_gate": (image_residual, selected_gate, True),
                        "selected_no_renms": (image_residual, selected_gate, False),
                        "all_refine": (image_residual, all_gate, True),
                        "short_only": (short_residual, selected_gate, True),
                        "long_only": (long_residual, selected_gate, True),
                        "mean_residual_selected": (mean_residual, selected_gate, True),
                        "mean_residual_all": (mean_residual, all_gate, True),
                        "residual_shuffle": (shuffled_residual, selected_gate, True),
                        "quality_shuffle": (image_residual, shuffled_gate, True),
                        "spatial_shuffle": (
                            spatial_residual[image_index, :count],
                            spatial_gate,
                            True,
                        ),
                    }
                    update_metric(metrics["coarse"], coarse_detection, batch, image_index)
                    for variant, (variant_residual, gate, rerun) in variant_specs.items():
                        gated = variant_residual * gate[:, None].to(variant_residual.dtype)
                        refined_boxes = refiner.apply_residual(image_boxes, gated)
                        refined_prediction = prediction(refined_boxes, image_scores, image_classes, rerun=rerun)
                        update_metric(metrics[variant], refined_prediction, batch, image_index)
                        gate_counts[variant] += int(gate.sum().item())

                    target_mask = batch["batch_idx"] == image_index
                    target_boxes = batch["bboxes"][target_mask].to(device).float()
                    target_classes = batch["cls"][target_mask].reshape(-1).to(device)
                    if target_boxes.shape[0]:
                        target_boxes[:, :4] *= target_boxes.new_tensor(
                            (images.shape[3], images.shape[2], images.shape[3], images.shape[2])
                        )
                    proposal_index, target_index = greedy_class_match(
                        image_boxes, image_classes, target_boxes, target_classes, match_iou
                    )
                    if not proposal_index.numel():
                        continue
                    matched_boxes = image_boxes[proposal_index]
                    matched_targets = align_equivalent_targets(matched_boxes, target_boxes[target_index])
                    matched_residual = image_residual[proposal_index]
                    matched_gate = selected_gate[proposal_index]
                    encoded = refiner.encode_targets(matched_boxes, matched_targets)
                    oracle_residual = refiner.clip_target(encoded)
                    coarse_iou = probiou(matched_boxes, matched_targets).reshape(-1)
                    all_iou = probiou(refiner.apply_residual(matched_boxes, matched_residual), matched_targets).reshape(-1)
                    selected_iou = probiou(
                        refiner.apply_residual(
                            matched_boxes, matched_residual * matched_gate[:, None].to(matched_residual.dtype)
                        ),
                        matched_targets,
                    ).reshape(-1)
                    short_iou = probiou(
                        refiner.apply_residual(matched_boxes, short_residual[proposal_index] * matched_gate[:, None]),
                        matched_targets,
                    ).reshape(-1)
                    long_iou = probiou(
                        refiner.apply_residual(matched_boxes, long_residual[proposal_index] * matched_gate[:, None]),
                        matched_targets,
                    ).reshape(-1)
                    oracle_iou = probiou(refiner.apply_residual(matched_boxes, oracle_residual), matched_targets).reshape(-1)
                    matched_short = matched_boxes[:, 2:4].amin(dim=1)
                    matched_long = matched_boxes[:, 2:4].amax(dim=1)
                    for local_index, proposal_slot in enumerate(proposal_index.tolist()):
                        all_gain = float(all_iou[local_index] - coarse_iou[local_index])
                        selected_gain = float(selected_iou[local_index] - coarse_iou[local_index])
                        oracle_gain = float(oracle_iou[local_index] - coarse_iou[local_index])
                        proposal_rows.append(
                            {
                                "image": image_name,
                                "proposal_index": proposal_slot,
                                "target_index": int(target_index[local_index]),
                                "confidence": float(image_scores[proposal_slot]),
                                "coarse_short": float(matched_short[local_index]),
                                "coarse_long": float(matched_long[local_index]),
                                "aspect_ratio": float(matched_long[local_index] / matched_short[local_index].clamp_min(1e-6)),
                                "quality_probability": float(image_quality[proposal_slot]),
                                "gate": int(matched_gate[local_index]),
                                "dshort": float(matched_residual[local_index, 0]),
                                "dlong": float(matched_residual[local_index, 1]),
                                "coarse_iou": float(coarse_iou[local_index]),
                                "all_refine_iou": float(all_iou[local_index]),
                                "selected_iou": float(selected_iou[local_index]),
                                "short_only_iou": float(short_iou[local_index]),
                                "long_only_iou": float(long_iou[local_index]),
                                "bounded_oracle_iou": float(oracle_iou[local_index]),
                                "all_refine_delta_iou": all_gain,
                                "selected_delta_iou": selected_gain,
                                "bounded_oracle_delta_iou": oracle_gain,
                                "learned_benefit_target": int(all_gain >= quality_min_gain),
                                "bounded_oracle_target": int(oracle_gain >= quality_min_gain),
                            }
                        )

        metric_rows = []
        for variant, metric in metrics.items():
            metric.process(plot=False)
            row = metric_summary(metric, variant)
            metric_rows.append(row)
            metric.clear_stats()
        lookup = {row["variant"]: row for row in metric_rows}
        coarse_map = float(lookup["coarse"]["map50_95"])
        for row in metric_rows:
            row["delta_map50_95_vs_coarse"] = float(row["map50_95"]) - coarse_map
            row["gate_ratio"] = gate_counts[row["variant"]] / valid_count if valid_count else math.nan

        quality_scores = [float(row["quality_probability"]) for row in proposal_rows]
        learned_labels = [int(row["learned_benefit_target"]) for row in proposal_rows]
        oracle_labels = [int(row["bounded_oracle_target"]) for row in proposal_rows]
        all_gains = [float(row["all_refine_delta_iou"]) for row in proposal_rows]
        oracle_gains = [float(row["bounded_oracle_delta_iou"]) for row in proposal_rows]
        selected_gains = [float(row["selected_delta_iou"]) for row in proposal_rows]
        gated_rows = [row for row in proposal_rows if int(row["gate"])]
        gated_gains = [float(row["all_refine_delta_iou"]) for row in gated_rows]
        quality_audit = {
            "match_iou": match_iou,
            "quality_min_gain": quality_min_gain,
            "matched_proposals": len(proposal_rows),
            "selected_gate_ratio_on_matches": len(gated_rows) / len(proposal_rows) if proposal_rows else math.nan,
            "selected_delta_iou_mean": sum(selected_gains) / len(selected_gains) if selected_gains else math.nan,
            "selected_improved_ratio": sum(value > 1e-6 for value in selected_gains) / len(selected_gains)
            if selected_gains
            else math.nan,
            "selected_worsened_ratio": sum(value < -1e-6 for value in selected_gains) / len(selected_gains)
            if selected_gains
            else math.nan,
            "gated_all_refine_delta_iou_mean": sum(gated_gains) / len(gated_gains) if gated_gains else math.nan,
            "gated_improved_ratio": sum(value > 1e-6 for value in gated_gains) / len(gated_gains)
            if gated_gains
            else math.nan,
            "gated_worsened_ratio": sum(value < -1e-6 for value in gated_gains) / len(gated_gains)
            if gated_gains
            else math.nan,
            "quality_vs_learned_gain_pearson": pearson(quality_scores, all_gains),
            "quality_vs_oracle_gain_pearson": pearson(quality_scores, oracle_gains),
            "learned_benefit_classification": binary_summary(quality_scores, learned_labels, threshold),
            "bounded_oracle_classification": binary_summary(quality_scores, oracle_labels, threshold),
        }

        subgroup_rows = []
        subgroup_rows.extend(summarize_rows(proposal_rows, "coarse_short", (4.0, 8.0, 16.0, 32.0)))
        subgroup_rows.extend(summarize_rows(proposal_rows, "aspect_ratio", (10.0, 30.0, 60.0, 100.0)))
        subgroup_rows.extend(summarize_rows(proposal_rows, "confidence", (0.25, 0.5, 0.75)))

        roundtrip_error = abs(_metric_delta(lookup, "roundtrip"))
        gate_off_error = abs(_metric_delta(lookup, "gate_off"))
        refined_error = (
            abs(float(lookup["selected_gate"]["map50_95"]) - args.expected_refined_map50_95)
            if args.expected_refined_map50_95 is not None
            else None
        )
        checks = {
            "baseline_abs_error": abs(coarse_map - args.expected_ca_map50_95),
            "baseline_pass": abs(coarse_map - args.expected_ca_map50_95) <= args.baseline_tolerance,
            "roundtrip_identity_abs_error": roundtrip_error,
            "roundtrip_identity_pass": roundtrip_error <= args.identity_tolerance,
            "gate_off_identity_abs_error": gate_off_error,
            "gate_off_identity_pass": gate_off_error <= args.identity_tolerance,
            "expected_refined_abs_error": refined_error,
            "expected_refined_pass": refined_error is None or refined_error <= args.refined_tolerance,
            "selected_gain_pass": _metric_delta(lookup, "selected_gate") >= 0.002,
        }
        audit = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "ca_weights": str(ca_path),
            "ca_sha256": ca_hash,
            "data": args.data,
            "split": args.split,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "amp": use_amp,
            "quality_threshold": threshold,
            "quality_threshold_source": "checkpoint.train-holdout selection",
            "holdout_mean_residual": holdout_mean.tolist(),
            "proposal_conf": proposal_conf,
            "nms_iou": nms_iou,
            "max_det": max_det,
            "valid_proposals": valid_count,
            "checks": checks,
            "mechanism_margins_map50_95": {
                "selected_minus_all_refine": _metric_delta(lookup, "selected_gate", "all_refine"),
                "selected_minus_mean_selected": _metric_delta(lookup, "selected_gate", "mean_residual_selected"),
                "selected_minus_mean_all": _metric_delta(lookup, "selected_gate", "mean_residual_all"),
                "selected_minus_residual_shuffle": _metric_delta(lookup, "selected_gate", "residual_shuffle"),
                "selected_minus_quality_shuffle": _metric_delta(lookup, "selected_gate", "quality_shuffle"),
                "selected_minus_spatial_shuffle": _metric_delta(lookup, "selected_gate", "spatial_shuffle"),
                "selected_minus_no_renms": _metric_delta(lookup, "selected_gate", "selected_no_renms"),
                "selected_minus_short_only": _metric_delta(lookup, "selected_gate", "short_only"),
                "selected_minus_long_only": _metric_delta(lookup, "selected_gate", "long_only"),
            },
            "hard_integrity_pass": all(
                checks[key]
                for key in (
                    "baseline_pass",
                    "roundtrip_identity_pass",
                    "gate_off_identity_pass",
                    "expected_refined_pass",
                    "selected_gain_pass",
                )
            ),
            "test_used": False,
        }

        write_csv(output_dir / "mechanism_metrics.csv", metric_rows)
        write_csv(output_dir / "matched_proposal_diagnostics.csv", proposal_rows)
        write_csv(output_dir / "subgroup_metrics.csv", subgroup_rows)
        write_json(output_dir / "quality_audit.json", quality_audit)
        write_json(output_dir / "truth_audit.json", audit)
        write_json(
            output_dir / "run_manifest.json",
            {
                "stage": "V3 frozen-checkpoint truth audit",
                "arguments": vars(args),
                "variants": VARIANTS,
                "quality_threshold": threshold,
                "quality_threshold_source": "checkpoint.train-holdout selection",
                "test_used": False,
            },
        )
        write_report(output_dir / "truth_audit_report.md", audit, metric_rows)
        if not audit["hard_integrity_pass"]:
            raise RuntimeError(f"V3 truth audit hard integrity failed; inspect {output_dir / 'truth_audit.json'}")
        print(output_dir / "truth_audit_report.md")
    finally:
        extractor.close()


if __name__ == "__main__":
    main()
