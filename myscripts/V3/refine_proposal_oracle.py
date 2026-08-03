"""Evaluate proposal-level OBB refinement oracles on val without using test.

The script keeps the CA scores and classes fixed, modifies only matched proposal
geometry, reruns rotated NMS, and reports full detection metrics.  It therefore
measures how much final mAP headroom a proposal refiner can exploit, rather than
only measuring IoU on TAL positive anchors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


VARIANTS = (
    "standard_ca",
    "topk_coarse",
    "oracle_scale",
    "oracle_center",
    "oracle_angle",
    "oracle_scale_center",
    "oracle_full_geometry",
)


def parse_iou_thresholds(value: str) -> tuple[float, ...]:
    """Parse a nonempty, unique list of IoU thresholds in (0, 1]."""
    try:
        values = tuple(dict.fromkeys(float(item.strip()) for item in value.split(",") if item.strip()))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not values or any(item <= 0.0 or item > 1.0 for item in values):
        raise argparse.ArgumentTypeError("IoU thresholds must be in (0, 1]")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Canonical pure-CA best.pt")
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--proposal-topk", type=int, default=1000)
    parser.add_argument("--proposal-conf", type=float, default=0.001)
    parser.add_argument("--oracle-match-iou", type=float, default=0.30)
    parser.add_argument("--recall-ious", type=parse_iou_thresholds, default=parse_iou_thresholds("0.5,0.75,0.9"))
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.imgsz != 640:
        parser.error("Innovation-one experiments are fixed at imgsz=640")
    if args.proposal_topk <= 0 or args.max_det <= 0 or args.batch <= 0:
        parser.error("proposal-topk, max-det, and batch must be positive")
    for name in ("proposal_conf", "oracle_match_iou", "nms_iou"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    if not args.weights.is_file():
        parser.error(f"weights not found: {args.weights}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_recall(rows: list[dict[str, Any]], thresholds: tuple[float, ...]) -> list[dict[str, Any]]:
    """Aggregate image-level proposal recall counts without averaging image ratios."""
    gt_total = sum(int(row["gt_count"]) for row in rows)
    result = []
    for threshold in thresholds:
        key = f"recalled_{threshold:.2f}"
        recalled = sum(int(row[key]) for row in rows)
        result.append(
            {
                "iou_threshold": threshold,
                "gt_count": gt_total,
                "recalled_gt": recalled,
                "proposal_recall": recalled / gt_total if gt_total else math.nan,
            }
        )
    return result


def metric_summary(metric: Any, variant: str) -> dict[str, Any]:
    values = metric.results_dict
    all_ap = metric.box.all_ap
    ap90 = float(all_ap[:, 8].mean()) if len(all_ap) else math.nan
    ap95 = float(all_ap[:, 9].mean()) if len(all_ap) else math.nan
    return {
        "variant": variant,
        "precision": float(values["metrics/precision(B)"]),
        "recall": float(values["metrics/recall(B)"]),
        "map50": float(values["metrics/mAP50(B)"]),
        "map50_95": float(values["metrics/mAP50-95(B)"]),
        "ap75": float(metric.box.map75),
        "ap90": ap90,
        "ap95": ap95,
    }


def build_validator_class(*, OBBValidator, OBBMetrics, batch_probiou, nms_module, torch, np, options):
    """Build the cloud-only custom validator while keeping local ``--help`` torch-free."""

    class ProposalOracleValidator(OBBValidator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.extra_metrics: dict[str, Any] = {}
            self.variant_summaries: list[dict[str, Any]] = []
            self.proposal_recall_rows: list[dict[str, Any]] = []
            self._current_batch = None

        def init_metrics(self, model):
            super().init_metrics(model)
            self.extra_metrics = {name: OBBMetrics(names=model.names) for name in VARIANTS if name != "standard_ca"}

        def preprocess(self, batch):
            prepared = super().preprocess(batch)
            self._current_batch = prepared
            return prepared

        @staticmethod
        def _periodic_angle_distance(first, second):
            difference = first - second
            return 0.5 * torch.atan2(torch.sin(2.0 * difference), torch.cos(2.0 * difference)).abs()

        def _align_targets(self, proposals, targets):
            """Choose the equivalent GT (w,h,theta) representation closest to each proposal."""
            alternative = targets.clone()
            alternative[:, 2] = targets[:, 3]
            alternative[:, 3] = targets[:, 2]
            alternative[:, 4] = targets[:, 4] + math.pi / 2.0
            use_alternative = self._periodic_angle_distance(alternative[:, 4], proposals[:, 4]) < self._periodic_angle_distance(
                targets[:, 4], proposals[:, 4]
            )
            return torch.where(use_alternative[:, None], alternative, targets)

        def _raw_tensor(self, preds):
            value = preds[0] if isinstance(preds, (tuple, list)) else preds
            if not isinstance(value, torch.Tensor) or value.ndim != 3:
                raise TypeError(f"Expected BCN inference tensor, received {type(value)}")
            return value.detach().clone()

        def _format_nms(self, raw_image):
            outputs = nms_module.non_max_suppression(
                raw_image.unsqueeze(0),
                options.proposal_conf,
                options.nms_iou,
                nc=self.nc,
                multi_label=True,
                agnostic=self.args.single_cls or self.args.agnostic_nms,
                max_det=options.max_det,
                rotated=True,
            )[0]
            return {
                "bboxes": torch.cat((outputs[:, :4], outputs[:, -1:]), dim=-1),
                "conf": outputs[:, 4],
                "cls": outputs[:, 5],
            }

        def _topk_raw(self, raw_image):
            class_scores = raw_image[4 : 4 + self.nc]
            confidence = class_scores.amax(dim=0)
            keep = confidence > options.proposal_conf
            indices = torch.where(keep)[0]
            if indices.numel() > options.proposal_topk:
                order = confidence[indices].argsort(descending=True)[: options.proposal_topk]
                indices = indices[order]
            return raw_image[:, indices].clone()

        def _match_proposals(self, selected, pbatch, image_index):
            proposal_count = selected.shape[1]
            gt_boxes = pbatch["bboxes"]
            gt_cls = pbatch["cls"]
            row = {
                "image": str(pbatch["im_file"]),
                "image_index": image_index,
                "gt_count": int(gt_boxes.shape[0]),
                "proposal_count": int(proposal_count),
            }
            if not proposal_count or not gt_boxes.shape[0]:
                for threshold in options.recall_ious:
                    row[f"recalled_{threshold:.2f}"] = 0
                return None, None, row

            boxes = torch.cat((selected[:4].T, selected[4 + self.nc : 5 + self.nc].T), dim=-1)
            scores, pred_cls = selected[4 : 4 + self.nc].T.max(dim=1)
            iou = batch_probiou(gt_boxes, boxes)
            same_class = gt_cls[:, None].long() == pred_cls[None, :].long()
            class_iou = torch.where(same_class, iou, torch.full_like(iou, -1.0))
            best_iou, best_gt = class_iou.max(dim=0)
            gt_best = class_iou.max(dim=1).values.clamp_min(0.0)
            for threshold in options.recall_ious:
                row[f"recalled_{threshold:.2f}"] = int((gt_best >= threshold).sum().item())
            matched = best_iou >= options.oracle_match_iou
            matched_targets = gt_boxes[best_gt.clamp_min(0)]
            matched_targets = self._align_targets(boxes, matched_targets)
            return matched, matched_targets, row

        def _oracle_raw(self, selected, matched, targets, variant):
            output = selected.clone()
            if matched is None or not matched.any() or variant == "topk_coarse":
                return output
            idx = torch.where(matched)[0]
            target = targets[idx]
            if variant in {"oracle_center", "oracle_scale_center", "oracle_full_geometry"}:
                output[0:2, idx] = target[:, 0:2].T
            if variant in {"oracle_scale", "oracle_scale_center", "oracle_full_geometry"}:
                output[2:4, idx] = target[:, 2:4].T
            if variant in {"oracle_angle", "oracle_full_geometry"}:
                output[4 + self.nc, idx] = target[:, 4]
            return output

        def postprocess(self, preds):
            raw = self._raw_tensor(preds)
            standard = super().postprocess(preds)
            payload: dict[str, list[dict[str, Any]]] = {"standard_ca": standard}
            for variant in VARIANTS:
                if variant != "standard_ca":
                    payload[variant] = []

            if self._current_batch is None:
                raise RuntimeError("Validator batch cache is empty")
            for image_index, raw_image in enumerate(raw):
                selected = self._topk_raw(raw_image)
                pbatch = self._prepare_batch(image_index, self._current_batch)
                matched, targets, recall_row = self._match_proposals(selected, pbatch, image_index)
                self.proposal_recall_rows.append(recall_row)
                for variant in VARIANTS:
                    if variant == "standard_ca":
                        continue
                    variant_raw = self._oracle_raw(selected, matched, targets, variant)
                    payload[variant].append(self._format_nms(variant_raw))
            return payload

        def _update_extra_metric(self, metric, preds, batch):
            for image_index, pred in enumerate(preds):
                pbatch = self._prepare_batch(image_index, batch)
                predn = self._prepare_pred({key: value.clone() for key, value in pred.items()})
                cls = pbatch["cls"].cpu().numpy()
                no_pred = predn["cls"].shape[0] == 0
                metric.update_stats(
                    {
                        **self._process_batch(predn, pbatch),
                        "target_cls": cls,
                        "target_img": np.unique(cls),
                        "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                        "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                    }
                )

        def update_metrics(self, payload, batch):
            super().update_metrics(payload["standard_ca"], batch)
            for variant, metric in self.extra_metrics.items():
                self._update_extra_metric(metric, payload[variant], batch)

        def get_stats(self):
            base_results = super().get_stats()
            self.variant_summaries = [metric_summary(self.metrics, "standard_ca")]
            for variant, metric in self.extra_metrics.items():
                metric.process(save_dir=self.save_dir, plot=False, on_plot=None)
                self.variant_summaries.append(metric_summary(metric, variant))
                metric.clear_stats()
            return base_results

    return ProposalOracleValidator


def write_report(path: Path, args: argparse.Namespace, metrics: list[dict[str, Any]], recall: list[dict[str, Any]]) -> None:
    lookup = {row["variant"]: row for row in metrics}
    standard = lookup["standard_ca"]
    topk = lookup["topk_coarse"]
    lines = [
        "# Proposal-level Refine Oracle 报告",
        "",
        "本报告只使用 val。分类分数与类别保持不变，oracle 仅替换匹配 proposal 的指定几何自由度，并重新执行旋转 NMS。",
        "",
        f"- CA 权重：`{args.weights}`",
        f"- pre-NMS top-K：{args.proposal_topk}",
        f"- proposal 置信度阈值：{args.proposal_conf}",
        f"- oracle 匹配阈值：{args.oracle_match_iou}",
        f"- NMS IoU：{args.nms_iou}",
        "",
        "## Proposal recall",
        "",
        "| IoU | GT | recalled | recall |",
        "|---:|---:|---:|---:|",
    ]
    for row in recall:
        lines.append(
            f"| {row['iou_threshold']:.2f} | {row['gt_count']} | {row['recalled_gt']} | {row['proposal_recall']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 完整指标",
            "",
            "| variant | mAP50-95 | Δ vs standard | AP75 | AP90 | AP95 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics:
        lines.append(
            f"| {row['variant']} | {row['map50_95']:.6f} | {row['map50_95'] - standard['map50_95']:+.6f} | "
            f"{row['ap75']:.6f} | {row['ap90']:.6f} | {row['ap95']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            f"- `topk_coarse-standard_ca` 为 {topk['map50_95'] - standard['map50_95']:+.6f}，用于检查 top-K 截断是否改变基线。",
            "- oracle 是上限实验，不代表网络可以自动达到该结果。",
            "- 若完整几何 oracle 有明显收益而旋转 ROI Probe 仍失败，瓶颈更可能位于特征或监督，而不是 proposal recall。",
            "- 本脚本不读取 test，也不作终止 Refine 的自动判断。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch

    from ultralytics import YOLO
    from ultralytics.models.yolo.obb.val import OBBValidator
    from ultralytics.utils import nms
    from ultralytics.utils.metrics import OBBMetrics, batch_probiou

    model = YOLO(str(args.weights), task="obb")
    validator_class = build_validator_class(
        OBBValidator=OBBValidator,
        OBBMetrics=OBBMetrics,
        batch_probiou=batch_probiou,
        nms_module=nms,
        torch=torch,
        np=np,
        options=args,
    )
    validator = validator_class(
        args={
            "model": str(args.weights),
            "data": args.data,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "split": args.split,
            "task": "obb",
            "mode": "val",
            "rect": True,
            "conf": args.proposal_conf,
            "iou": args.nms_iou,
            "max_det": args.max_det,
            "plots": False,
            "save_json": False,
            "save_txt": False,
            "verbose": False,
            "project": str(args.output_dir),
            "name": "validator",
            "exist_ok": True,
        },
        _callbacks=model.callbacks,
    )
    validator(model=model.model)

    metrics = sorted(validator.variant_summaries, key=lambda row: VARIANTS.index(row["variant"]))
    recall = aggregate_recall(validator.proposal_recall_rows, args.recall_ious)
    standard = next(row for row in metrics if row["variant"] == "standard_ca")
    for row in metrics:
        row["delta_map50_95_vs_standard"] = row["map50_95"] - standard["map50_95"]
    write_csv(args.output_dir / "proposal_oracle_metrics.csv", metrics)
    write_csv(args.output_dir / "proposal_recall_by_image.csv", validator.proposal_recall_rows)
    write_csv(args.output_dir / "proposal_recall_summary.csv", recall)
    manifest = {
        "weights": str(args.weights),
        "data": args.data,
        "split": args.split,
        "imgsz": args.imgsz,
        "proposal_topk": args.proposal_topk,
        "proposal_conf": args.proposal_conf,
        "oracle_match_iou": args.oracle_match_iou,
        "nms_iou": args.nms_iou,
        "max_det": args.max_det,
        "test_used": False,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(args.output_dir / "proposal_oracle_report.md", args, metrics, recall)
    print(args.output_dir)


if __name__ == "__main__":
    main()
