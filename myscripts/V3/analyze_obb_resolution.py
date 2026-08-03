"""Audit OBB target resolution and one-pixel geometry sensitivity at imgsz=640.

The loader always uses validation-style preprocessing, even for the training
split.  This keeps the measurements deterministic and prevents mosaic or other
training augmentation from changing the geometry being audited.  The test split
is intentionally unavailable while the Refine design is still being selected.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SHORT_BINS = (
    ("<2", 0.0, 2.0),
    ("2-4", 2.0, 4.0),
    ("4-8", 4.0, 8.0),
    ("8-16", 8.0, 16.0),
    ("16-32", 16.0, 32.0),
    (">=32", 32.0, math.inf),
)
AR_BINS = (
    ("<3", 1.0, 3.0),
    ("3-5", 3.0, 5.0),
    ("5-10", 5.0, 10.0),
    ("10-20", 10.0, 20.0),
    (">=20", 20.0, math.inf),
)
SENSITIVITY_FIELDS = (
    "center_short_1px_drop",
    "center_long_1px_drop",
    "short_plus_1px_drop",
    "short_minus_1px_drop",
    "long_plus_1px_drop",
    "long_minus_1px_drop",
    "endpoint_1px_angle_drop",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Dataset YAML")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=0, help="0 means the complete split")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.imgsz != 640:
        parser.error("Innovation-one experiments are fixed at imgsz=640")
    if args.batch <= 0 or args.workers < 0 or args.max_images < 0:
        parser.error("batch must be positive; workers and max-images must be non-negative")


def assign_bin(value: float, bins: Iterable[tuple[str, float, float]]) -> str:
    """Return the left-closed bin label for a scalar value."""
    for label, lower, upper in bins:
        if lower <= value < upper:
            return label
    raise ValueError(f"value {value} is outside configured bins")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _quantile(np, values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q)) if values else math.nan


def summarize_rows(np, rows: list[dict[str, Any]], group_field: str) -> list[dict[str, Any]]:
    """Aggregate object-level rows by a named bin or class field."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_field])].append(row)
    result = []
    for group, members in groups.items():
        short = [float(row["short_px"]) for row in members]
        long = [float(row["long_px"]) for row in members]
        aspect = [float(row["aspect_ratio"]) for row in members]
        summary: dict[str, Any] = {
            "group_field": group_field,
            "group": group,
            "count": len(members),
            "short_p10": _quantile(np, short, 0.10),
            "short_p50": _quantile(np, short, 0.50),
            "short_p90": _quantile(np, short, 0.90),
            "long_p50": _quantile(np, long, 0.50),
            "aspect_p50": _quantile(np, aspect, 0.50),
        }
        for field in SENSITIVITY_FIELDS:
            values = [float(row[field]) for row in members]
            summary[f"{field}_mean"] = float(np.mean(values)) if values else math.nan
            summary[f"{field}_p50"] = _quantile(np, values, 0.50)
            summary[f"{field}_p90"] = _quantile(np, values, 0.90)
        result.append(summary)
    return result


def _paired_probiou(probiou, first, second):
    return probiou(first, second).reshape(-1)


def build_sensitivity_rows(torch, probiou, boxes, classes, image_paths, names, image_height: int, image_width: int):
    """Convert one normalized batch to object-level pixel geometry rows."""
    if not boxes.shape[0]:
        return []
    scale = boxes.new_tensor((image_width, image_height, image_width, image_height, 1.0))
    base = boxes * scale
    width, height, angle = base[:, 2], base[:, 3], base[:, 4]
    short = torch.minimum(width, height)
    long = torch.maximum(width, height)
    short_is_width = width <= height
    long_angle = torch.where(short_is_width, angle + math.pi / 2.0, angle)
    short_angle = long_angle + math.pi / 2.0

    def shifted(direction):
        output = base.clone()
        output[:, 0] += direction.cos()
        output[:, 1] += direction.sin()
        return output

    center_short = shifted(short_angle)
    center_long = shifted(long_angle)

    def resized(short_delta: float, long_delta: float):
        output = base.clone()
        new_short = (short + short_delta).clamp_min(0.5)
        new_long = (long + long_delta).clamp_min(0.5)
        output[:, 2] = torch.where(short_is_width, new_short, new_long)
        output[:, 3] = torch.where(short_is_width, new_long, new_short)
        return output

    endpoint_angle = base.clone()
    endpoint_angle[:, 4] += torch.atan2(torch.ones_like(long), (long / 2.0).clamp_min(0.5))
    variants = {
        "center_short_1px_drop": center_short,
        "center_long_1px_drop": center_long,
        "short_plus_1px_drop": resized(1.0, 0.0),
        "short_minus_1px_drop": resized(-1.0, 0.0),
        "long_plus_1px_drop": resized(0.0, 1.0),
        "long_minus_1px_drop": resized(0.0, -1.0),
        "endpoint_1px_angle_drop": endpoint_angle,
    }
    drops = {name: 1.0 - _paired_probiou(probiou, base, value) for name, value in variants.items()}

    rows = []
    for index in range(base.shape[0]):
        class_id = int(classes[index].item())
        short_value = float(short[index].item())
        long_value = float(long[index].item())
        aspect = long_value / max(short_value, 1e-9)
        class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else names[class_id]
        row: dict[str, Any] = {
            "image": str(image_paths[index]),
            "class_id": class_id,
            "class_name": class_name,
            "center_x_px": float(base[index, 0].item()),
            "center_y_px": float(base[index, 1].item()),
            "width_px": float(width[index].item()),
            "height_px": float(height[index].item()),
            "angle_rad": float(angle[index].item()),
            "short_px": short_value,
            "long_px": long_value,
            "aspect_ratio": aspect,
            "short_bin": assign_bin(short_value, SHORT_BINS),
            "aspect_bin": assign_bin(aspect, AR_BINS),
        }
        for name in SENSITIVITY_FIELDS:
            row[name] = float(drops[name][index].item())
        rows.append(row)
    return rows


def write_report(path: Path, args: argparse.Namespace, rows: list[dict[str, Any]], summary: list[dict[str, Any]], np) -> None:
    short = [float(row["short_px"]) for row in rows]
    long = [float(row["long_px"]) for row in rows]
    aspect = [float(row["aspect_ratio"]) for row in rows]
    tiny4 = sum(value < 4.0 for value in short)
    tiny8 = sum(value < 8.0 for value in short)
    lines = [
        "# OBB 目标分辨率与 1 像素敏感度报告",
        "",
        f"- split：`{args.split}`（validation-style preprocessing）",
        f"- imgsz：{args.imgsz}",
        f"- 图像数：{len(set(row['image'] for row in rows))}",
        f"- 实例数：{len(rows)}",
        f"- short < 4 px：{tiny4} ({tiny4 / len(rows):.2%})" if rows else "- short < 4 px：N/A",
        f"- short < 8 px：{tiny8} ({tiny8 / len(rows):.2%})" if rows else "- short < 8 px：N/A",
        "",
        "## 总体尺度",
        "",
        "| variable | p05 | p25 | p50 | p75 | p95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in (("short_px", short), ("long_px", long), ("aspect_ratio", aspect)):
        lines.append(
            f"| {name} | {_quantile(np, values, 0.05):.4f} | {_quantile(np, values, 0.25):.4f} | "
            f"{_quantile(np, values, 0.50):.4f} | {_quantile(np, values, 0.75):.4f} | "
            f"{_quantile(np, values, 0.95):.4f} |"
        )
    lines.extend(
        [
            "",
            "## 按短边分桶的敏感度",
            "",
            "数值为原框与 1 像素扰动框之间的 `1 - ProbIoU`；越大表示该自由度越受像素分辨率限制。",
            "",
            "| short bin | count | short p50 | center-short mean | short+1 mean | angle-endpoint mean |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    order = {label: index for index, (label, _, _) in enumerate(SHORT_BINS)}
    selected = sorted((row for row in summary if row["group_field"] == "short_bin"), key=lambda row: order[row["group"]])
    for row in selected:
        lines.append(
            f"| {row['group']} | {row['count']} | {row['short_p50']:.4f} | "
            f"{row['center_short_1px_drop_mean']:.6f} | {row['short_plus_1px_drop_mean']:.6f} | "
            f"{row['endpoint_1px_angle_drop_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 使用边界",
            "",
            "- 该报告度量的是输入与标注几何的离散分辨率，不是 Refine 模型精度。",
            "- 1 像素扰动采用局部长轴/短轴坐标；角度扰动定义为长边端点移动约 1 像素。",
            "- train split 也采用 val 风格读取，因此结果不受 mosaic、随机缩放等增强影响。",
            "- 本脚本禁止 test；设计冻结前不使用测试集选择结构或阈值。",
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

    from ultralytics.cfg import DEFAULT_CFG, get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils.metrics import probiou

    cfg = get_cfg(
        DEFAULT_CFG,
        overrides={
            "task": "obb",
            "data": args.data,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "rect": True,
            "cache": False,
            "augment": False,
            "plots": False,
        },
    )
    data = check_det_dataset(args.data)
    split_path = data.get(args.split)
    if not split_path:
        parser.error(f"dataset YAML does not define split={args.split}")
    dataset = build_yolo_dataset(cfg, split_path, args.batch, data, mode="val", rect=True, stride=32)
    loader = build_dataloader(
        dataset,
        batch=args.batch,
        workers=args.workers,
        shuffle=False,
        rank=-1,
        drop_last=False,
        pin_memory=False,
    )

    rows: list[dict[str, Any]] = []
    seen_images = 0
    for batch in loader:
        batch_size = len(batch["im_file"])
        allowed = batch_size
        if args.max_images:
            allowed = min(batch_size, args.max_images - seen_images)
            if allowed <= 0:
                break
        keep = batch["batch_idx"] < allowed
        boxes = batch["bboxes"][keep].float()
        classes = batch["cls"][keep].reshape(-1)
        image_indices = batch["batch_idx"][keep].long()
        paths = [batch["im_file"][int(index)] for index in image_indices.tolist()]
        rows.extend(
            build_sensitivity_rows(
                torch,
                probiou,
                boxes,
                classes,
                paths,
                data["names"],
                int(batch["img"].shape[2]),
                int(batch["img"].shape[3]),
            )
        )
        seen_images += allowed
        if args.max_images and seen_images >= args.max_images:
            break

    summary = []
    for field in ("short_bin", "aspect_bin", "class_name"):
        summary.extend(summarize_rows(np, rows, field))
    write_csv(args.output_dir / "obb_resolution_instances.csv", rows)
    write_csv(args.output_dir / "obb_resolution_summary.csv", summary)
    manifest = {
        "data": args.data,
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "max_images": args.max_images,
        "images_seen": seen_images,
        "instances_seen": len(rows),
        "preprocessing": "validation-style; augment=False; rect=True",
        "test_used": False,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(args.output_dir / "obb_resolution_report.md", args, rows, summary, np)
    print(args.output_dir)


if __name__ == "__main__":
    main()
