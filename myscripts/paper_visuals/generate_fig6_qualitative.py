"""Generate Fig. 6: aligned GT/Baseline/CA/final qualitative comparison panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from myscripts.paper_visuals.common import (
    COLORS,
    configure_matplotlib,
    ensure_output_dir,
    require_columns,
    require_nonempty,
    save_figure,
)


configure_matplotlib()
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


MANIFEST_COLUMNS = ["scene_id", "image_path", "gt_path", "baseline_path", "ca_path", "final_path"]


def resolve_path(value, base: Path) -> Path:
    """Resolve a manifest path relative to the manifest file."""
    if pd.isna(value) or not str(value).strip():
        raise ValueError("清单中存在空路径")
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def load_annotations(path: Path, width: int, height: int) -> list[dict]:
    """Load YOLO-OBB TXT, LabelMe JSON, or generic detection JSON annotations."""
    if not path.exists():
        raise FileNotFoundError(path)
    annotations = []
    if path.suffix.lower() == ".txt":
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) < 9:
                raise ValueError(f"{path}:{line_number} 至少需要 class + 8 个顶点坐标")
            values = np.asarray([float(value) for value in fields[1:9]], dtype=float).reshape(4, 2)
            if np.nanmax(np.abs(values)) <= 1.5:
                values[:, 0] *= width
                values[:, 1] *= height
            confidence = float(fields[9]) if len(fields) >= 10 else None
            annotations.append({"class": fields[0], "points": values, "confidence": confidence})
        return annotations

    if path.suffix.lower() != ".json":
        raise ValueError(f"不支持的标注格式: {path.suffix}; 请使用 YOLO-OBB TXT 或 JSON")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "shapes" in data:
        for item in data["shapes"]:
            points = np.asarray(item.get("points", []), dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
                continue
            annotations.append(
                {
                    "class": str(item.get("label", "object")),
                    "points": points,
                    "confidence": item.get("confidence"),
                }
            )
        return annotations

    items = data.get("detections", data.get("annotations", data)) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError(f"{path} 缺少 shapes、detections 或 annotations 列表")
    for item in items:
        points = np.asarray(item.get("points", item.get("xyxyxyxy", [])), dtype=float).reshape(-1, 2)
        if len(points) < 2:
            continue
        if np.nanmax(np.abs(points)) <= 1.5:
            points[:, 0] *= width
            points[:, 1] *= height
        annotations.append(
            {
                "class": str(item.get("class", item.get("label", "object"))),
                "points": points,
                "confidence": item.get("confidence", item.get("conf")),
            }
        )
    return annotations


def parse_roi(row, width: int, height: int) -> tuple[int, int, int, int]:
    """Read an optional common ROI; all four comparison columns use the same crop."""
    columns = ["roi_x1", "roi_y1", "roi_x2", "roi_y2"]
    if not set(columns).issubset(row.index) or any(pd.isna(row[column]) for column in columns):
        return 0, 0, width, height
    x1, y1, x2, y2 = [int(round(float(row[column]))) for column in columns]
    x1, x2 = sorted((max(0, x1), min(width, x2)))
    y1, y2 = sorted((max(0, y1), min(height, y2)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        raise ValueError(f"场景 {row['scene_id']} 的 ROI 无效: {(x1, y1, x2, y2)}")
    return x1, y1, x2, y2


def shift_annotations(annotations: list[dict], roi) -> list[dict]:
    """Shift annotations into crop coordinates and retain intersecting objects."""
    x1, y1, x2, y2 = roi
    shifted = []
    for item in annotations:
        points = np.asarray(item["points"], dtype=float)
        if points[:, 0].max() < x1 or points[:, 0].min() > x2 or points[:, 1].max() < y1 or points[:, 1].min() > y2:
            continue
        clone = dict(item)
        clone["points"] = points - np.array([x1, y1])
        shifted.append(clone)
    return shifted


def draw_annotations(ax, annotations, *, color, linestyle, show_confidence, linewidth):
    """Draw polygonal OBBs with optional confidence labels."""
    for item in annotations:
        points = np.asarray(item["points"], dtype=float)
        ax.add_patch(
            Polygon(points, closed=True, fill=False, edgecolor=color, linewidth=linewidth, linestyle=linestyle)
        )
        if show_confidence and item.get("confidence") is not None:
            label = f"{float(item['confidence']):.2f}"
            anchor = points[np.argmin(points[:, 1])]
            ax.text(
                anchor[0],
                anchor[1],
                label,
                fontsize=6.2,
                color="white",
                bbox={"facecolor": color, "edgecolor": "none", "pad": 1.0, "alpha": 0.9},
            )


def build_figure(frame: pd.DataFrame, manifest_path: Path, show_confidence: bool):
    """Build a row-per-scene, column-per-method comparison grid."""
    base = manifest_path.parent
    rows = []
    for _, row in frame.iterrows():
        image_path = resolve_path(row["image_path"], base)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        roi = parse_roi(row, width, height)
        crop = np.asarray(image.crop(roi))
        annotations = {
            key: shift_annotations(load_annotations(resolve_path(row[column], base), width, height), roi)
            for key, column in (
                ("gt", "gt_path"),
                ("baseline", "baseline_path"),
                ("ca", "ca_path"),
                ("final", "final_path"),
            )
        }
        rows.append((row, crop, annotations))

    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 4, figsize=(7.25, max(2.05, 1.82 * n_rows)), squeeze=False)
    titles = ["Ground Truth", "Baseline", "CA", "CA + Refine\n(normal refine)"]
    for column, title in enumerate(titles):
        axes[0, column].set_title(title, fontsize=9.2, pad=5)

    for row_index, (row, crop, annotations) in enumerate(rows):
        diagonal = float(np.hypot(crop.shape[1], crop.shape[0]))
        # Keep high-DPI exports crisp without covering very thin conductors.
        linewidth = float(np.clip(diagonal / 1400.0, 0.65, 1.15))
        for column, key in enumerate(("gt", "baseline", "ca", "final")):
            ax = axes[row_index, column]
            ax.imshow(crop)
            if key == "gt":
                draw_annotations(
                    ax,
                    annotations["gt"],
                    color=COLORS["success"],
                    linestyle="-",
                    show_confidence=False,
                    linewidth=linewidth * 1.12,
                )
            else:
                draw_annotations(
                    ax,
                    annotations["gt"],
                    color=COLORS["success"],
                    linestyle="--",
                    show_confidence=False,
                    linewidth=linewidth * 0.88,
                )
                draw_annotations(
                    ax,
                    annotations[key],
                    color=COLORS["refine"],
                    linestyle="-",
                    show_confidence=show_confidence,
                    linewidth=linewidth * 1.12,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.55)
                spine.set_edgecolor(COLORS["gray"])
        note = "" if "note" not in row.index or pd.isna(row["note"]) else f"  {row['note']}"
        axes[row_index, 0].set_ylabel(f"{row['scene_id']}{note}", fontsize=7.3, rotation=90, labelpad=4)

    fig.text(
        0.995,
        0.008,
        "绿色：GT（对比列为虚线）；红色实线：预测",
        ha="right",
        va="bottom",
        fontsize=6.8,
        color=COLORS["dark"],
    )
    fig.subplots_adjust(left=0.065, right=0.995, top=0.92, bottom=0.045, wspace=0.025, hspace=0.08)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--show-confidence", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    require_nonempty(frame, args.manifest)
    require_columns(frame, MANIFEST_COLUMNS, args.manifest)
    fig = build_figure(frame, args.manifest.resolve(), args.show_confidence)
    paths = save_figure(fig, ensure_output_dir(args.output_dir), "fig6_qualitative_comparison", dpi=600)
    plt.close(fig)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
