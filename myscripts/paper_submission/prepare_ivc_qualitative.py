"""Select auditable qualitative cases and render the final IVC comparison figure.

The selector uses only the frozen qualitative export.  Each prediction set is
matched greedily to same-class ground truth by convex-polygon IoU.  It retains
two strong improvements, one near-neutral case and one honest failure case.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from myscripts.paper_visuals.common import save_figure
from myscripts.paper_visuals.generate_fig6_qualitative import build_figure, load_annotations, resolve_path


def polygon_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))) * 0.5


def signed_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5


def convex_intersection(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clipping for the convex OBB quadrilaterals."""
    output = np.asarray(subject, dtype=float)
    clip = np.asarray(clip, dtype=float)
    ccw = signed_area(clip) >= 0

    def cross2d(first, second):
        return float(first[0] * second[1] - first[1] * second[0])

    def inside(point, edge_a, edge_b):
        cross = cross2d(edge_b - edge_a, point - edge_a)
        return cross >= -1e-9 if ccw else cross <= 1e-9

    def intersection(p1, p2, q1, q2):
        r, s = p2 - p1, q2 - q1
        den = cross2d(r, s)
        if abs(den) < 1e-12:
            return p2
        t = cross2d(q1 - p1, s) / den
        return p1 + t * r

    for edge_index in range(len(clip)):
        if len(output) == 0:
            break
        edge_a = clip[edge_index]
        edge_b = clip[(edge_index + 1) % len(clip)]
        input_points = output
        output_points = []
        previous = input_points[-1]
        for current in input_points:
            current_inside = inside(current, edge_a, edge_b)
            previous_inside = inside(previous, edge_a, edge_b)
            if current_inside:
                if not previous_inside:
                    output_points.append(intersection(previous, current, edge_a, edge_b))
                output_points.append(current)
            elif previous_inside:
                output_points.append(intersection(previous, current, edge_a, edge_b))
            previous = current
        output = np.asarray(output_points, dtype=float)
    return output


def polygon_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = convex_intersection(first, second)
    inter_area = polygon_area(intersection) if len(intersection) >= 3 else 0.0
    union = polygon_area(first) + polygon_area(second) - inter_area
    return inter_area / union if union > 0 else 0.0


def matched_iou(gt: list[dict], predictions: list[dict]) -> float:
    if not gt:
        return 1.0 if not predictions else 0.0
    candidates = []
    for gt_index, gt_item in enumerate(gt):
        for pred_index, pred_item in enumerate(predictions):
            if str(gt_item["class"]) == str(pred_item["class"]):
                candidates.append(
                    (polygon_iou(np.asarray(gt_item["points"]), np.asarray(pred_item["points"])), gt_index, pred_index)
                )
    used_gt, used_pred, total = set(), set(), 0.0
    for iou, gt_index, pred_index in sorted(candidates, reverse=True):
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        total += iou
    return total / len(gt)


def score_manifest(frame: pd.DataFrame, manifest: Path) -> pd.DataFrame:
    base = manifest.parent
    records = []
    for _, row in frame.iterrows():
        image_path = resolve_path(row["image_path"], base)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        pixels = np.asarray(image)
        gt = load_annotations(resolve_path(row["gt_path"], base), width, height)
        baseline = load_annotations(resolve_path(row["baseline_path"], base), width, height)
        ca = load_annotations(resolve_path(row["ca_path"], base), width, height)
        refined = load_annotations(resolve_path(row["final_path"], base), width, height)
        records.append(
            {
                "index": row.name,
                "gt_count": len(gt),
                "content_fraction": float((pixels.mean(axis=2) > 8).mean()),
                "baseline_iou": matched_iou(gt, baseline),
                "ca_iou": matched_iou(gt, ca),
                "refined_iou": matched_iou(gt, refined),
            }
        )
    scored = pd.DataFrame(records)
    scored["gain"] = scored["refined_iou"] - scored["ca_iou"]
    return scored


def select_rows(frame: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    usable = scored[
        (scored["gt_count"] >= 1)
        & (scored["gt_count"] <= 12)
        & (scored["content_fraction"] >= 0.95)
    ].copy()
    positive = usable[usable["gain"] > 0.04].sort_values(["gain", "refined_iou"], ascending=False)
    if len(positive) < 2:
        raise RuntimeError("Frozen export does not contain two positive qualitative cases")
    selected = [int(positive.iloc[0]["index"])]
    second_pool = positive[positive["index"] != selected[0]]
    selected.append(int(second_pool.iloc[0]["index"]))
    remaining = usable[~usable["index"].isin(selected)].copy()
    neutral = remaining.iloc[(remaining["gain"].abs()).argmin()]
    selected.append(int(neutral["index"]))
    remaining = remaining[remaining["index"] != int(neutral["index"])]
    failure = remaining.sort_values("gain", ascending=True).iloc[0]
    selected.append(int(failure["index"]))

    labels = ["Improvement A", "Improvement B", "Near-neutral", "Failure case"]
    output = frame.loc[selected].copy().reset_index(drop=True)
    details = scored.set_index("index").loc[selected].reset_index(drop=True)
    output["scene_id"] = labels
    output["note"] = [
        f"ΔIoU={value:+.3f}" for value in details["gain"]
    ]
    output["ca_matched_iou"] = details["ca_iou"]
    output["refined_matched_iou"] = details["refined_iou"]
    output["matched_iou_gain"] = details["gain"]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    frame = pd.read_csv(manifest)
    scored = score_manifest(frame, manifest)
    selected = select_rows(frame, scored)
    for column in ("image_path", "gt_path", "baseline_path", "ca_path", "final_path"):
        selected[column] = selected[column].map(lambda value: str(resolve_path(value, manifest.parent)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_manifest = args.output_dir / "fig6_qualitative_manifest.csv"
    selected.to_csv(selected_manifest, index=False, encoding="utf-8-sig")
    fig = build_figure(selected, selected_manifest, False, english=True)
    paths = save_figure(fig, args.output_dir, "fig6_ivc_qualitative", dpi=600)
    plt.close(fig)
    print(selected_manifest)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
