"""Export reproducible pixel diagnostics for the weak side of neighboring-wire pairs."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .data import WireTiles, example
from .geometry import corners, long_axis
from .paired import footprint, iou, validate_geometry
from .proposals import save_json


PIXEL_FIELDS = ("case", "x", "y", "missed", "covered_by_crop", "probability", "axis_distance_px",
                "along_axis_px", "along_fraction", "beyond_axis_endpoint", "gt_width_estimate_px",
                "target_prompt", "neighbor_prompt")


def select_failures(report, threshold):
    """Deduplicate weak wires while retaining every qualifying pair as selection evidence."""
    selected = {}
    for pair in report["pairs"]:
        values = pair["mask_ious"]["axes"]
        if None in values or sum(value >= threshold for value in values) != 1:
            continue
        weak = 0 if values[0] < threshold else 1
        key = (pair["tile"], pair["proposal_ids"][weak], pair["truth_ids"][weak])
        selected.setdefault(key, []).append(pair)
    return sorted(selected.items())


def thin_mask(mask):
    """Zhang-Suen thinning on a binary raster; preserve component endpoints."""
    result = mask.copy()
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            pad = np.pad(result, 1)
            north = pad[:-2, 1:-1]
            east = pad[1:-1, 2:]
            south = pad[2:, 1:-1]
            west = pad[1:-1, :-2]
            ring = (north, pad[:-2, 2:], east, pad[2:, 2:], south, pad[2:, :-2], west, pad[:-2, :-2])
            count = sum(neighbor.astype(np.uint8) for neighbor in ring)
            transitions = sum((~a & b).astype(np.uint8) for a, b in zip(ring, ring[1:] + ring[:1]))
            if step == 0:
                clear = ~(north & east & south) & ~(east & south & west)
            else:
                clear = ~(north & east & west) & ~(north & south & west)
            remove = result & (count >= 2) & (count <= 6) & (transitions == 1) & clear
            changed |= bool(remove.any())
            result[remove] = False
    return result


def width_estimate(mask):
    """Assign nearest skeleton's digital diameter, 2*EDT-1. This is a raster width estimate."""
    from scipy.ndimage import distance_transform_edt, find_objects, label

    skeleton = thin_mask(mask)
    if not mask.any():
        raise ValueError("Cannot estimate width from an empty target")
    # Explicit background padding makes image-edge widths finite and reproducible.
    radius = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    # Thinning can remove tiny even-sized components. Retain their deepest pixel.
    components, count = label(mask, structure=np.ones((3, 3), dtype=bool))
    widths = np.zeros(mask.shape, dtype=np.float32)
    for component, region in enumerate(find_objects(components, max_label=count), 1):
        local = components[region] == component
        local_skeleton = skeleton[region] & local
        local_radius = radius[region]
        if not local_skeleton.any():
            deepest = np.argmax(np.where(local, local_radius, -1))
            local_skeleton.flat[deepest] = True
            skeleton[region] |= local_skeleton
        nearest = distance_transform_edt(~local_skeleton, return_distances=False, return_indices=True)
        diameter = np.maximum(1, 2 * local_radius[tuple(nearest)] - 1)
        widths[region][local] = diameter[local]
    return skeleton, widths


def axis_coordinates(box, xx, yy):
    x, y, length, _, angle = long_axis(box)
    dx, dy = xx - x, yy - y
    along = dx * np.cos(angle) + dy * np.sin(angle)
    across = -dx * np.sin(angle) + dy * np.cos(angle)
    distance = np.hypot(across, np.maximum(np.abs(along) - length / 2, 0))
    return along, distance, along / length + .5


def mask_image(mask):
    return Image.fromarray(mask.astype(np.uint8) * 255)


def blend(image, mask, color):
    rgb = np.asarray(image.convert("RGB")).copy()
    rgb[mask] = np.rint(.4 * rgb[mask] + .6 * np.asarray(color)).astype(np.uint8)
    return Image.fromarray(rgb)


def error_overlay(image, truth, pred):
    shown = blend(image, truth & pred, (0, 255, 0))
    shown = blend(shown, truth & ~pred, (255, 40, 40))
    return blend(shown, pred & ~truth, (50, 100, 255))


def heat_image(values):
    return Image.fromarray(np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)).convert("RGB")


def panel(items, path, zoom=1):
    font = ImageFont.load_default()
    width = max(im.width for _, im in items) * zoom
    height = max(im.height for _, im in items) * zoom + 24
    canvas = Image.new("RGB", (3 * width, ((len(items) + 2) // 3) * height), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for n, (title, picture) in enumerate(items):
        x, y = (n % 3) * width, (n // 3) * height
        draw.text((x + 5, y + 5), title, fill="white", font=font)
        canvas.paste(picture.resize((picture.width * zoom, picture.height * zoom), Image.Resampling.NEAREST),
                     (x, y + 24))
    canvas.save(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inputs(pairs, paired, tiles, checkpoint):
    if pairs["detector"] != paired["detector"] or pairs["detector"] != tiles.provenance:
        raise ValueError("Both reports and the proposal cache must have the same detector provenance")
    if checkpoint["prompt"] != "axes" or checkpoint["boxes"] != "refined":
        raise ValueError("Expected the existing axes checkpoint trained on refined boxes")
    if checkpoint["detector"] != tiles.provenance:
        raise ValueError("The mask checkpoint uses a different detector")
    if checkpoint["margin"] != pairs["mask_crop_margin_px"] or checkpoint["margin"] != paired["margin_px"]:
        raise ValueError("Crop margins differ between checkpoint and reports")
    if pairs["mask_threshold"] != paired["threshold"] or pairs["coarse_obb_match_iou"] != paired["match_iou"]:
        raise ValueError("The two reports use different mask or matching thresholds")


def export_case(image, boxes, masks, void, proposal, truth_id, probability, crop, features, folder,
                pixel_writer, case_name, threshold):
    target = masks[truth_id] & ~void
    others = np.logical_or.reduce(masks) & ~target & ~void
    covered = footprint(crop, image.size)
    raw = np.asarray(crop.restore(Image.fromarray(probability >= threshold), image.size))
    pred = raw & ~void
    missed = target & ~pred
    full_prob = np.asarray(crop.restore(Image.fromarray(probability), image.size))
    fields = [np.asarray(crop.restore(Image.fromarray(f), image.size)) for f in features[3:]]
    skeleton, widths = width_estimate(target)
    yy, xx = np.nonzero(target)
    along, distance, fraction = axis_coordinates(boxes[proposal], xx, yy)
    values = {"x": xx, "y": yy, "missed": missed[yy, xx], "covered_by_crop": covered[yy, xx],
              "probability": full_prob[yy, xx], "axis_distance_px": distance,
              "along_axis_px": along, "along_fraction": fraction,
              "beyond_axis_endpoint": (fraction < 0) | (fraction > 1),
              "gt_width_estimate_px": widths[yy, xx], "target_prompt": fields[0][yy, xx],
              "neighbor_prompt": fields[1][yy, xx]}
    for n in range(len(xx)):
        pixel_writer.writerow({"case": case_name, **{k: array[n].item() for k, array in values.items()}})
    np.savez_compressed(folder / "arrays.npz", probability_crop=probability, probability_tile=full_prob,
                        target_prompt_crop=features[3], neighbor_prompt_crop=features[4],
                        gt=target, pred=pred, valid=~void, covered=covered, skeleton=skeleton,
                        gt_width_estimate=widths, **values)
    for name, mask in {"gt": target, "prediction": pred, "missed": missed, "false_positive": pred & ~target,
                       "neighbors": others, "void": void, "covered": covered, "skeleton": skeleton}.items():
        mask_image(mask).save(folder / (name + ".png"))
    crop_rgb = crop.crop(image)
    crop_gt = np.asarray(crop.crop(Image.fromarray(target), mask=True))
    crop_valid = np.asarray(crop.crop(Image.fromarray(~void), mask=True))
    crop_pred = (probability >= threshold) & crop_valid
    crop_rgb.save(folder / "crop_rgb.png")
    mask_image(crop_gt).save(folder / "crop_gt.png")
    mask_image(crop_pred).save(folder / "crop_prediction.png")
    heat_image(features[3]).save(folder / "target_prompt.png")
    heat_image(features[4]).save(folder / "neighbor_prompt.png")
    annotated = image.copy()
    drawing = ImageDraw.Draw(annotated)
    for j, box in enumerate(boxes):
        points = [tuple(point) for point in corners(box)]
        drawing.line(points + points[:1], fill=(255, 220, 0) if j == proposal else (0, 200, 255), width=1)
        drawing.text(points[0], str(j), fill=(255, 220, 0) if j == proposal else (0, 200, 255))
    panel([("RGB + proposal IDs (target yellow)", annotated), ("Ground truth (green)", blend(image, target, (0, 255, 0))),
           ("Prediction (magenta)", blend(image, pred, (255, 0, 255))),
           ("TP green / FN red / FP blue", error_overlay(image, target, pred)),
           ("Target prompt", heat_image(fields[0])), ("Neighbor prompt", heat_image(fields[1]))], folder / "panel.png")
    panel([("RGB crop", crop_rgb), ("GT crop", mask_image(crop_gt).convert("RGB")),
           ("Prediction crop", mask_image(crop_pred).convert("RGB")),
           ("TP green / FN red / FP blue", error_overlay(crop_rgb, crop_gt, crop_pred)),
           ("Target prompt", heat_image(features[3])), ("Neighbor prompt", heat_image(features[4]))],
          folder / "crop_panel_4x.png", zoom=4)
    hit_count, count, predicted = int((pred & target).sum()), int(target.sum()), int(pred.sum())
    stats = {"mask_iou": float(iou(pred, target)), "target_pixels": count, "predicted_pixels": predicted,
             "hits": hit_count, "missed_pixels": int(missed.sum()), "target_recall": hit_count / count,
             "neighbor_pixels": int((pred & others).sum()),
             "precision": hit_count / predicted if predicted else None,
             "crop_coverage": int((target & covered).sum()) / count,
             "missed_inside_crop": int((missed & covered).sum()), "missed_outside_crop": int((missed & ~covered).sum())}
    failed = values["missed"]
    stats["missed_axis_distance_median_px"] = float(np.median(distance[failed])) if failed.any() else None
    stats["missed_gt_width_median_px"] = float(np.median(widths[yy, xx][failed])) if failed.any() else None
    return stats, values


def add_bins(totals, values):
    specs = {"width_estimate_px": (values["gt_width_estimate_px"], (0, 1.5, 2.5, 5.5, float("inf"))),
             "axis_distance_px": (values["axis_distance_px"], (0, 1, 2, 4, 8, float("inf"))),
             "along_fraction": (values["along_fraction"], (-float("inf"), 0, .1, .25, .75, .9, 1, float("inf"))),
             "neighbor_prompt": (values["neighbor_prompt"], (0, .1, .5, .9, 1.000001))}
    for name, (array, edges) in specs.items():
        for lo, hi in zip(edges, edges[1:]):
            choose = (array >= lo) & (array < hi)
            key = f"{name} [{lo:g}, {hi:g})"
            totals[key][0] += int(choose.sum())
            totals[key][1] += int(values["missed"][choose].sum())


def run(args):
    import torch
    from .learning import load_mask_head, plane

    print(f"[failures] reading {args.pairs_report} and {args.paired_report}", flush=True)
    pairs = json.loads(Path(args.pairs_report).read_bytes())
    paired = json.loads(Path(args.paired_report).read_bytes())
    args.boxes, args.limit = "refined", None
    args.match_iou = pairs["coarse_obb_match_iou"]
    tiles = WireTiles(args)
    checkpoint, model = load_mask_head(args.weights, args.device)
    verify_inputs(pairs, paired, tiles, checkpoint)
    selected = select_failures(pairs, args.iou_threshold)
    previous = {(r["tile"], r["proposal_id"], r["truth_id"]): r for r in paired["paired_instances"]}
    indices = {path.name: i for i, path in enumerate(tiles.paths)}
    for key, evidence in selected:
        if key not in previous or key[0] not in indices:
            raise ValueError(f"Missing paired record or original image for {key}")
        for pair in evidence:
            side = pair["truth_ids"].index(key[2])
            if abs(pair["mask_ious"]["axes"][side] - previous[key]["refined"]["mask_iou"]) > 1e-5:
                raise ValueError(f"The two reports disagree on the selected wire's IoU: {key}")
    output = Path(args.output_dir)
    archive = output.with_name(output.name + ".tar.gz")
    if archive.exists():
        raise FileExistsError(f"Choose a new output directory; archive already exists: {archive}")
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    for name in ("pairs_report", "paired_report", "weights"):
        shutil.copy2(getattr(args, name), inputs / Path(getattr(args, name)).name)
    manifest = {name: {"path": str(getattr(args, name)), "sha256": sha256(getattr(args, name))}
                for name in ("pairs_report", "paired_report", "weights", "cache")}
    print(f"[failures] selected {len(selected)} unique wires from neighboring pairs at IoU {args.iou_threshold}", flush=True)
    totals = defaultdict(lambda: [0, 0])
    records = []
    exported_tiles = set()
    image = None
    current = None
    with gzip.open(output / "target_pixels.csv.gz", "wt", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PIXEL_FIELDS)
        writer.writeheader()
        with torch.inference_mode():
            for rank, (key, evidence) in enumerate(selected, 1):
                tile, proposal, truth_id = key
                if tile != current:
                    image, boxes, _, masks, void, matches, _ = tiles.read(indices[tile], match_visible_only=True)
                    validate_geometry(tiles.cache[tile], tile)
                    current = tile
                if matches.get(proposal) != truth_id:
                    raise ValueError(f"Changed proposal-to-truth identity for {key}")
                reference = previous[key]
                if not np.allclose(boxes[proposal], reference["refined"]["box_xywhr"], atol=1e-5, rtol=0):
                    raise ValueError(f"Changed refined box for {key}")
                stem = Path(tile).stem
                case = f"{rank:03d}_{stem}_p{proposal}_g{truth_id}"
                folder = output / "cases" / case
                folder.mkdir(parents=True)
                if tile not in exported_tiles:
                    source = output / "images" / stem
                    source.mkdir(parents=True)
                    shutil.copy2(tiles.paths[indices[tile]], source / tile)
                    image.save(source / "image.png")
                    shutil.copy2(tiles.jsons / (stem + ".json"), source / "annotation.json")
                    exported_tiles.add(tile)
                crop, features = example(image, boxes, proposal, "axes", checkpoint["margin"])
                probability = model(plane(features, args.device))[0, 0].sigmoid().cpu().numpy()
                stats, values = export_case(image, boxes, masks, void, proposal, truth_id, probability,
                                            crop, features, folder, writer, case, pairs["mask_threshold"])
                expected = reference["refined"]
                agrees = (abs(stats["mask_iou"] - expected["mask_iou"]) <= 1e-5
                          and stats["target_pixels"] == reference["target_pixels"]
                          and stats["hits"] == expected["hits"] and stats["predicted_pixels"] == expected["predicted_pixels"])
                row = {"case": case, "tile": tile, "proposal_id": proposal, "truth_id": truth_id,
                       "reproduces_report": agrees, "reported_iou": expected["mask_iou"], **stats}
                records.append(row)
                save_json(folder / "case.json", {**row, "selection_pairs": evidence,
                          "coarse_box": tiles.cache[tile]["coarse"][proposal], "refined_box": boxes[proposal],
                          "reported": reference, "crop_matrix": crop.matrix.tolist(),
                          "crop_size": list(crop.size), "original_image": f"../../images/{stem}/image.png"})
                add_bins(totals, values)
                print(f"[failures] {rank}/{len(selected)} {tile} p{proposal} g{truth_id} "
                      f"IoU {stats['mask_iou']:.4f} recall {stats['target_recall']:.4f} reproduced={agrees}", flush=True)
    with (output / "cases.csv").open("w", encoding="utf-8-sig", newline="") as file:
        fields = list(records[0]) if records else ("case", "tile", "proposal_id", "truth_id")
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    bins = [{"bin": name, "target_pixels": count, "missed_pixels": missed,
             "recall": (count - missed) / count if count else None} for name, (count, missed) in totals.items()]
    with (output / "pixel_bins.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("bin", "target_pixels", "missed_pixels", "recall"))
        writer.writeheader()
        writer.writerows(bins)
    save_json(inputs / "selected_cache.json", {**tiles.provenance,
              "frames": {name: tiles.cache[name] for name in sorted(exported_tiles)}})
    summary = {"selected_wires": len(records), "tiles": len(exported_tiles), "iou_selection_threshold": args.iou_threshold,
               "reproduction_mismatches": [r["case"] for r in records if not r["reproduces_report"]],
               "mask_threshold": pairs["mask_threshold"], "manifest": manifest, "pixel_bins": bins,
               "width_method": "Zhang-Suen skeleton (deepest-pixel fallback for lost tiny components); "
                               "2*EDT-1 at skeleton, clipped to at least 1; "
                               "nearest-skeleton propagation. Raster estimate, not physical diameter. "
                               "Even-width, endpoint and curved regions have discretization error.",
               "cases": records}
    save_json(output / "summary.json", summary)
    write_index(output, records)
    temporary = archive.with_name(archive.name + ".tmp")
    with tarfile.open(temporary, "w:gz") as tar:
        tar.add(output, arcname=output.name)
    os.replace(temporary, archive)
    print(f"[failures] archive: {archive} ({archive.stat().st_size / 2**20:.1f} MiB)", flush=True)
    print(f"[failures] reproduction mismatches: {len(summary['reproduction_mismatches'])}", flush=True)


def write_index(output, records):
    import html

    cards = []
    for row in records:
        case = html.escape(row["case"])
        cards.append(f'<section><h2>{case}</h2><p>IoU {row["mask_iou"]:.4f}; '
                     f'recall {row["target_recall"]:.4f}; reproduced {row["reproduces_report"]}</p>'
                     f'<a href="cases/{case}/crop_panel_4x.png"><img src="cases/{case}/panel.png" '
                     'style="max-width:100%"></a></section>')
    (output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Wire failure diagnostics</title>'
                                      '<h1>Wire failure diagnostics</h1><p>Green TP, red FN, blue FP. '
                                      'Click a panel for the crop at 4x nearest-neighbor zoom.</p>' + ''.join(cards), encoding="utf-8")
    (output / "README.txt").write_text(
        "解压后打开 index.html 浏览全部案例，点击整图面板查看原始裁剪的四倍最近邻放大图。\n"
        "绿色表示正确覆盖，红色表示目标漏分，蓝色表示目标外的预测。黄色框为当前目标，青色框为其他候选。\n"
        "summary.json 是汇总；cases.csv 与 pixel_bins.csv 可用表格软件查看；target_pixels.csv.gz 解压后含每个目标真值像素。\n"
        "cases 下的 arrays.npz 保留原始浮点提示、概率、掩膜、骨架、宽度估计与像素坐标，不含可执行对象。\n"
        "images 按图块保存原始 RGB 和实例标注；inputs 保存两份报告、权重以及这些图块的完整候选缓存。\n"
        "轴线距离以原图像素计，along_fraction=0/1 为预测轴线几何两端，可小于0或大于1；这些端点不是物理末端。\n"
        "宽度是最近骨架半径导出的栅格估计，偶数宽度、端点、弯曲与标注误差都会影响它，不能当作物理线径。\n"
        "漏分原因需要结合图像判断。若 reproduction_mismatches 非空，先核查权重、图像、标注及推理数值差异。\n",
        encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("images", "jsons", "cache", "weights", "pairs-report", "paired-report", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iou-threshold", type=float, default=.50)
    args = parser.parse_args(argv)
    if not 0 < args.iou_threshold <= 1:
        parser.error("--iou-threshold must be in (0, 1]")
    run(args)


if __name__ == "__main__":
    main()
