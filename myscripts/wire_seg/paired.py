"""Measure transfer from size refinement to instance masks on a fixed CA-defined cohort."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .data import WireTiles, example, near_parallel, source_groups
from .geometry import WireCrop, corners, long_axis, paint
from .proposals import save_json


VARIANTS = ("coarse", "refined")


def validate_geometry(entry, name):
    coarse, refined = entry["coarse"], entry["refined"]
    if len(coarse) != len(refined) or len(coarse) != len(entry["scores"]):
        raise ValueError(f"Unpaired proposal counts in {name}")
    for i, (a, b) in enumerate(zip(coarse, refined)):
        long_axis(b)  # validate dimensions, coordinates and angle
        if not np.allclose(np.take(a, [0, 1, 4]), np.take(b, [0, 1, 4]), atol=1e-5, rtol=0):
            raise ValueError(f"Expected size-only V3.1.1 refinement in {name}, proposal {i}")


def footprint(crop, size):
    return np.asarray(crop.restore(Image.new("1", crop.size, 1), size))


def iou(pred, truth):
    union = np.count_nonzero(pred | truth)
    return np.count_nonzero(pred & truth) / union if union else 0.


def box_pixels(box, size):
    return np.asarray(paint(size, [corners(box)])) > 0


def reference_case(i, pairs, masks, truth_boxes, region):
    """Fix exposure and the distractor once, using the coarse crop and exclusive GT pixels."""
    j = pairs[i]
    exclusive = region & ~masks[j]
    areas = [np.count_nonzero(exclusive & m) for m in masks]
    neighbors = [k for k, area in enumerate(areas) if area]
    choices = [(areas[k], p) for p, k in pairs.items() if areas[k]]
    partner = max(choices)[1] if choices else None
    return bool(neighbors), near_parallel(j, neighbors, truth_boxes), partner


def evaluate_tile(image, entry, masks, void, pairs, truth_boxes, margin, predict):
    """Run both arms in original tile coordinates. No branch-specific rematching."""
    masks = [m & ~void for m in masks]
    foreground = np.logical_or.reduce(masks) if masks else np.zeros_like(void)
    rows = []
    ap_inputs = {v: [] for v in VARIANTS}
    for i, score in enumerate(entry["scores"]):
        j = pairs.get(i)
        matched = j is not None and masks[j].any()
        if matched:
            target = masks[j]
            truth_box_pixels = box_pixels(truth_boxes[j], image.size)
            reference = footprint(WireCrop(entry["coarse"][i], margin), image.size)
            exposed, parallel, partner = reference_case(i, pairs, masks, truth_boxes, reference)
            row = {"proposal_id": i, "truth_id": j, "score": score, "exposed": exposed,
                   "near_parallel": parallel, "swap_neighbor_proposal_id": partner,
                   "swap_neighbor_truth_id": pairs[partner] if partner is not None else None,
                   "target_pixels": int(target.sum())}
        for variant in VARIANTS:
            boxes = entry[variant]
            crop, feature = example(image, boxes, i, "axes", margin)
            raw = np.asarray(crop.restore(Image.fromarray(predict(feature)), image.size))
            pred = raw & ~void
            if pred.any() or not raw.any():  # same void-only exclusion as the standalone mask AP
                overlaps = []
                for k, m in enumerate(masks):
                    intersection = np.count_nonzero(pred & m)
                    if intersection:
                        overlaps.append((k, intersection / np.count_nonzero(pred | m)))
                ap_inputs[variant].append((score, sorted(overlaps, key=lambda kv: -kv[1])))
            if not matched:
                continue
            covered = reference if variant == "coarse" else footprint(crop, image.size)
            metrics = {"box_xywhr": boxes[i], "tile_obb_iou": iou(box_pixels(boxes[i], image.size), truth_box_pixels),
                       "covered_target_pixels": int(np.count_nonzero(target & covered)),
                       "hits": int(np.count_nonzero(pred & target)), "predicted_pixels": int(pred.sum()),
                       "neighbor_pixels": int(np.count_nonzero(pred & foreground & ~target)),
                       "mask_iou": iou(pred, target), "swap_success": None}
            if partner is not None:
                # Hold RGB and crop fixed within an arm; both arms are scored in the same coarse window.
                feature[3:] = np.stack(crop.prompts(boxes, partner, "axes"))
                switched = np.asarray(crop.restore(Image.fromarray(predict(feature)), image.size)) & ~void
                a, b = target & reference, masks[pairs[partner]] & reference
                first, second = pred & reference, switched & reference
                metrics["swap_success"] = bool(iou(first, a) > iou(first, b) and iou(second, b) > iou(second, a))
            row[variant] = metrics
        if matched:
            row["mask_iou_delta"] = row["refined"]["mask_iou"] - row["coarse"]["mask_iou"]
            rows.append(row)
    return rows, ap_inputs, sum(bool(m.any()) for m in masks)


def ratio(n, d):
    return n / d if d else None


def summarize(rows):
    total_target = sum(r["target_pixels"] for r in rows)
    result = {"instances": len(rows)}
    for v in VARIANTS:
        obs = [r[v] for r in rows]
        swaps = [m["swap_success"] for m in obs if m["swap_success"] is not None]
        result[v] = {
            "mean_tile_obb_iou": ratio(sum(m["tile_obb_iou"] for m in obs), len(obs)),
            "target_crop_coverage": ratio(sum(m["covered_target_pixels"] for m in obs), total_target),
            "mean_instance_iou": ratio(sum(m["mask_iou"] for m in obs), len(obs)),
            "target_pixel_recall": ratio(sum(m["hits"] for m in obs), total_target),
            "neighbor_pixel_leakage": ratio(sum(m["neighbor_pixels"] for m in obs), sum(m["predicted_pixels"] for m in obs)),
            "swap_both_correct_fraction": ratio(sum(swaps), len(swaps)),
        }
    result["refined_minus_coarse"] = {
        k: result["refined"][k] - value if value is not None and result["refined"][k] is not None else None
        for k, value in result["coarse"].items()
    }
    transitions = {"both_correct": 0, "coarse_only": 0, "refined_only": 0, "neither": 0}
    for row in rows:
        a, b = (row[v]["swap_success"] for v in VARIANTS)
        if a is not None:
            transitions[{(True, True): "both_correct", (True, False): "coarse_only",
                         (False, True): "refined_only", (False, False): "neither"}[a, b]] += 1
    result["swap_pairs"] = sum(transitions.values())
    result["swap_transitions"] = transitions
    # Cross-tabulate within instances; separate average improvements can come from different wires.
    def direction(delta):
        if delta > 0:
            return "improved"
        return "worsened" if delta < 0 else "unchanged"

    changes = ("improved", "unchanged", "worsened")
    cross = {box: dict.fromkeys(changes, 0) for box in changes}
    for row in rows:
        box_change = direction(row["refined"]["tile_obb_iou"] - row["coarse"]["tile_obb_iou"])
        cross[box_change][direction(row["mask_iou_delta"])] += 1
    result["box_iou_change_by_mask_iou_change"] = cross
    result["iou_change"] = {mask: sum(counts[mask] for counts in cross.values()) for mask in changes}
    return result


def run_paired(args):
    import torch
    from .learning import TileProgress, infer_crop, load_mask_head, mask_ap

    print(f"[paired] loading {args.weights} and {args.cache}", flush=True)
    checkpoint, model = load_mask_head(args.weights, args.device)
    tiles = WireTiles(args)
    if checkpoint["prompt"] != "axes" or checkpoint["boxes"] != "refined":
        raise ValueError("Use the axes mask head trained with refined boxes")
    if tiles.provenance != checkpoint["detector"]:
        raise ValueError("Mask checkpoint and cached detector provenance do not match")
    margin = checkpoint["margin"]
    rows, detections, n_truth = [], {v: [] for v in VARIANTS}, 0
    proposal_count = 0
    progress = TileProgress(len(tiles.paths))

    def predict(feature):
        return infer_crop(model, feature, args.device, args.threshold)

    with torch.inference_mode():
        for index, path in enumerate(tiles.paths):
            image, _, _, masks, void, pairs, truth_boxes = tiles.read(index)
            entry = tiles.cache[path.name]
            validate_geometry(entry, path.name)
            proposal_count += len(entry["scores"])
            cases, ap_inputs, visible_truth = evaluate_tile(
                image, entry, masks, void, pairs, truth_boxes, margin, predict
            )
            source, video = source_groups(path.stem)
            rows.extend(dict(tile=path.name, source=source, video_prefix=video, **row) for row in cases)
            for v in VARIANTS:
                detections[v].extend((score, index, overlaps) for score, overlaps in ap_inputs[v])
            n_truth += visible_truth
            progress.update(index + 1, "[paired axes]")
    ap = {v: mask_ap(detections[v], n_truth) for v in VARIANTS}
    means = {v: float(np.mean(list(ap[v].values()))) if n_truth else None for v in VARIANTS}
    groups = [source_groups(p.stem) for p in tiles.paths]
    summary = {
        "cohort": "coarse OBB matching; exposure and swap partner fixed on the coarse footprint",
        "swap_scoring_window": "coarse footprint for both predictions and both ground truths",
        "tiles": len(tiles.paths), "ground_truth_instances": n_truth, "proposals": proposal_count,
        "ap_detections_after_void_filter": {v: len(detections[v]) for v in VARIANTS},
        "tile_obb_iou_definition": "rasterized proposal and polygon-fitted GT OBB in the original tile",
        "weights": str(Path(args.weights)), "cache": str(Path(args.cache)), "detector": tiles.provenance,
        "prompt": "axes", "margin_px": margin, "threshold": args.threshold, "match_iou": args.match_iou,
        "mask_AP_by_IoU": ap, "mask_mAP50_95": means,
        "mask_mAP50_95_delta": means["refined"] - means["coarse"] if n_truth else None,
        "matched": summarize(rows), "matched_with_neighbors": summarize([r for r in rows if r["exposed"]]),
        "matched_with_neighbor_within_15deg": summarize([r for r in rows if r["near_parallel"]]),
        "overlap_with_mask_training": {
            "source_images": len(set(s for s, _ in groups).intersection(checkpoint["train_sources"])),
            "video_prefixes": len(set(v for _, v in groups).intersection(checkpoint["train_videos"])),
        },
    }
    save_json(args.output, {**summary, "paired_instances": rows})
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[paired] report and instance deltas: {args.output}", flush=True)
