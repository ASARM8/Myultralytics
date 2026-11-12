"""Evaluate ordinary per-proposal masks on an annotation-defined cohort of neighboring wires."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .data import WireTiles, example, near_parallel, source_groups
from .geometry import WireCrop
from .paired import footprint, iou, validate_geometry
from .proposals import save_json


PROMPTS = ("target", "points", "boxes", "axes")
GT_MARGIN = 12
IOU_THRESHOLDS = (0.50, 0.75)


def truth_neighbors(masks, truth_boxes, size):
    """An unordered pair qualifies if either GT strip contains exclusive visible pixels of the other."""
    visible = [j for j, mask in enumerate(masks) if mask.any()]
    neighbors = set()
    for j in visible:
        strip = footprint(WireCrop(truth_boxes[j], GT_MARGIN), size) & ~masks[j]
        for k in visible:
            if j != k and np.any(strip & masks[k]):
                neighbors.add(tuple(sorted((j, k))))
    return sorted(neighbors)


def evaluate_pairs(image, boxes, masks, void, matches, truth_boxes, margin, predict):
    """Reuse fixed coarse-box identities. Each proposal is scored on its own normal refined crop."""
    masks = [mask & ~void for mask in masks]
    pairs = truth_neighbors(masks, truth_boxes, image.size)
    proposal_for = {truth: proposal for proposal, truth in matches.items()}
    scores = {prompt: {} for prompt in PROMPTS}
    for truth in sorted({j for pair in pairs for j in pair}):
        proposal = proposal_for.get(truth)
        if proposal is None:
            continue
        for prompt in PROMPTS:
            # All post-NMS boxes are present in the prompt, including false positives.
            crop, features = example(image, boxes, proposal, prompt, margin)
            pred = Image.fromarray(predict(prompt, features))
            restored = np.asarray(crop.restore(pred, image.size)) & ~void
            scores[prompt][truth] = float(iou(restored, masks[truth]))

    return [
        {"truth_ids": [a, b], "proposal_ids": [proposal_for.get(a), proposal_for.get(b)],
         "near_parallel": near_parallel(a, [b], truth_boxes),
         "mask_ious": {p: [scores[p].get(a), scores[p].get(b)] for p in PROMPTS}}
        for a, b in pairs
    ]


def summarize_pairs(rows):
    total = len(rows)
    detected = sum(None not in row["proposal_ids"] for row in rows)
    results = {}
    for threshold in IOU_THRESHOLDS:
        key = f"{threshold:.2f}"
        successes = {p: set() for p in PROMPTS}
        for index, row in enumerate(rows):
            for prompt, (a, b) in row["mask_ious"].items():
                if a is not None and b is not None and min(a, b) >= threshold:
                    successes[prompt].add(index)
        axes = successes["axes"]
        by_prompt = {}
        for prompt, passed in successes.items():
            by_prompt[prompt] = {"successful_pairs": len(passed),
                                 "fraction_all": len(passed) / total if total else None,
                                 "fraction_both_detected": len(passed) / detected if detected else None}
            if prompt != "axes":
                by_prompt[prompt]["axes_only"] = len(axes - passed)
                by_prompt[prompt]["comparator_only"] = len(passed - axes)
        results[key] = by_prompt
    return {"gt_pairs": total, "both_detected": detected, "missing_one_or_both": total - detected,
            "detection_coverage": detected / total if total else None, "by_mask_iou": results}


def run_neighbor_pairs(args):
    import torch
    from .learning import TileProgress, infer_crop, load_mask_head

    print(f"[neighbor-pairs] loading {args.cache}", flush=True)
    tiles = WireTiles(args)
    groups = [source_groups(p.stem) for p in tiles.paths]
    sources, videos = {s for s, _ in groups}, {v for _, v in groups}
    models, weights = {}, {}
    margin = None
    for prompt in PROMPTS:
        path = Path(args.weights_dir) / prompt / "mask.pt"
        print(f"[neighbor-pairs] loading {path}", flush=True)
        ckpt, models[prompt] = load_mask_head(path, args.device)
        if ckpt["prompt"] != prompt or ckpt["boxes"] != "refined" or ckpt["detector"] != tiles.provenance:
            raise ValueError(f"{path}: expected {prompt} head trained with the cached refined detector")
        if margin is not None and ckpt["margin"] != margin:
            raise ValueError(f"{path}: all four heads must use the same crop margin")
        margin = ckpt["margin"]
        weights[prompt] = {"path": str(path), "epoch": ckpt["epoch"], "seed": ckpt["seed"],
                           "overlapping_train_sources": len(sources.intersection(ckpt["train_sources"])),
                           "overlapping_train_videos": len(videos.intersection(ckpt["train_videos"]))}

    def predict(prompt, features):
        return infer_crop(models[prompt], features, args.device, args.threshold)

    rows = []
    progress = TileProgress(len(tiles.paths))
    with torch.inference_mode():
        for index, path in enumerate(tiles.paths):
            image, boxes, _, masks, void, matches, truth = tiles.read(index, match_visible_only=True)
            validate_geometry(tiles.cache[path.name], path.name)
            cases = evaluate_pairs(image, boxes, masks, void, matches, truth, margin, predict)
            rows.extend({"tile": path.name, **case} for case in cases)
            progress.update(index + 1, "[neighbor-pairs]", f"| GT pairs {len(rows)}")

    summary = {"all": summarize_pairs(rows),
               "near_parallel": summarize_pairs([row for row in rows if row["near_parallel"]])}
    report = {
        "pair_definition": "Unordered GT instances in one tile; either GT OBB strip contains at least "
                           "one non-void pixel exclusive to the other instance relative to the first.",
        "gt_strip_short_side_margin_px": GT_MARGIN, "gt_strip_long_side_margin_px": 4,
        "parallel_degrees": 15, "coarse_obb_match_iou": args.match_iou,
        "mask_threshold": args.threshold, "mask_crop_margin_px": margin,
        "prediction": "One-to-one coarse OBB assignment; own refined crop per proposal, full-tile visible mask IoU.",
        "cache": str(args.cache), "detector": tiles.provenance, "weights": weights,
        "tiles": len(tiles.paths), "summary": summary, "pairs": rows,
    }
    save_json(args.output, report)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[neighbor-pairs] report: {args.output}", flush=True)
