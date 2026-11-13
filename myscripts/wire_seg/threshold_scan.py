"""Score several mask cutoffs from one ordinary forward pass per frozen proposal."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from .data import WireTiles, example, match_visible_boxes, near_parallel, source_groups
from .export_failures import sha256
from .learning import MaskStats, TileProgress, crop_probability, load_mask_head, mask_ap, mask_overlaps, neighbor_context
from .neighbor_pairs import GT_MARGIN, summarize_pairs, truth_neighbors
from .paired import validate_geometry
from .proposals import save_json


class ScanMetrics:
    def __init__(self):
        self.detections = []
        self.matched = MaskStats()
        self.exposed = MaskStats()
        self.parallel = MaskStats()
        self.pairs = []

    def result(self, n_truth, prompt):
        ap = mask_ap(self.detections, n_truth)
        return {
            "mask_AP_by_IoU": ap,
            "mask_mAP50_95": float(np.mean(list(ap.values()))) if n_truth else None,
            "ap_detections_after_void_filter": len(self.detections),
            "matched_pixel_precision": self.matched.hits / self.matched.predicted if self.matched.predicted else None,
            "matched": self.matched.result(),
            "matched_with_neighbors": self.exposed.result(),
            "matched_with_neighbor_within_15deg": self.parallel.result(),
            "neighbor_pairs": {
                "all": summarize_pairs(self.pairs, (prompt,)),
                "near_parallel": summarize_pairs([p for p in self.pairs if p["near_parallel"]], (prompt,)),
            },
            "pairs": self.pairs,
        }


def scan_tile(tiles, index, prompt, margin, predict, metrics):
    path = tiles.paths[index]
    image, boxes, scores, masks, void, matches, truth_boxes = tiles.read(index)
    entry = tiles.cache[path.name]
    validate_geometry(entry, path.name)
    masks = [m & ~void for m in masks]
    visible_ids = [j for j, mask in enumerate(masks) if mask.any()]
    foreground = np.logical_or.reduce(masks) if masks else np.zeros_like(void)
    # eval matches all annotations before masking void. neighbor-pairs matches visible IDs only.
    pair_matches = matches
    if len(visible_ids) != len(masks):
        pair_matches = match_visible_boxes(entry["coarse"], truth_boxes, visible_ids, image.size, tiles.match_iou)
    proposals = {truth: proposal for proposal, truth in pair_matches.items()}
    pairs = truth_neighbors(masks, truth_boxes, image.size)
    pair_ious = {t: {} for t in metrics}
    for i, score in enumerate(scores):
        crop, features = example(image, boxes, i, prompt, margin)
        probability = predict(features)
        # Nearest-neighbor restoration commutes with thresholding at a positive cutoff.
        full_prob = np.asarray(crop.restore(Image.fromarray(probability), image.size))
        target_id = matches.get(i)
        matched = target_id is not None and masks[target_id].any()
        if matched:
            target = masks[target_id]
            foreign = foreground & ~target
            footprint, parallel = neighbor_context(crop, image.size, foreign, masks, target_id, truth_boxes)
        for threshold, stats in metrics.items():
            raw = full_prob >= threshold
            pred = raw & ~void
            # neighbor-pairs assigns IoU zero even when a whole prediction is void.
            has_visible_pixels = pred.any()
            overlaps = mask_overlaps(pred, masks) if has_visible_pixels else []
            if i in pair_matches:
                truth = pair_matches[i]
                pair_ious[threshold][truth] = next((v for j, v in overlaps if j == truth), 0.)
            if not has_visible_pixels and raw.any():
                continue
            stats.detections.append((score, index, overlaps))
            if matched:
                stats.matched.add(pred, target, foreign)
                if footprint is not None:
                    stats.exposed.add(pred, target, foreign)
                    if parallel:
                        stats.parallel.add(pred, target, foreign)

    for a, b in pairs:
        identity = {"tile": path.name, "truth_ids": [a, b], "proposal_ids": [proposals.get(a), proposals.get(b)],
                    "near_parallel": near_parallel(a, [b], truth_boxes)}
        for threshold, stats in metrics.items():
            values = pair_ious[threshold]
            stats.pairs.append({**identity, "mask_ious": {prompt: [values.get(a), values.get(b)]}})
    return len(visible_ids), len(boxes)


@torch.inference_mode()
def run_scan(args):
    print(f"[threshold-scan] loading {args.weights} and {args.cache}", flush=True)
    checkpoint, model = load_mask_head(args.weights, args.device)
    tiles = WireTiles(args)
    if checkpoint["boxes"] != "refined" or checkpoint["detector"] != tiles.provenance:
        raise ValueError("Expected a mask head trained with the cached refined detector")
    prompt, margin = checkpoint["prompt"], checkpoint["margin"]
    metrics = {t: ScanMetrics() for t in args.thresholds}
    print(f"[threshold-scan {prompt}] {len(tiles.paths)} tiles | mask thresholds {list(metrics)}", flush=True)
    progress = TileProgress(len(tiles.paths))
    n_truth = n_proposals = 0

    def predict(features):
        return crop_probability(model, features, args.device)

    for index in range(len(tiles.paths)):
        truths, proposals = scan_tile(tiles, index, prompt, margin, predict, metrics)
        n_truth += truths
        n_proposals += proposals
        progress.update(index + 1, f"[threshold-scan {prompt}]", f"| proposals {n_proposals}")

    results = []
    for threshold, stats in metrics.items():
        row = stats.result(n_truth, prompt)
        results.append({"mask_threshold": threshold, **row})
        pairs = row["neighbor_pairs"]["all"]
        success = pairs["by_mask_iou"]["0.50"][prompt]["successful_pairs"]
        print(f"[threshold-scan] {threshold:g} | mAP {row['mask_mAP50_95']} | "
              f"neighbor leakage (exposed) {row['matched_with_neighbors']['neighbor_pixel_leakage']} | "
              f"both IoU >= 0.50: {success}/{pairs['gt_pairs']}", flush=True)
    groups = [source_groups(p.stem) for p in tiles.paths]
    save_json(args.output, {
        "prompt": prompt, "boxes": args.boxes, "margin_px": margin, "mask_thresholds": args.thresholds,
        "coarse_obb_match_iou": args.match_iou, "gt_neighbor_margin_px": GT_MARGIN,
        "images": args.images, "jsons": args.jsons, "limit": args.limit,
        "tiles": len(tiles.paths), "ground_truth_instances": n_truth, "proposals": n_proposals,
        "weights": args.weights, "weights_sha256": sha256(args.weights), "seed": checkpoint["seed"],
        "epoch": checkpoint["epoch"], "cache": args.cache, "cache_sha256": sha256(args.cache),
        "detector": tiles.provenance,
        "overlap_with_mask_training": {
            "source_images": len({s for s, _ in groups}.intersection(checkpoint["train_sources"])),
            "video_prefixes": len({v for _, v in groups}.intersection(checkpoint["train_videos"])),
        },
        "results": results,
    })
    print(f"[threshold-scan] report: {args.output}", flush=True)
