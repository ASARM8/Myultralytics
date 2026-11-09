"""Paired instance examples, predicted-neighbor diagnostics and native image coordinates."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import WireCrop, annotations, long_axis, match_boxes
from .proposals import image_paths


class WireTiles:
    def __init__(self, args, *, need_labels=True):
        self.paths = image_paths(args.images)
        self.jsons = Path(args.jsons) if need_labels else None
        self.variant = args.boxes
        self.match_iou = args.match_iou
        self.cache = None
        if args.cache:
            payload = json.loads(Path(args.cache).read_text(encoding="utf-8"))
            if payload["version"] != 1:
                raise ValueError("Unsupported proposal cache")
            self.cache = payload["frames"]
            self.provenance = {k: v for k, v in payload.items() if k != "frames"}
            if set(self.cache) != {p.name for p in self.paths}:
                raise ValueError("Proposal cache and image directory contain different files")
        else:
            self.provenance = {"source": "ground-truth geometry only"}
        if args.limit:
            # Spread a quick audit across the directory rather than one contiguous video.
            ids = np.linspace(0, len(self.paths)-1, min(args.limit, len(self.paths)), dtype=int)
            self.paths = [self.paths[i] for i in ids]

    def read(self, index):
        path = self.paths[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        truth, masks, void = [], [], np.zeros(image.size[::-1], dtype=bool)
        if self.jsons is not None:
            truth, masks, void = annotations(self.jsons / f"{path.stem}.json", image.size)
        if self.cache is None:
            return image, truth, [1.] * len(truth), masks, void, dict(enumerate(range(len(truth)))), truth
        entry = self.cache[path.name]
        coarse, boxes, scores = entry["coarse"], entry[self.variant], entry["scores"]
        if tuple(entry["size"]) != image.size or not (len(coarse) == len(boxes) == len(scores)):
            raise ValueError(f"Inconsistent cache entry for {path}")
        for box in boxes:
            long_axis(box)
        pairs = match_boxes(coarse, truth, image.size, self.match_iou) if truth else {}
        return image, boxes, scores, masks, void, pairs, truth


def example(image, boxes, target, kind, margin):
    crop = WireCrop(boxes[target], margin)
    rgb = np.asarray(crop.crop(image), dtype=np.float32).transpose(2, 0, 1) / 255
    feature = np.concatenate([rgb, np.stack(crop.prompts(boxes, target, kind))])
    return crop, feature


def source_groups(stem):
    source = re.sub(r"_x\d+_y\d+$", "", stem)
    return source, source.split("_")[0]


def near_parallel(target, neighbors, truth):
    """GT subgroup for reporting only; orientations are pi-periodic."""
    return any(abs(np.cos(truth[target][4] - truth[j][4])) >= np.cos(np.deg2rad(15)) for j in neighbors)


def audit(args):
    tiles = WireTiles(args)
    matched = exposed = parallel = usable = wires = proposals = 0
    target_pixels = covered = other_pixels = 0
    sources, videos = set(), set()
    for index, path in enumerate(tiles.paths):
        image, boxes, _, masks, void, pairs, truth = tiles.read(index)
        wires += len(masks)
        proposals += len(boxes)
        matched += len(pairs)
        for i, target in pairs.items():
            crop = WireCrop(boxes[i], args.margin)
            footprint = np.asarray(crop.restore(Image.new("L", crop.size, 255), image.size)) > 0
            valid = footprint & ~void
            target_pixels += np.count_nonzero(masks[target] & ~void)
            covered += np.count_nonzero(valid & masks[target])
            neighbors = set()
            foreign = np.zeros_like(void)
            for j, mask in enumerate(masks):
                if j != target and np.any(valid & mask & ~masks[target]):
                    neighbors.add(j)
                    foreign |= mask
            other_pixels += np.count_nonzero(valid & foreign & ~masks[target])
            if neighbors:
                exposed += 1
                parallel += near_parallel(target, neighbors, truth)
                usable += bool(neighbors.intersection(pairs.values()))
                source, video = source_groups(path.stem)
                sources.add(source)
                videos.add(video)
    return {"proposal_source": tiles.provenance, "boxes": tiles.variant if tiles.cache is not None else "truth",
            "margin_px": args.margin, "tiles": len(tiles.paths), "truth_instances": wires, "proposals": proposals,
            "matched": matched, "targets_with_neighbor_pixels": exposed, "with_neighbor_within_15deg": parallel,
            "exposed_source_images": len(sources),
            "exposed_video_prefixes": len(videos), "target_pixel_coverage": covered / target_pixels if target_pixels else None,
            "foreign_over_covered_target_pixels": other_pixels / covered if covered else None,
            "neighbor_detected_fraction": usable / exposed if exposed and tiles.cache is not None else None}
