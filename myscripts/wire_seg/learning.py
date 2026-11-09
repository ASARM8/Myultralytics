"""Train only the per-instance mask head; predict and score masks in tile coordinates."""
from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from .data import WireTiles, example, near_parallel, source_groups
from .proposals import save_json


class WireMaskHead(nn.Sequential):
    def __init__(self):
        layers = []
        channels = 5  # RGB, target-axis prompt, neighboring-instance prompt
        for dilation in (1, 2, 4, 2):
            layers.extend([nn.Conv2d(channels, 24, 3, dilation=dilation, padding=dilation),
                           nn.GroupNorm(4, 24), nn.SiLU()])
            channels = 24
        layers.append(nn.Conv2d(24, 1, 1))
        super().__init__(*layers)


def plane(array, device):
    return torch.from_numpy(array).unsqueeze(0).to(device)


def train(args):
    tiles = WireTiles(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = WireMaskHead().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        order = list(range(len(tiles.paths)))
        random.shuffle(order)
        loss_sum = count = positives = 0
        for index in order:
            image, boxes, _, masks, void, pairs, _ = tiles.read(index)
            ids = list(range(len(boxes)))
            random.shuffle(ids)
            for i in ids:
                crop, feature = example(image, boxes, i, args.prompt, args.margin)
                target = masks[pairs[i]] if i in pairs else np.zeros_like(void)
                # Ignore phantom padding as well as void pixels.
                valid = np.asarray(crop.crop(Image.fromarray(~void), mask=True)).copy()
                if not valid.any():
                    continue
                truth = np.asarray(crop.crop(Image.fromarray(target), mask=True), dtype=np.float32)
                keep = plane(valid, args.device)
                y = plane(truth.copy(), args.device)[keep]
                pred = model(plane(feature, args.device))[:, 0][keep]
                loss = F.binary_cross_entropy_with_logits(pred, y)
                if y.any():
                    prob = pred.sigmoid()
                    loss = loss + 1 - (2 * (prob*y).sum() + 1) / (prob.sum() + y.sum() + 1)
                    positives += 1
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                count += 1
                loss_sum += float(loss.detach())
        if not positives:
            raise ValueError("No matched foreground pixels in training; check proposal coordinates, polygons and matching")
        print(f"epoch {epoch+1}  instances {count}  positives {positives}  loss {loss_sum/count:.5f}", flush=True)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            name = tmp.name
        try:
            torch.save({"format": 1, "state": model.state_dict(), "prompt": args.prompt, "margin": args.margin,
                        "boxes": args.boxes, "detector": tiles.provenance, "epoch": epoch+1, "seed": args.seed,
                        "train_sources": sorted({source_groups(p.stem)[0] for p in tiles.paths}),
                        "train_videos": sorted({source_groups(p.stem)[1] for p in tiles.paths})}, name)
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)


def mask_ap(detections, n_truth):
    """101-point mask AP; each detection may claim one unmatched instance in its own tile."""
    values = {}
    ranked = sorted(detections, key=lambda item: -item[0])
    for threshold in range(50, 100, 5):
        used, flags = set(), []
        for _, tile, overlaps in ranked:
            hit = next((j for j, iou in overlaps if iou >= threshold/100 and (tile, j) not in used), None)
            flags.append(hit is not None)
            if hit is not None:
                used.add((tile, hit))
        if not n_truth:
            values[str(threshold)] = None
            continue
        tp = np.cumsum(flags, dtype=float)
        precision = tp / np.arange(1, len(tp)+1)
        envelope = np.maximum.accumulate(np.r_[precision, 0.][::-1])[::-1]
        samples = np.searchsorted(tp / n_truth, np.linspace(0, 1, 101))
        values[str(threshold)] = float(envelope[samples].mean())
    return values


class MaskStats:
    def __init__(self):
        self.count = self.predicted = self.target = self.hits = self.leaks = 0
        self.iou_sum = 0.

    def add(self, pred, truth, foreign):
        hit = np.count_nonzero(pred & truth)
        self.hits += hit
        self.target += np.count_nonzero(truth)
        self.predicted += np.count_nonzero(pred)
        self.leaks += np.count_nonzero(pred & foreign & ~truth)
        self.iou_sum += hit / max(np.count_nonzero(pred | truth), 1)
        self.count += 1

    def result(self):
        return {"instances": self.count, "mean_instance_iou": self.iou_sum/self.count if self.count else None,
                "target_pixel_recall": self.hits/self.target if self.target else None,
                "neighbor_pixel_leakage": self.leaks/self.predicted if self.predicted else None}


@torch.inference_mode()
def run_masks(args):
    evaluate = args.command == "eval"
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    if checkpoint["format"] != 1:
        raise ValueError("Unsupported wire mask checkpoint")
    args.boxes = args.boxes or checkpoint["boxes"]
    tiles = WireTiles(args, need_labels=evaluate)
    if tiles.provenance != checkpoint["detector"]:
        raise ValueError("Mask checkpoint and cached detector provenance do not match")
    model = WireMaskHead().to(args.device).eval()
    model.load_state_dict(checkpoint["state"])
    groups = [source_groups(p.stem) for p in tiles.paths]
    shared_sources = set(s for s, _ in groups).intersection(checkpoint["train_sources"])
    shared_videos = set(v for _, v in groups).intersection(checkpoint["train_videos"])
    output = Path(args.output)
    stats, exposed, parallel = MaskStats(), MaskStats(), MaskStats()
    swap_trials = swap_wins = 0
    detections, results, n_truth = [], {}, 0
    for index, path in enumerate(tiles.paths):
        image, boxes, scores, masks, void, pairs, truth_boxes = tiles.read(index)
        masks = [m & ~void for m in masks]
        n_truth += sum(bool(m.any()) for m in masks)
        counts = sum((m.astype(np.int32) for m in masks), np.zeros_like(void, dtype=np.int32))
        if not evaluate:
            (output / path.stem).mkdir(parents=True, exist_ok=True)
            results[path.name] = []
        for i, box in enumerate(boxes):
            crop, feature = example(image, boxes, i, checkpoint["prompt"], checkpoint["margin"])
            pred = model(plane(feature, args.device))[0, 0].sigmoid().cpu().numpy() >= args.threshold
            restored = crop.restore(Image.fromarray(pred), image.size)
            if not evaluate:
                file = Path(path.stem) / f"{i:04d}.png"
                restored.save(output / file)
                results[path.name].append({"mask": file.as_posix(), "score": scores[i], "box_xywhr": box})
                continue
            visible = np.asarray(restored) & ~void
            if not visible.any() and np.asarray(restored).any():
                continue  # an entire prediction fell inside ignored pixels
            overlaps = []
            for j, target in enumerate(masks):
                intersection = np.count_nonzero(visible & target)
                if intersection:
                    overlaps.append((j, intersection / np.count_nonzero(visible | target)))
            detections.append((scores[i], index, sorted(overlaps, key=lambda pair: -pair[1])))
            if i in pairs:
                target = masks[pairs[i]]
                if not target.any():
                    continue
                others = (counts > 0) & ~target
                stats.add(visible, target, others)
                local_neighbors = crop.crop(Image.fromarray(others), mask=True)
                if local_neighbors.getbbox() is not None:
                    exposed.add(visible, target, others)
                    footprint = np.asarray(crop.restore(Image.new("1", crop.size, 1), image.size))
                    neighbors = [j for j, m in enumerate(masks) if j != pairs[i] and np.any(m & ~target & footprint)]
                    if near_parallel(pairs[i], neighbors, truth_boxes):
                        parallel.add(visible, target, others)
                    if args.swap_target:
                        # Choose the matched neighbor with most exclusive visible pixels.
                        # Ground truth selects diagnostic pairs only, never a mask input.
                        choices = []
                        for k, other_id in pairs.items():
                            if k == i:
                                continue
                            local = crop.crop(Image.fromarray(masks[other_id] & ~target), mask=True)
                            area = np.count_nonzero(local)
                            if area:
                                choices.append((area, k, other_id))
                        if choices:
                            _, k, other_id = max(choices)
                            swapped = feature.copy()
                            swapped[3:] = np.stack(crop.prompts(boxes, k, checkpoint["prompt"]))
                            alternative = model(plane(swapped, args.device))[0, 0].sigmoid().cpu().numpy()
                            switched = np.asarray(crop.restore(Image.fromarray(alternative >= args.threshold), image.size)) & ~void
                            a, b = target & footprint, masks[other_id] & footprint
                            def iou(mask, truth):
                                return np.count_nonzero(mask & truth) / max(1, np.count_nonzero(mask | truth))
                            swap_trials += 1
                            swap_wins += int(iou(visible, a) > iou(visible, b) and iou(switched, b) > iou(switched, a))
    if not evaluate:
        save_json(output / "instances.json", results)
        return
    ap = mask_ap(detections, n_truth)
    report = {"boxes": args.boxes, "prompt": checkpoint["prompt"], "margin_px": checkpoint["margin"],
              "tiles": len(tiles.paths), "ground_truth_instances": n_truth, "mask_AP_by_IoU": ap,
              "mask_mAP50_95": float(np.mean(list(ap.values()))) if n_truth else None,
              "matched": stats.result(), "matched_with_neighbors": exposed.result(),
              "matched_with_neighbor_within_15deg": parallel.result(),
              "swap_target_pairs": swap_trials, "swap_both_correct_fraction": swap_wins/swap_trials if swap_trials else None,
              "overlap_with_mask_training": {"source_images": len(shared_sources), "video_prefixes": len(shared_videos)}}
    print(report)
    save_json(output, report)
