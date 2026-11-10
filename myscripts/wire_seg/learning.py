"""Train only the per-instance mask head; predict and score masks in tile coordinates."""
from __future__ import annotations

import os
import random
import tempfile
from time import monotonic
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from .data import WireTiles, example, near_parallel, source_groups
from .batching import InstanceBatches, WireBatchDataset
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

    def forward(self, x, extent=None):
        if extent is None:
            return super().forward(x)
        # GroupNorm must see the original crop only. Clear its padded border
        # after every block so the next dilation sees the usual zero boundary.
        pixels = extent.sum((2, 3), dtype=torch.float32)
        for offset in range(0, len(self) - 1, 3):
            x = self[offset](x)
            norm = self[offset + 1]
            n, c, h, w = x.shape
            group = x.float().reshape(n, norm.num_groups, c // norm.num_groups, h, w)
            keep = extent[:, :, None]
            denom = pixels * (c // norm.num_groups)
            mean = (group * keep).sum((2, 3, 4)) / denom
            centered = (group - mean[:, :, None, None, None]) * keep
            var = centered.square().sum((2, 3, 4)) / denom
            normalized = (centered * torch.rsqrt(var + norm.eps)[:, :, None, None, None]).reshape_as(x)
            affine = normalized * norm.weight[None, :, None, None] + norm.bias[None, :, None, None]
            x = self[offset + 2](affine.to(x.dtype)) * extent
        return self[-1](x)


def plane(array, device):
    return torch.from_numpy(array).unsqueeze(0).to(device)


class TileProgress:
    """Flush a line on the first tile, every 15 seconds, and on completion."""

    def __init__(self, total, unit="tiles"):
        self.total = total
        self.unit = unit
        self.started = self.last_log = monotonic()
        self.first = True

    def update(self, done, label, detail=""):
        now = monotonic()
        if not self.first and done != self.total and now - self.last_log < 15:
            return
        self.first = False
        self.last_log = now
        elapsed = max(now - self.started, 1e-6)
        rate = done / elapsed
        hours, remainder = divmod(round((self.total - done) / rate), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"{label} | {self.unit} {done}/{self.total} ({100*done/self.total:.1f}%) | "
              f"{rate:.2f} {self.unit}/s | ETA {hours:02d}:{minutes:02d}:{seconds:02d} {detail}", flush=True)


def instance_losses(logits, labels):
    """One BCE plus foreground-only Dice loss per instance regardless of crop size."""
    keep = labels >= 0
    truth = (labels == 1).float()
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, truth, reduction="none")
    bce = (bce * keep).sum((1, 2)) / keep.sum((1, 2)).clamp_min(1)
    prob = logits.sigmoid() * keep
    area = truth.sum((1, 2))
    dice = 1 - (2 * (prob * truth).sum((1, 2)) + 1) / (prob.sum((1, 2)) + area + 1)
    return bce + dice * (area > 0), area > 0


def train(args):
    print(f"[train {args.prompt}] loading proposal cache {args.cache}", flush=True)
    tiles = WireTiles(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = WireMaskHead().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    device = torch.device(args.device)
    cuda = device.type == "cuda"
    amp = args.amp and cuda
    amp_dtype = torch.float16
    if amp:
        with torch.cuda.device(device):
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)
    sampler = InstanceBatches(tiles, args)
    if not sampler.instances:
        raise ValueError("No cached training proposals")
    loader = torch.utils.data.DataLoader(
        WireBatchDataset(tiles, sampler, args), batch_size=None, sampler=sampler,
        num_workers=args.workers, pin_memory=cuda, persistent_workers=args.workers > 0,
        prefetch_factor=1 if args.workers else None,
    )
    precision = str(amp_dtype) if amp else "float32"
    print(f"[train {args.prompt}] batch limit {args.batch_size} | workers {args.workers} | "
          f"pixel budget {args.max_batch_pixels} | {precision}", flush=True)
    progress = TileProgress(args.epochs * sampler.instances, "proposals")
    done = 0
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(args.epochs):
        sampler.epoch = epoch
        loss_sum = count = positives = 0
        print(f"[train {args.prompt}] starting epoch {epoch+1}/{args.epochs}, {len(tiles.paths)} tiles", flush=True)
        for features, labels, padded, proposed in loader:
            done += proposed
            if not len(features):
                continue
            features = features.to(device, non_blocking=cuda)
            labels = labels.to(device, non_blocking=cuda)
            extent = (labels != -2)[:, None] if padded else None
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
                logits = model(features, extent)[:, 0]
            losses, foreground = instance_losses(logits, labels)
            scaler.scale(losses.mean()).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_loss, batch_positives = torch.stack((losses.detach().sum(), foreground.sum())).tolist()
            positives += int(batch_positives)
            count += len(features)
            loss_sum += batch_loss
            detail = f"| batch {len(features)} | mean loss {loss_sum/count:.5f}"
            if cuda:
                detail += f" | peak allocated {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB"
            progress.update(done, f"[train {args.prompt}] epoch {epoch+1}/{args.epochs}", detail)
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
                        "batch_size": args.batch_size, "max_batch_pixels": args.max_batch_pixels, "precision": precision,
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


def load_mask_head(weights, device):
    checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
    if checkpoint["format"] != 1:
        raise ValueError("Unsupported wire mask checkpoint")
    model = WireMaskHead().to(device).eval()
    model.load_state_dict(checkpoint["state"])
    return checkpoint, model


def infer_crop(model, feature, device, threshold):
    return model(plane(feature, device))[0, 0].sigmoid().cpu().numpy() >= threshold


@torch.inference_mode()
def run_masks(args):
    evaluate = args.command == "eval"
    print(f"[{args.command}] loading {args.weights} and {args.cache}", flush=True)
    checkpoint, model = load_mask_head(args.weights, args.device)
    args.boxes = args.boxes or checkpoint["boxes"]
    tiles = WireTiles(args, need_labels=evaluate)
    if tiles.provenance != checkpoint["detector"]:
        raise ValueError("Mask checkpoint and cached detector provenance do not match")
    groups = [source_groups(p.stem) for p in tiles.paths]
    shared_sources = set(s for s, _ in groups).intersection(checkpoint["train_sources"])
    shared_videos = set(v for _, v in groups).intersection(checkpoint["train_videos"])
    output = Path(args.output)
    stats, exposed, parallel = MaskStats(), MaskStats(), MaskStats()
    swap_trials = swap_wins = 0
    detections, results, n_truth = [], {}, 0
    progress = TileProgress(len(tiles.paths))
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
            pred = infer_crop(model, feature, args.device, args.threshold)
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
                            alternative = infer_crop(model, swapped, args.device, args.threshold)
                            switched = np.asarray(crop.restore(Image.fromarray(alternative), image.size)) & ~void
                            a, b = target & footprint, masks[other_id] & footprint
                            def iou(mask, truth):
                                return np.count_nonzero(mask & truth) / max(1, np.count_nonzero(mask | truth))
                            swap_trials += 1
                            swap_wins += int(iou(visible, a) > iou(visible, b) and iou(switched, b) > iou(switched, a))
        progress.update(index + 1, f"[{args.command} {checkpoint['prompt']}]")
    if not evaluate:
        save_json(output / "instances.json", results)
        return
    print(f"[eval {checkpoint['prompt']}] computing final mask AP for {len(detections)} detections", flush=True)
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
