"""Prepare native-resolution wire crops in CPU workers for a batched mask-head update."""
from __future__ import annotations

import random

import numpy as np
from PIL import Image

from .data import example
from .geometry import WireCrop


class InstanceBatches:
    """Keep proposals from each tile together when possible to amortize image decoding."""

    def __init__(self, tiles, args):
        self.sizes = [
            [WireCrop(box, args.margin).size for box in tiles.cache[p.name][tiles.variant]]
            for p in tiles.paths
        ]
        self.instances = sum(map(len, self.sizes))
        self.batch_size = args.batch_size
        self.max_pixels = args.max_batch_pixels
        self.seed = args.seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        order = list(range(len(self.sizes)))
        rng.shuffle(order)
        batch = []
        width = height = 0
        for tile in order:
            ids = list(range(len(self.sizes[tile])))
            rng.shuffle(ids)
            for i in ids:
                w, h = self.sizes[tile][i]
                pixels = (len(batch) + 1) * max(w, width) * max(h, height)
                if batch and (len(batch) >= self.batch_size or pixels > self.max_pixels):
                    yield batch
                    batch = []
                    width = height = 0
                batch.append((tile, i))
                width, height = max(w, width), max(h, height)
        if batch:
            yield batch


class WireBatchDataset:
    def __init__(self, tiles, sampler, args):
        self.tiles = tiles
        self.sizes = sampler.sizes
        self.prompt = args.prompt
        self.margin = args.margin

    def __getitem__(self, indices):
        import torch

        shapes = [self.sizes[tile][i] for tile, i in indices]
        w, h = map(max, zip(*shapes))
        features = np.zeros((len(indices), 5, h, w), np.float32)
        # -2: outside this crop; -1: ignored source pixel or outside the source image.
        labels = np.full((len(indices), h, w), -2, np.int8)
        previous = None
        rows = 0
        for tile, i in indices:
            if tile != previous:
                image, boxes, _, masks, void, pairs, _ = self.tiles.read(tile)
                valid_image = Image.fromarray(~void)
                previous = tile
            crop, feature = example(image, boxes, i, self.prompt, self.margin)
            cw, ch = crop.size
            valid = np.asarray(crop.crop(valid_image, mask=True))
            if not valid.any():
                continue
            features[rows, :, :ch, :cw] = feature
            target = labels[rows, :ch, :cw]
            if i in pairs:
                target[:] = crop.crop(Image.fromarray(masks[pairs[i]]), mask=True)
            else:
                target.fill(0)
            target[~valid] = -1
            rows += 1
        padded = any(size != (w, h) for size in shapes)
        return torch.from_numpy(features[:rows]), torch.from_numpy(labels[:rows]), padded, len(indices)
