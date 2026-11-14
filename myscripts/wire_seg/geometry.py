"""Pixel-coordinate geometry for single-wire masks on native-resolution TTPLA tiles."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def long_axis(box):
    x, y, w, h, angle = map(float, box)
    if not all(map(math.isfinite, (x, y, w, h, angle))) or min(w, h) <= 0:
        raise ValueError(f"Invalid xywhr detection: {box}")
    if h > w:
        w, h, angle = h, w, angle + math.pi / 2
    return x, y, w, h, angle % math.pi


def local_frame(box):
    x, y, length, width, angle = long_axis(box)
    along = np.array([math.cos(angle), math.sin(angle)])
    across = np.array([-along[1], along[0]])
    return np.array([x, y]), along, across, length, width


def corners(box):
    center, along, across, length, width = local_frame(box)
    return np.array([center + u * length * along / 2 + v * width * across / 2
                     for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def polygon_box(points):
    """Minimum-area rotated rectangle over convex-hull edge orientations."""
    pts = sorted(set(map(tuple, points)))
    def half(sequence):
        hull = []
        for x, y in sequence:
            while len(hull) > 1:
                ax, ay = hull[-2]
                bx, by = hull[-1]
                if (bx - ax) * (y - ay) > (by - ay) * (x - ax):
                    break
                hull.pop()
            hull.append((x, y))
        return hull

    hull = np.asarray(half(pts)[:-1] + half(reversed(pts))[:-1], dtype=float)
    if len(hull) < 3:
        raise ValueError("Degenerate wire polygon")
    best_area, best = math.inf, None
    for edge in np.roll(hull, -1, axis=0) - hull:
        along = edge / np.linalg.norm(edge)
        axes = np.array([along, [-along[1], along[0]]])
        coords = hull @ axes.T
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        size = hi - lo
        area = float(size.prod())
        if area < best_area:
            center = ((lo + hi) / 2) @ axes
            best = long_axis((*center, *size, math.atan2(along[1], along[0])))
            best_area = area
    return best


def paint(size, polygons):
    image = Image.new("L", size)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        draw.polygon([tuple(p) for p in polygon], fill=255)
    return image


def annotations(path, size):
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if (data["imageWidth"], data["imageHeight"]) != size:
        raise ValueError(f"Polygon/image size mismatch in {path}")
    instances, invisible = {}, []
    for index, shape in enumerate(data["shapes"]):
        label = shape["label"].strip().lower()
        if label != "void" and not any(k in label for k in ("cable", "conductor", "wire")):
            continue
        if shape.get("shape_type", "polygon") != "polygon":
            raise ValueError(f"Expected instance polygons in {path}")
        points = np.asarray(shape["points"], dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError(f"Invalid polygon in {path}, shape {index}")
        if label == "void":
            invisible.append(points)
        else:
            group = shape.get("group_id")
            key = ("shape", index) if group is None else ("group", group)
            instances.setdefault(key, []).append(points)
    masks = [np.asarray(paint(size, parts)) > 0 for parts in instances.values()]
    boxes = [polygon_box(np.concatenate(parts)) for parts in instances.values()]
    return boxes, masks, np.asarray(paint(size, invisible)) > 0


def match_boxes(boxes, truth, size, threshold):
    """Assign once on the coarse detections so all prompt variants see the same identities."""
    def packed_mask(box):
        quad = corners(box)
        x, y = np.maximum(np.floor(quad.min(axis=0)).astype(int), 0)
        r, b = np.minimum(np.floor(quad.max(axis=0)).astype(int) + 1, size)
        if x >= r or y >= b:
            return x, y, np.zeros((0, 0), dtype=bool), 0
        mask = np.asarray(paint((r - x, b - y), [quad - [x, y]])) > 0
        return x, y, mask, int(mask.sum())
    targets = [packed_mask(box) for box in truth]
    pairs = []
    for i, box in enumerate(boxes):
        x, y, m, area = packed_mask(box)
        for j, (gx, gy, g, ga) in enumerate(targets):
            l, t = max(x, gx), max(y, gy)
            r, b = min(x + m.shape[1], gx + g.shape[1]), min(y + m.shape[0], gy + g.shape[0])
            if r <= l or b <= t:
                continue
            overlap = np.count_nonzero(m[t-y:b-y, l-x:r-x] & g[t-gy:b-gy, l-gx:r-gx])
            iou = overlap / max(area + ga - overlap, 1)
            if iou >= threshold:
                pairs.append((iou, i, j))
    assigned, used = {}, set()
    for _, i, j in sorted(pairs, reverse=True):
        if i not in assigned and j not in used:
            assigned[i] = j
            used.add(j)
    return assigned


class WireCrop:
    """Rotate to the proposal axis at one crop pixel per source pixel."""

    def __init__(self, box, margin):
        center, along, across, length, width = local_frame(box)
        self.length, self.width = length, width
        self.size = (max(1, math.ceil(length + 8)), max(1, math.ceil(width + 2 * margin)))
        basis = np.stack([along, across], axis=1)
        # OBB coordinates index pixel centers; Pillow affine coordinates index pixel edges.
        origin = center + 0.5 - basis @ (np.array(self.size) / 2)
        self.matrix = np.column_stack([basis, origin])
        self.inverse = np.column_stack([basis.T, -basis.T @ origin])

    def crop(self, image, *, mask=False):
        mode = Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
        return image.transform(self.size, Image.Transform.AFFINE, self.matrix.ravel(), mode)

    def restore(self, mask, size):
        return mask.transform(size, Image.Transform.AFFINE, self.inverse.ravel(), Image.Resampling.NEAREST)

    def prompts(self, boxes, target, kind):
        """Finite target axis; optionally remove only neighbors' endpoint constraint."""
        xx = np.arange(self.size[0], dtype=np.float32)[None, :] + 0.5
        yy = np.arange(self.size[1], dtype=np.float32)[:, None] + 0.5
        target_map = np.zeros(self.size[::-1], np.float32)
        neighbor = np.zeros_like(target_map)
        for i, box in enumerate(boxes):
            if i != target and kind == "target":
                continue
            center, along, _, length, width = local_frame(box)
            center = self.inverse[:, :2] @ (center + 0.5) + self.inverse[:, 2]
            along = self.inverse[:, :2] @ along
            sigma = max(1.5, min(width / 2, 6))
            infinite = i != target and kind == "axes_infinite"
            radius = abs(along) * length / 2 + width / 2 + 5 * sigma
            # A remote finite segment may still have an infinite supporting line
            # crossing this crop. Do not reject that line by segment bounds.
            if not infinite and (np.any(center + radius < 0) or np.any(center - radius > self.size)):
                continue
            dx, dy = xx - center[0], yy - center[1]
            u = along[0] * dx + along[1] * dy
            v = -along[1] * dx + along[0] * dy
            end = np.maximum(np.abs(u) - length / 2, 0)
            if infinite:
                end = np.zeros_like(end)
            if i != target and kind == "boxes":
                field = (end == 0) & (np.abs(v) <= width / 2)
            else:
                if i != target and kind == "points":
                    end = np.minimum.reduce([np.abs(u - s * length) for s in (-0.4, 0, 0.4)])
                field = np.exp(-(v*v + end*end) / (2*sigma*sigma))
            dest = target_map if i == target else neighbor
            np.maximum(dest, field, out=dest)
        return target_map, neighbor
