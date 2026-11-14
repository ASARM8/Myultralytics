"""Cache frozen CA/Refine proposals in original image coordinates, without reading labels."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def image_paths(folder):
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
    if not paths or len({p.stem for p in paths}) != len(paths):
        raise ValueError(f"Expected images with unique stems in {folder}")
    return paths


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False)
    try:
        with f:
            json.dump(data, f, allow_nan=False, ensure_ascii=False)
        os.replace(f.name, path)
    finally:
        Path(f.name).unlink(missing_ok=True)


def infer_frame(bundle, image, wire_class, measure=None):
    """Run the same online geometry path for cache creation and timing.

    ``measure(name, operation)`` optionally times a stage without changing its
    outputs. Coordinates, class filtering and residual gain match the cache.
    """
    import cv2
    import torch
    from myscripts.V3_1_1_low_gain.config import DEFAULT_RESIDUAL_SCALE

    measure = measure or (lambda _name, operation: operation())
    w, h = image.size
    gain = 640 / max(h, w)
    rw, rh = max(1, round(w * gain)), max(1, round(h * gain))
    left, top = (640 - rw) // 2, (640 - rh) // 2

    def prepare():
        canvas = np.full((640, 640, 3), 114, np.uint8)
        canvas[top:top + rh, left:left + rw] = cv2.resize(
            np.asarray(image), (rw, rh), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(canvas).permute(2, 0, 1)[None]

    batch = measure("detector_input_ms", prepare)
    imgs, p2, p3, detections = measure("detector_and_nms_ms", lambda: bundle.extractor.infer({"img": batch}))
    detection = detections[0]
    boxes, scores = detection["bboxes"], detection["conf"]

    def refine():
        if not len(boxes):
            return boxes.clone()
        output = bundle.refiner(p2, p3, boxes[None], scores[None], imgs.shape[2:])
        return bundle.refiner.apply_residual(boxes, DEFAULT_RESIDUAL_SCALE * output["residual"][0])

    refined = measure("refine_ms", refine)

    def download():
        keep = detection["cls"] == wire_class

        def pixels(tensor):
            original = tensor[keep].float().clone()
            original[:, 0] -= left
            original[:, 1] -= top
            original[:, :4] /= gain
            return original.cpu().tolist()

        return {"size": [w, h], "coarse": pixels(boxes), "refined": pixels(refined),
                "scores": scores[keep].cpu().tolist()}

    return measure("geometry_transfer_ms", download), tuple(imgs.shape[2:])


def cache_proposals(args):
    import torch
    from myscripts.V3_1_1.evidence_runtime import load_refine_bundle
    from myscripts.V3_1_1_low_gain.config import DEFAULT_RESIDUAL_SCALE

    paths = image_paths(args.images)
    bundle = load_refine_bundle(args.refine, args.ca, device_arg=args.device, amp=False, imgsz=640)
    try:
        names = bundle.ca_model.names
        if len(names) > 1 and args.wire_class is None:
            raise ValueError("Specify --wire-class for a multi-class detector")
        wire_class = args.wire_class if args.wire_class is not None else 0
        if not 0 <= wire_class < len(names):
            raise ValueError(f"No wire class {wire_class} in {names}")
        frames = {}
        with torch.inference_mode():
            for path in paths:
                with Image.open(path) as source:
                    image = source.convert("RGB")
                frames[path.name], _ = infer_frame(bundle, image, wire_class)
        save_json(args.output, {"version": 1, "ca_sha256": bundle.ca_hash, "refine_sha256": bundle.checkpoint_hash,
                                "scale": DEFAULT_RESIDUAL_SCALE, "wire_class": wire_class, "frames": frames})
    finally:
        bundle.close()
