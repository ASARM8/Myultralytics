"""Run directly with python -m unittest discover -s tests -p test_wire_instance_seg.py."""
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from myscripts.wire_seg.data import audit
from myscripts.wire_seg.batching import InstanceBatches
from myscripts.wire_seg.geometry import WireCrop, annotations, corners, match_boxes, paint, polygon_box
from myscripts.wire_seg.proposals import save_json
from train_wire_seg import main, parser


class WireGeometryTest(unittest.TestCase):
    def test_batch_pixel_budget_preserves_all_instances(self):
        from types import SimpleNamespace

        paths = [Path("a.png"), Path("b.png")]
        tiles = SimpleNamespace(paths=paths, variant="refined", cache={
            p.name: {"refined": [[0, 0, w, 3, 0] for w in (20, 30, 50, 500)]} for p in paths})
        args = SimpleNamespace(margin=4, seed=4, batch_size=3, max_batch_pixels=1000)
        sampler = InstanceBatches(tiles, args)
        batches = list(sampler)
        self.assertEqual(sorted(pair for batch in batches for pair in batch), [(t, i) for t in range(2) for i in range(4)])
        self.assertEqual(batches, list(sampler))
        self.assertTrue(any(len(batch) > 1 for batch in batches))
        for batch in batches:
            w, h = map(max, zip(*(sampler.sizes[t][i] for t, i in batch)))
            self.assertLessEqual(len(batch), 3)
            self.assertTrue(len(batch) == 1 or w * h * len(batch) <= args.max_batch_pixels)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Run batched model parity on the server")
    def test_padded_batch_matches_separate_crops_and_gradients(self):
        import torch
        from myscripts.wire_seg.learning import WireMaskHead, instance_losses

        model = WireMaskHead()
        shapes = [(17, 33), (11, 23)]
        features = torch.zeros(2, 5, 17, 33)
        labels = torch.full((2, 17, 33), -2, dtype=torch.int8)
        separate = []
        for i, (h, w) in enumerate(shapes):
            crop = torch.randn(1, 5, h, w)
            features[i, :, :h, :w] = crop[0]
            labels[i, :h, :w] = i
            labels[i, 0, :w] = -1
            separate.append(model(crop)[:, 0])
        together = model(features, (labels != -2)[:, None])[:, 0]
        losses, _ = instance_losses(together, labels)
        reference = []
        for i, (h, w) in enumerate(shapes):
            torch.testing.assert_close(together[i, :h, :w], separate[i][0], atol=1e-5, rtol=1e-4)
            reference.extend(instance_losses(separate[i], labels[i:i+1, :h, :w])[0].unbind())
        grad = torch.autograd.grad(losses.mean(), model[0].weight)[0]
        expected = torch.autograd.grad(torch.stack(reference).mean(), model[0].weight)[0]
        torch.testing.assert_close(grad, expected, atol=1e-5, rtol=1e-4)

    def test_oriented_crop_and_prompt_switch(self):
        boxes = [(40, 40, 50, 3, math.pi/4), (36, 44, 50, 3, math.pi/4)]
        crop = WireCrop(boxes[0], 12)
        own, other = crop.prompts(boxes, 0, "axes")
        reverse_own, reverse_other = crop.prompts(boxes, 1, "axes")
        np.testing.assert_array_equal(own, reverse_other)
        np.testing.assert_array_equal(other, reverse_own)
        self.assertFalse(np.array_equal(own, other))
        truth = paint((80, 80), [corners(boxes[0])])
        restored = np.asarray(crop.restore(crop.crop(truth, mask=True), truth.size)) > 0
        actual = np.asarray(truth) > 0
        self.assertGreater(np.count_nonzero(restored & actual) / np.count_nonzero(restored | actual), 0.80)
        self.assertEqual(match_boxes(boxes[::-1], boxes, truth.size, .3), {0: 1, 1: 0})
        np.testing.assert_allclose(polygon_box(corners(boxes[0]))[:4], boxes[0][:4], atol=1e-6)
        for mode in ("points", "boxes", "target"):
            _, field = crop.prompts(boxes, 0, mode)
            self.assertFalse(np.array_equal(other, field))


class WirePipelineTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.images, self.jsons = self.root / "images", self.root / "jsons"
        self.images.mkdir()
        self.jsons.mkdir()
        self.boxes = [[25, 16, 38, 3, 0], [25, 22, 36, 3, 0]]
        polys = [corners(b).tolist() for b in self.boxes]
        shapes = [{"label": "cable", "shape_type": "polygon", "points": p} for p in polys]
        size = (52, 36)
        image = paint(size, polys).convert("RGB")
        self.name = "04_12_x0_y0"
        image.save(self.images / (self.name + ".png"))
        self.annotation = {"imageWidth": size[0], "imageHeight": size[1], "shapes": shapes}
        save_json(self.jsons / (self.name + ".json"), self.annotation)
        self.cache = self.root / "proposals.json"
        entry = {"size": size, "coarse": self.boxes, "refined": self.boxes, "scores": [.95, .9]}
        save_json(self.cache, {"version": 1, "frames": {self.name + ".png": entry}})
        self.inputs = ["--images", str(self.images), "--jsons", str(self.jsons), "--cache", str(self.cache)]

    def test_cache_audit_and_zero_neighbor_control(self):
        def measure(margin):
            return audit(parser().parse_args(["audit", *self.inputs, "--margin", str(margin)]))
        self.assertEqual(measure(0)["targets_with_neighbor_pixels"], 0)
        wide = measure(12)
        self.assertEqual(wide["matched"], 2)
        self.assertEqual(wide["targets_with_neighbor_pixels"], 2)
        self.assertEqual(wide["neighbor_detected_fraction"], 1)

    def test_void_and_grouped_pieces(self):
        for shape in self.annotation["shapes"]:
            shape["group_id"] = 7
        self.annotation["shapes"].append({"label": "void", "points": [[0,0], [4,0], [4,4]]})
        path = self.jsons / "grouped.json"
        save_json(path, self.annotation)
        _, masks, void = annotations(path, (52, 36))
        self.assertEqual(len(masks), 1)
        self.assertTrue(void[1, 2])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Run mask-head training smoke test in the server environment")
    def test_training_evaluation_and_unlabelled_prediction(self):
        from myscripts.wire_seg.learning import mask_ap
        self.assertEqual(mask_ap([(0.9, 0, [(0, 1.0)]), (0.8, 0, [(0, 1.0)])], 2)["50"], 51/101)
        weights = self.root / "mask.pt"
        main(["train", *self.inputs, "--epochs", "1", "--batch-size", "2", "--device", "cpu", "--output", str(weights)])
        report = self.root / "eval.json"
        main(["eval", *self.inputs, "--weights", str(weights), "--device", "cpu", "--swap-target", "--output", str(report)])
        metric = json.loads(report.read_text())
        self.assertEqual(metric["matched"]["instances"], 2)
        self.assertEqual(metric["swap_target_pairs"], 2)
        output = self.root / "pred"
        main(["predict", "--images", str(self.images), "--cache", str(self.cache), "--weights", str(weights),
              "--device", "cpu", "--output", str(output)])
        self.assertEqual(len(list(output.rglob("*.png"))), 2)


if __name__ == "__main__":
    unittest.main()
