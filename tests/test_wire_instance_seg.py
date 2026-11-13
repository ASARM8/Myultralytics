"""Run directly with python -m unittest discover -s tests -p test_wire_instance_seg.py."""
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from myscripts.wire_seg.data import WireTiles, audit
from myscripts.wire_seg.batching import InstanceBatches
from myscripts.wire_seg.geometry import WireCrop, annotations, corners, match_boxes, paint, polygon_box
from myscripts.wire_seg.paired import evaluate_tile, summarize, validate_geometry
from myscripts.wire_seg.neighbor_pairs import PROMPTS, evaluate_pairs, summarize_pairs
from myscripts.wire_seg.proposals import save_json
from train_wire_seg import main, parser


class WireGeometryTest(unittest.TestCase):
    def test_neighbor_cohort_is_unordered_gt_only_and_keeps_misses(self):
        size = (80, 80)
        truth = [[35, y, 40, 2, 0] for y in (16, 21, 26, 30, 65)]
        masks = [np.asarray(paint(size, [corners(box)])) > 0 for box in truth]
        # Wire 2 is missed, 3 is entirely void, and 4 is distant.
        boxes = [truth[1], truth[0], truth[4]]
        image = Image.new("RGB", size)
        def empty_mask(prompt, feature):
            return np.zeros(feature.shape[1:], bool)
        rows = evaluate_pairs(image, boxes, masks, masks[3], {0: 1, 1: 0, 2: 4}, truth, 12, empty_mask)
        self.assertEqual([r["truth_ids"] for r in rows], [[0, 1], [0, 2], [1, 2]])
        self.assertEqual([r["proposal_ids"] for r in rows], [[1, 0], [1, None], [0, None]])
        self.assertTrue(all(r["near_parallel"] for r in rows))
        self.assertEqual(rows[1]["mask_ious"]["axes"], [0., None])
        result = summarize_pairs(rows)
        self.assertEqual(result["both_detected"], 1)
        self.assertEqual(result["detection_coverage"], 1/3)
        missed = evaluate_pairs(image, [], masks, masks[3], {}, truth, 12, empty_mask)
        self.assertEqual([r["truth_ids"] for r in missed], [r["truth_ids"] for r in rows])
        axes = summarize_pairs(missed)["by_mask_iou"]["0.50"]["axes"]
        self.assertEqual(axes["fraction_all"], 0)
        self.assertIsNone(axes["fraction_both_detected"])
        self.assertIsNone(summarize_pairs([])["by_mask_iou"]["0.50"]["axes"]["fraction_all"])

    def test_neighbor_thresholds_and_paired_wins(self):
        rows = [
            {"proposal_ids": [1, 2], "mask_ious": {p: [.75, .50] for p in PROMPTS}},
            {"proposal_ids": [1, 3], "mask_ious": {p: [.75, .75] for p in PROMPTS}},
            {"proposal_ids": [1, None], "mask_ious": {p: [.90, None] for p in PROMPTS}},
        ]
        rows[0]["mask_ious"]["points"] = [.75, .49]
        rows[1]["mask_ious"]["axes"] = [.75, .74]
        metrics = summarize_pairs(rows)["by_mask_iou"]
        self.assertEqual(metrics["0.50"]["axes"]["fraction_all"], 2/3)
        self.assertEqual(metrics["0.50"]["axes"]["fraction_both_detected"], 1)
        self.assertEqual(metrics["0.50"]["points"]["axes_only"], 1)
        self.assertEqual(metrics["0.75"]["points"]["comparator_only"], 1)
        self.assertEqual(metrics["0.75"]["points"]["successful_pairs"], 1)

    def test_paired_json_round_trip_with_numpy_iou_deltas(self):
        coarse = {"tile_obb_iou": 0.5, "mask_iou": 0.5, "hits": 1, "covered_target_pixels": 2,
                  "predicted_pixels": 1, "neighbor_pixels": 0, "swap_success": None}
        rows = []
        # All boxes stay identical while the masks change in three different directions.
        for hits in (2, 1, 0, 0):
            refined = {**coarse, "hits": hits, "predicted_pixels": hits, "mask_iou": np.float64(hits / 2)}
            rows.append({"target_pixels": 2, "coarse": coarse, "refined": refined,
                         "mask_iou_delta": refined["mask_iou"] - coarse["mask_iou"]})
        report = {"matched": summarize(rows), "empty": summarize([]), "paired_instances": rows}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paired.json"
            save_json(path, report)
            saved = json.loads(path.read_text(encoding="utf-8"))
        expected = {"improved": 1, "unchanged": 1, "worsened": 2}
        counts = saved["matched"]["iou_change"]
        self.assertEqual(counts, expected)
        self.assertTrue(all(type(count) is int for count in counts.values()))
        self.assertEqual(saved["matched"]["box_iou_change_by_mask_iou_change"]["unchanged"], expected)
        self.assertEqual(saved["empty"]["iou_change"], dict.fromkeys(expected, 0))
        self.assertIsNone(saved["empty"]["coarse"]["mean_instance_iou"])

    def test_paired_geometry_keeps_neighbor_when_refined_crop_loses_it(self):
        size = (60, 42)
        truth = [[30, 18, 36, 2, 0], [30, 24, 36, 2, 0]]
        masks = [np.asarray(paint(size, [corners(box)])) > 0 for box in truth]
        coarse = [[30, y, 36, 16, 0] for y in (18, 24)]
        refined = [[30, y, 16, 3, 0] for y in (18, 24)]
        entry = {"coarse": coarse, "refined": refined, "scores": [.9, .8]}
        validate_geometry(entry, "pair")
        rows, detections, count = evaluate_tile(
            Image.new("RGB", size), entry, masks, np.zeros(size[::-1], bool), {0: 0, 1: 1},
            truth, 0, lambda feature: feature[3] > .9
        )
        self.assertEqual(count, 2)
        self.assertEqual([r["swap_neighbor_proposal_id"] for r in rows], [1, 0])
        self.assertTrue(all(r["exposed"] and r["near_parallel"] for r in rows))
        self.assertEqual([s for s, _ in detections["refined"]], [.9, .8])
        result = summarize(rows)
        self.assertEqual(result["swap_pairs"], 2)
        self.assertEqual(result["swap_transitions"]["coarse_only"], 2)
        self.assertLess(result["refined_minus_coarse"]["target_crop_coverage"], 0)
        self.assertGreater(result["refined_minus_coarse"]["mean_tile_obb_iou"], 0)
        self.assertIsNone(summarize([])["refined_minus_coarse"]["mean_instance_iou"])
        with self.assertRaisesRegex(ValueError, "proposal 0"):
            validate_geometry({**entry, "refined": refined[::-1]}, "unpaired")

    def test_paired_ap_includes_unmatched_detections_and_missed_truth(self):
        size = (20, 20)
        image = Image.new("RGB", size)
        box = [10, 10, 8, 2, 0]
        mask = np.asarray(paint(size, [corners(box)])) > 0
        entry = {"coarse": [box], "refined": [box], "scores": [.8]}
        rows, detections, count = evaluate_tile(image, entry, [mask], np.zeros_like(mask), {},
                                                [box], 1, lambda feature: feature[3] < 0)
        self.assertEqual(rows, [])
        self.assertEqual(count, 1)
        for arm in ("coarse", "refined"):
            self.assertEqual(detections[arm], [(.8, [])])

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

    def test_neighbor_matching_retains_original_truth_ids_after_void_filter(self):
        self.annotation["shapes"].append({"label": "void", "points": [[0, 0], [51, 0], [51, 19], [0, 19]]})
        save_json(self.jsons / (self.name + ".json"), self.annotation)
        tiles = WireTiles(parser().parse_args(["audit", *self.inputs]))
        self.assertEqual(tiles.read(0, match_visible_only=True)[5], {1: 1})

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Run the four-head inference test on the server")
    def test_neighbor_pairs_cli_uses_all_four_heads_without_training(self):
        import torch
        from myscripts.wire_seg.learning import WireMaskHead

        model = WireMaskHead()
        for prompt in PROMPTS:
            folder = self.root / prompt
            folder.mkdir()
            torch.save({"format": 1, "prompt": prompt, "boxes": "refined", "margin": 12,
                        "detector": {"version": 1}, "state": model.state_dict(), "epoch": 15,
                        "seed": 0, "train_sources": [], "train_videos": []}, folder / "mask.pt")
        output = self.root / "pairs.json"
        main(["neighbor-pairs", *self.inputs, "--weights-dir", str(self.root), "--device", "cpu",
              "--output", str(output)])
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["summary"]["all"]["both_detected"], 1)
        self.assertEqual(report["summary"]["near_parallel"]["gt_pairs"], 1)
        self.assertEqual(len(report["pairs"]), 1)
        self.assertEqual(set(report["pairs"][0]["mask_ious"]), set(PROMPTS))
        self.assertEqual(report["pairs"][0]["proposal_ids"], [0, 1])

        # A multi-cutoff pass must reproduce both existing evaluators with only one forward per proposal.
        from unittest.mock import patch
        from myscripts.wire_seg.learning import crop_probability
        weights = str(self.root / "axes" / "mask.pt")
        scan = self.root / "scan.json"
        with patch("myscripts.wire_seg.threshold_scan.crop_probability", wraps=crop_probability) as forward:
            main(["threshold-scan", *self.inputs, "--weights", weights, "--thresholds", "0.25", "0.5",
                  "--device", "cpu", "--output", str(scan)])
        self.assertEqual(forward.call_count, len(self.boxes))
        results = json.loads(scan.read_text())["results"]
        self.assertEqual([row["mask_threshold"] for row in results], [.25, .5])
        for row in results:
            single = self.root / "single.json"
            main(["eval", *self.inputs, "--weights", weights, "--threshold", str(row["mask_threshold"]),
                  "--device", "cpu", "--output", str(single)])
            reference = json.loads(single.read_text())
            for key in ("mask_AP_by_IoU", "mask_mAP50_95", "matched", "matched_with_neighbors",
                        "matched_with_neighbor_within_15deg"):
                self.assertEqual(row[key], reference[key])
        self.assertEqual(results[1]["pairs"][0]["mask_ious"]["axes"], report["pairs"][0]["mask_ious"]["axes"])
        for cohort in ("all", "near_parallel"):
            for cutoff in ("0.50", "0.75"):
                self.assertEqual(results[1]["neighbor_pairs"][cohort]["by_mask_iou"][cutoff]["axes"],
                                 report["summary"][cohort]["by_mask_iou"][cutoff]["axes"])

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
        paired_path = self.root / "paired.json"
        main(["paired", *self.inputs, "--weights", str(weights), "--device", "cpu", "--output", str(paired_path)])
        paired = json.loads(paired_path.read_text())
        self.assertEqual(len(paired["paired_instances"]), 2)
        self.assertEqual(paired["matched"]["swap_pairs"], 2)
        self.assertEqual(paired["mask_AP_by_IoU"]["refined"], metric["mask_AP_by_IoU"])
        self.assertEqual(paired["mask_mAP50_95_delta"], 0)
        self.assertEqual(paired["matched"]["iou_change"]["unchanged"], 2)
        for delta in paired["matched"]["refined_minus_coarse"].values():
            self.assertIn(delta, (0, None))
        output = self.root / "pred"
        main(["predict", "--images", str(self.images), "--cache", str(self.cache), "--weights", str(weights),
              "--device", "cpu", "--output", str(output)])
        self.assertEqual(len(list(output.rglob("*.png"))), 2)


if __name__ == "__main__":
    unittest.main()
