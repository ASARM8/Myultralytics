"""Small regression checks for the downloadable wire diagnostics bundle."""
import csv
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from myscripts.wire_seg.data import example
from myscripts.wire_seg.export_failures import PIXEL_FIELDS, axis_coordinates, export_case, main, select_failures, width_estimate
from myscripts.wire_seg.geometry import WireCrop, corners, paint
from myscripts.wire_seg.paired import footprint
from myscripts.wire_seg.proposals import save_json


HAS_SCIPY = importlib.util.find_spec("scipy") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


class FailureSelectionTest(unittest.TestCase):
    def test_select_weak_side_once_and_keep_all_pair_contexts(self):
        def pair(ids, proposals, values):
            return {"tile": "a.jpg", "truth_ids": ids, "proposal_ids": proposals,
                    "mask_ious": {"axes": values}}
        rows = [pair([0, 1], [4, 5], [.2, .8]), pair([1, 0], [5, 4], [.8, .2]),
                pair([0, 2], [4, 6], [.2, .9]), pair([2, 3], [6, None], [.9, None]),
                pair([0, 3], [4, 7], [.2, .1]), pair([1, 2], [5, 6], [.8, .9])]
        selected = select_failures({"pairs": rows}, .5)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0][0], ("a.jpg", 4, 0))
        self.assertEqual(len(selected[0][1]), 3)

    def test_finite_axis_coordinates_follow_the_long_side(self):
        along, dist, fraction = axis_coordinates([5, 5, 2, 10, 0], np.array([5, 5, 7]), np.array([0, 12, 5]))
        np.testing.assert_allclose(along, [-5, 7, 0], atol=1e-6)
        np.testing.assert_allclose(dist, [0, 2, 2], atol=1e-6)
        np.testing.assert_allclose(fraction, [0, 1.2, .5], atol=1e-6)

    @unittest.skipUnless(HAS_SCIPY, "Run raster-width diagnostics in the existing server environment")
    def test_width_estimate_and_tiny_disconnected_components(self):
        mask = np.zeros((25, 35), bool)
        mask[4:9, 3:27] = True
        mask[18:20, 4:6] = True
        skeleton, widths = width_estimate(mask)
        self.assertTrue(skeleton[18:20, 4:6].any())
        self.assertEqual(widths[6, 14], 5)
        self.assertTrue((widths[mask] >= 1).all())
        self.assertTrue((widths[~mask] == 0).all())

    @unittest.skipUnless(HAS_SCIPY, "Run PNG, NPZ and pixel-table export on the server")
    def test_case_export_has_consistent_pixel_table_and_arrays(self):
        size = (52, 36)
        boxes = [[25, 16, 38, 3, 0], [25, 22, 36, 3, 0]]
        masks = [np.asarray(paint(size, [corners(b)])) > 0 for b in boxes]
        image = Image.new("RGB", size, (40, 60, 80))
        void = np.zeros(size[::-1], bool)
        crop, features = example(image, boxes, 0, "axes", 12)
        probability = np.asarray(crop.crop(Image.fromarray(masks[0]), mask=True), dtype=np.float32).copy()
        probability[:, :probability.shape[1] // 2] = 0
        pixels = io.StringIO()
        writer = csv.DictWriter(pixels, fieldnames=PIXEL_FIELDS)
        writer.writeheader()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            stats, _ = export_case(image, boxes, masks, void, 0, 0, probability, crop, features,
                                  folder, writer, "sample", .5)
            records = list(csv.DictReader(io.StringIO(pixels.getvalue())))
            self.assertEqual(len(records), stats["target_pixels"])
            self.assertEqual(sum(r["missed"] == "True" for r in records), stats["missed_pixels"])
            with np.load(folder / "arrays.npz", allow_pickle=False) as arrays:
                self.assertEqual(arrays["gt"].shape, size[::-1])
                self.assertEqual(int(arrays["pred"].sum()), stats["predicted_pixels"])
                self.assertEqual(len(arrays["x"]), stats["target_pixels"])
            self.assertTrue((folder / "panel.png").is_file())
            self.assertTrue((folder / "crop_panel_4x.png").is_file())

    @unittest.skipUnless(HAS_SCIPY and HAS_TORCH, "Run the complete diagnostic archive check on the server")
    def test_one_command_builds_an_archive_and_reproduces_report(self):
        import torch
        from myscripts.wire_seg.learning import WireMaskHead

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, jsons = root / "images", root / "jsons"
            images.mkdir()
            jsons.mkdir()
            size = (60, 70)
            boxes = [[30, 10, 30, 2, 0], [30, 38, 50, 40, 0]]
            masks = [np.asarray(paint(size, [corners(b)])) > 0 for b in boxes]
            image = Image.new("RGB", size)
            image.save(images / "case.png")
            save_json(jsons / "case.json", {"imageWidth": size[0], "imageHeight": size[1],
                      "shapes": [{"label": "cable", "points": corners(b).tolist()} for b in boxes]})
            cache = root / "cache.json"
            detector = {"version": 1}
            save_json(cache, {**detector, "frames": {"case.png": {"size": size, "coarse": boxes,
                      "refined": boxes, "scores": [.9, .8]}}})
            model = WireMaskHead()
            for param in model.parameters():
                param.data.zero_()
            weights = root / "mask.pt"
            torch.save({"format": 1, "prompt": "axes", "boxes": "refined", "margin": 12,
                        "detector": detector, "state": model.state_dict()}, weights)
            predictions = [footprint(WireCrop(b, 12), size) for b in boxes]
            overlaps = [float((p & m).sum() / (p | m).sum()) for p, m in zip(predictions, masks)]
            self.assertLess(overlaps[0], .5)
            self.assertGreater(overlaps[1], .5)
            pairs_path, paired_path = root / "pairs.json", root / "paired.json"
            save_json(pairs_path, {"detector": detector, "mask_crop_margin_px": 12, "mask_threshold": .5,
                      "coarse_obb_match_iou": .3, "pairs": [{"tile": "case.png", "proposal_ids": [0, 1],
                      "truth_ids": [0, 1], "mask_ious": {"axes": overlaps}}]})
            pred, truth = predictions[0], masks[0]
            save_json(paired_path, {"detector": detector, "margin_px": 12, "threshold": .5, "match_iou": .3,
                      "paired_instances": [{"tile": "case.png", "proposal_id": 0, "truth_id": 0,
                      "target_pixels": int(truth.sum()), "refined": {"box_xywhr": boxes[0],
                      "mask_iou": overlaps[0], "hits": int((pred & truth).sum()), "predicted_pixels": int(pred.sum())}}]})
            output = root / "bundle"
            main(["--images", str(images), "--jsons", str(jsons), "--cache", str(cache), "--weights", str(weights),
                  "--pairs-report", str(pairs_path), "--paired-report", str(paired_path),
                  "--output-dir", str(output), "--device", "cpu"])
            summary = json.loads((output / "summary.json").read_bytes())
            self.assertEqual(summary["selected_wires"], 1)
            self.assertEqual(summary["reproduction_mismatches"], [])
            with tarfile.open(root / "bundle.tar.gz") as archive:
                self.assertIn("bundle/index.html", archive.getnames())
                self.assertIn("bundle/target_pixels.csv.gz", archive.getnames())
                self.assertIn("bundle/inputs/selected_cache.json", archive.getnames())


if __name__ == "__main__":
    unittest.main()
