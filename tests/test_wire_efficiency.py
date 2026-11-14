"""Server smoke tests for online profiling and separation of timing populations."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from myscripts.wire_seg.profile_pipeline import STAGES, build_parser, describe, summarize


class EfficiencySummaryTest(unittest.TestCase):
    def test_synchronized_pass_does_not_inflate_end_to_end_fps(self):
        rows = []
        for repeat in range(1, 4):
            for name, proposals, ms in (("empty.png", 0, 10), ("wires.png", 2, 30)):
                rows.append({"phase": "end_to_end", "repeat": repeat, "tile": name,
                             "proposals": proposals, "crop_pixels": proposals * 100,
                             "total_ms": ms, **dict.fromkeys(STAGES, 0.)})
        rows.append({**rows[0], "phase": "stages", "total_ms": 1000, **dict.fromkeys(STAGES, 100.)})
        result = summarize(rows, [], .05)
        self.assertEqual(result["serial_tiles_per_second"], 50)
        self.assertEqual(result["end_to_end_ms"]["count"], 6)
        self.assertEqual(result["stage_instrumented_total_ms"]["mean"], 1000)
        self.assertTrue(result["stability_pass"])
        self.assertEqual(result["by_proposal_count"][0]["unique_tiles"], 1)
        self.assertIsNone(result["by_proposal_count"][-1]["latency_ms"]["mean"])
        self.assertFalse(summarize(rows[:2], [], .05)["stability_pass"])
        # The report must remain strict JSON even for empty workload bins.
        json.dumps(result, allow_nan=False)

    def test_variable_repeats_fail_stability_and_quantiles_interpolate(self):
        rows = [{"phase": "end_to_end", "repeat": i, "tile": "a", "proposals": 1,
                 "crop_pixels": 10, "total_ms": ms} for i, ms in enumerate((10, 20, 30), 1)]
        self.assertFalse(summarize(rows, [], .05)["stability_pass"])
        self.assertEqual(describe([10, 30])["p95"], 29)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("cv2"),
                     "Run the online pipeline test in the existing server environment")
class EfficiencyPipelineTest(unittest.TestCase):
    def setUp(self):
        import torch
        from PIL import Image
        from myscripts.wire_seg.learning import WireMaskHead

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.path = self.images / "wire.png"
        Image.new("RGB", (80, 40), "white").save(self.path)
        self.boxes = torch.tensor([[320., 320., 320., 16., 0.],
                                   [320., 360., 240., 16., 0.]])
        self.scores = torch.tensor([.9, .01])
        self.calls = 0
        self.refine_calls = 0

        def infer(batch):
            self.calls += 1
            features = torch.zeros(1, 1, 1, 1)
            return batch["img"], features, features, [{"bboxes": self.boxes,
                "conf": self.scores, "cls": torch.zeros(len(self.boxes))}]

        def refine(*_args):
            self.refine_calls += 1
            return {"residual": torch.zeros(1, len(self.boxes), 4)}

        class Refiner:
            def __call__(self, *args):
                return refine(*args)

            def apply_residual(self, boxes, _residual):
                return boxes.clone()

        self.bundle = SimpleNamespace(torch=torch, device=torch.device("cpu"),
            extractor=SimpleNamespace(infer=infer), refiner=Refiner(),
            ca_model=SimpleNamespace(names={0: "wire"}), ca_hash="ca", checkpoint_hash="refine", close=lambda: None)
        self.model = WireMaskHead().eval()
        from myscripts.V3_1_1_low_gain.config import DEFAULT_RESIDUAL_SCALE
        self.detector = {"version": 1, "ca_sha256": "ca", "refine_sha256": "refine",
                         "scale": DEFAULT_RESIDUAL_SCALE, "wire_class": 0}
        self.checkpoint = {"format": 1, "state": self.model.state_dict(), "boxes": "refined", "prompt": "axes",
                           "margin": 12, "detector": self.detector}

    def test_online_cache_geometry_and_mask_outputs_agree(self):
        import numpy as np
        import torch
        from PIL import Image
        from myscripts.wire_seg.data import example
        from myscripts.wire_seg.learning import crop_probability
        from myscripts.wire_seg.profile_pipeline import compare_frame, run_one
        from myscripts.wire_seg.proposals import cache_proposals

        cache = self.root / "cache.json"
        args = SimpleNamespace(images=self.images, refine="refine", ca="ca", device="cpu", wire_class=0, output=cache)
        with patch("myscripts.V3_1_1.evidence_runtime.load_refine_bundle", return_value=self.bundle):
            cache_proposals(args)
        reference = json.loads(cache.read_text())["frames"][self.path.name]
        with torch.inference_mode():
            row, frame, masks = run_one(self.path, self.bundle, self.checkpoint, self.model, 0, .5)
            staged, second, stage_masks = run_one(self.path, self.bundle, self.checkpoint, self.model, 0, .5,
                                                  stage_timing=True)
            self.assertTrue(compare_frame(frame, reference)["matches"])
            reordered = {**reference, "scores": reference["scores"][::-1]}
            self.assertFalse(compare_frame(frame, reordered)["matches"])
            missing = {**reference, "refined": reference["refined"][:1]}
            self.assertFalse(compare_frame(frame, missing)["matches"])
            np.testing.assert_allclose(frame["coarse"], [[40, 20, 40, 2, 0], [40, 25, 30, 2, 0]])
            self.assertEqual(row["proposals"], 2)  # The low-confidence wire is not dropped.
            self.assertEqual(row["crop_pixels"], (48 + 38) * 26)
            self.assertEqual(self.calls, 3)  # Every timing pass really invokes the detector.
            self.assertEqual(self.refine_calls, 3)
            self.assertEqual(frame, second)
            with Image.open(self.path) as source:
                image = source.convert("RGB")
            for i, actual in enumerate(masks):
                crop, feature = example(image, frame["refined"], i, "axes", 12)
                expected = crop.restore(Image.fromarray(crop_probability(self.model, feature, "cpu") >= .5), image.size)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(actual, stage_masks[i])
            self.assertTrue(all(staged[key] > 0 for key in STAGES))
            self.boxes = torch.empty(0, 5)
            self.scores = torch.empty(0)
            empty, _, masks = run_one(self.path, self.bundle, self.checkpoint, self.model, 0, .5, stage_timing=True)
            self.assertEqual((empty["proposals"], empty["crop_pixels"], masks), (0, 0, []))
            self.assertEqual(empty["mask_inference_ms"], 0)
            self.assertEqual(self.refine_calls, 3)

    def test_profile_exports_readable_reports_and_archive(self):
        import tarfile
        import torch
        from myscripts.wire_seg.profile_pipeline import run_profile

        weights = self.root / "mask.pt"
        torch.save(self.checkpoint, weights)
        output = self.root / "profile"
        args = build_parser().parse_args(["--images", str(self.images), "--ca", "ca", "--refine", "refine",
            "--weights", str(weights), "--output-dir", str(output), "--device", "cpu", "--warmup", "1",
            "--repeats", "3", "--stage-repeats", "1"])
        with patch("myscripts.V3_1_1.evidence_runtime.load_refine_bundle", return_value=self.bundle):
            run_profile(args)
        result = json.loads((output / "summary.json").read_text())
        self.assertEqual(result["end_to_end_ms"]["count"], 3)
        self.assertEqual(result["stage_instrumented_total_ms"]["count"], 1)
        self.assertEqual(result["consistency"]["repeat_output_mismatches"], 0)
        self.assertEqual(self.calls, 8)  # Four warmups and four measured live calls.
        with tarfile.open(str(output) + ".tar.gz") as archive:
            self.assertIn("profile/per_image.csv", archive.getnames())
            self.assertIn("profile/manifest.json", archive.getnames())


if __name__ == "__main__":
    unittest.main()
