"""Torch-free protocol tests for the V3.1.1 paper evidence tools."""

from myscripts.V3_1_1.evidence_runtime import CANONICAL_BASELINE_WEIGHTS, CANONICAL_CA_WEIGHTS
from myscripts.V3_1_1.export_qualitative_v311 import build_parser as build_export_parser
from myscripts.V3_1_1.export_qualitative_v311 import stable_export_id
from myscripts.V3_1_1.export_qualitative_v311 import validate_args as validate_export_args
from myscripts.V3_1_1.profile_refine_v311 import build_parser as build_profile_parser
from myscripts.V3_1_1.profile_refine_v311 import summarize_timings
from myscripts.V3_1_1.profile_refine_v311 import validate_args as validate_profile_args


def test_profile_protocol_is_fp32_batch1_imgsz640():
    args = build_profile_parser().parse_args(
        ["--checkpoint", "/tmp/refine.pt", "--data", "/tmp/data.yaml", "--output-dir", "/tmp/out"]
    )
    assert args.ca_weights == CANONICAL_CA_WEIGHTS
    assert args.imgsz == 640
    assert args.batch == 1
    assert args.amp is False
    assert args.split == "val"


def test_export_protocol_uses_fixed_baseline_and_ca_paths():
    args = build_export_parser().parse_args(
        ["--checkpoint", "/tmp/refine.pt", "--data", "/tmp/data.yaml", "--output-dir", "/tmp/out"]
    )
    assert args.baseline_weights == CANONICAL_BASELINE_WEIGHTS
    assert args.ca_weights == CANONICAL_CA_WEIGHTS
    assert args.imgsz == 640
    assert args.amp is False
    assert args.copy_images is True


def test_export_identifier_is_stable_and_avoids_same_stem_collision():
    first = stable_export_id("/data/a/frame.jpg")
    second = stable_export_id("/data/b/frame.jpg")
    assert first == stable_export_id("/data/a/frame.jpg")
    assert first != second
    assert first.startswith("frame__")


def test_timing_summary_reports_mean_median_and_p95():
    rows = [{"total_compute_ms": value} for value in (1.0, 2.0, 3.0, 4.0)]
    summary = summarize_timings(rows, "total_compute_ms")
    assert summary["mean"] == 2.5
    assert summary["median"] == 2.5
    assert abs(summary["p95"] - 3.85) < 1e-12


def test_profile_rejects_amp_for_the_official_fp32_protocol():
    parser = build_profile_parser()
    args = parser.parse_args(
        ["--checkpoint", "/tmp/refine.pt", "--data", "/tmp/data.yaml", "--amp", "--output-dir", "/tmp/out"]
    )
    try:
        validate_profile_args(parser, args)
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("official profiler accepted AMP")


def test_export_rejects_noncanonical_baseline_path():
    parser = build_export_parser()
    args = parser.parse_args(
        [
            "--baseline-weights",
            "/tmp/wrong.pt",
            "--checkpoint",
            "/tmp/refine.pt",
            "--data",
            "/tmp/data.yaml",
            "--output-dir",
            "/tmp/out",
        ]
    )
    try:
        validate_export_args(parser, args)
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("qualitative exporter accepted a noncanonical Baseline path")
