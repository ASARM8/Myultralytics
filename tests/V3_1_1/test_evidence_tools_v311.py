"""Torch-free protocol tests for the V3.1.1 paper evidence tools."""

from myscripts.V3_1_1.collect_ivc_evidence import CANONICAL_DATA, CANONICAL_REFINE
from myscripts.V3_1_1.collect_ivc_evidence import EVIDENCE_PROTOCOL_VERSION
from myscripts.V3_1_1.collect_ivc_evidence import EvidenceCollector
from myscripts.V3_1_1.collect_ivc_evidence import build_command_stages
from myscripts.V3_1_1.collect_ivc_evidence import build_parser as build_collection_parser
from myscripts.V3_1_1.collect_ivc_evidence import build_run_identity
from myscripts.V3_1_1.collect_ivc_evidence import validate_args as validate_collection_args
from myscripts.V3_1_1.evidence_runtime import CANONICAL_BASELINE_WEIGHTS, CANONICAL_CA_WEIGHTS
from myscripts.V3_1_1.export_qualitative_v311 import build_parser as build_export_parser
from myscripts.V3_1_1.export_qualitative_v311 import stable_export_id
from myscripts.V3_1_1.export_qualitative_v311 import validate_args as validate_export_args
from myscripts.V3_1_1.profile_refine_v311 import build_parser as build_profile_parser
from myscripts.V3_1_1.profile_refine_v311 import summarize_timings
from myscripts.V3_1_1.profile_refine_v311 import validate_args as validate_profile_args
from myscripts.V3_1_1.profile_comparative_v311 import build_parser as build_comparative_parser
from myscripts.V3_1_1.profile_comparative_v311 import _aggregate_method
from myscripts.V3_1_1.profile_comparative_v311 import balanced_method_orders
from myscripts.V3_1_1.profile_comparative_v311 import validate_args as validate_comparative_args
from myscripts.V3_1_1.profile_detector_worker_v311 import build_parser as build_detector_worker_parser
from myscripts.V3_1_1.profile_detector_worker_v311 import validate_args as validate_detector_worker_args
from myscripts.paper_visuals.collect_validation_metrics import iou_ap_columns


def test_profile_protocol_is_fp32_batch1_imgsz640():
    args = build_profile_parser().parse_args(
        ["--checkpoint", "/tmp/refine.pt", "--data", "/tmp/data.yaml", "--output-dir", "/tmp/out"]
    )
    assert args.ca_weights == CANONICAL_CA_WEIGHTS
    assert args.imgsz == 640
    assert args.batch == 1
    assert args.amp is False
    assert args.split == "val"
    assert args.warmup == 500


def test_comparative_profile_uses_balanced_three_position_order():
    assert balanced_method_orders() == (
        ("Baseline", "CA", "CA+Refine"),
        ("CA", "CA+Refine", "Baseline"),
        ("CA+Refine", "Baseline", "CA"),
    )


def test_comparative_aggregate_reports_repeat_and_memory_stability():
    records = []
    for repeat, mean in enumerate((10.0, 10.1, 9.9), start=1):
        records.append(
            {
                "method": "CA",
                "repeat": repeat,
                "order_position": repeat,
                "summary": {
                    "weights": CANONICAL_CA_WEIGHTS,
                    "weights_sha256": "a" * 64,
                    "weights_unchanged": True,
                    "parameters": 100,
                    "gflops": 20.0,
                    "gflops_scope": "test",
                    "measured_images": 2,
                    "input_shape_counts": {"672x672": 2},
                    "timing_ms_per_image": {"total_compute_ms": {"mean": mean}},
                    "gpu_memory": {"peak_allocated_gib": 1.0, "incremental_peak_gib": 0.25},
                },
                "rows": [
                    {"total_compute_ms": mean - 0.1, "proposal_count": 2},
                    {"total_compute_ms": mean + 0.1, "proposal_count": 4},
                ],
            }
        )
    summary = _aggregate_method("CA", records, latency_tolerance=0.05, memory_tolerance=0.02)
    assert summary["repeat_count"] == 3
    assert summary["latency_stability_pass"] is True
    assert summary["isolated_peak_gpu_memory_gib"]["stability_pass"] is True
    assert summary["proposal_count"] == {"mean": 3.0, "median": 3.0, "p95": 3.9, "max": 4}
    assert summary["flop_profile_proposals"] is None
    assert abs(summary["latency_mean_of_repeat_means_ms"] - 10.0) < 1e-12


def test_refine_aggregate_declares_proposal_dependent_flops():
    records = []
    for repeat in range(1, 4):
        records.append(
            {
                "method": "CA+Refine",
                "repeat": repeat,
                "order_position": repeat,
                "summary": {
                    "ca_weights": CANONICAL_CA_WEIGHTS,
                    "ca_sha256": "a" * 64,
                    "refine_checkpoint": "/tmp/refine.pt",
                    "refine_sha256": "b" * 64,
                    "weights_unchanged": True,
                    "total_parameters": 120,
                    "detector_gflops": 20.0,
                    "refiner_profiled_gflops": 0.5,
                    "refiner_flop_profile_proposals": 2,
                    "measured_images": 2,
                    "input_shape_counts": {"672x672": 2},
                    "timing_ms_per_image": {"total_compute_ms": {"mean": 12.0}},
                    "gpu_memory": {"peak_allocated_gib": 1.2, "incremental_peak_gib": 0.3},
                },
                "rows": [
                    {"total_compute_ms": 11.9, "proposal_count": 2},
                    {"total_compute_ms": 12.1, "proposal_count": 4},
                ],
            }
        )
    summary = _aggregate_method("CA+Refine", records, latency_tolerance=0.05, memory_tolerance=0.02)
    assert summary["gflops"] == 20.5
    assert summary["flop_profile_proposals"] == 2
    assert "at 2 proposals" in summary["gflops_scope"]
    assert summary["proposal_count"]["p95"] == 3.9


def test_detector_worker_locks_checkpoint_to_method():
    parser = build_detector_worker_parser()
    args = parser.parse_args(
        [
            "--method",
            "CA",
            "--weights",
            CANONICAL_CA_WEIGHTS,
            "--data",
            "/tmp/data.yaml",
            "--conf",
            "0.01",
            "--nms-iou",
            "0.7",
            "--max-det",
            "300",
            "--output-dir",
            "/tmp/out",
        ]
    )
    validate_detector_worker_args(parser, args)
    assert args.batch == 1
    assert args.imgsz == 640
    assert args.split == "val"
    assert args.warmup == 500


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


def test_comparative_profile_locks_common_fp32_protocol(tmp_path):
    refine_summary = tmp_path / "profile_summary.json"
    refine_summary.write_text("{}\n", encoding="utf-8")
    parser = build_comparative_parser()
    args = parser.parse_args(
        [
            "--refine-profile-summary",
            str(refine_summary),
            "--data",
            "/tmp/data.yaml",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    validate_comparative_args(parser, args)
    assert args.baseline_weights == CANONICAL_BASELINE_WEIGHTS
    assert args.ca_weights == CANONICAL_CA_WEIGHTS
    assert args.split == "val"
    assert args.imgsz == 640
    assert args.batch == 1
    assert args.amp is False
    assert args.repeats == 3
    assert args.warmup == 500


def test_complete_iou_threshold_columns_are_exported():
    import numpy as np

    all_ap = np.arange(20, dtype=float).reshape(2, 10) / 20.0
    columns = iou_ap_columns(all_ap)
    assert list(columns) == [f"ap{threshold}" for threshold in range(50, 100, 5)]
    assert columns["ap50"] == float(all_ap[:, 0].mean())
    assert columns["ap95"] == float(all_ap[:, 9].mean())


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


def test_collection_defaults_lock_current_ivc_protocol():
    parser = build_collection_parser()
    args = parser.parse_args([])
    validate_collection_args(parser, args)
    assert args.data == CANONICAL_DATA
    assert args.baseline_weights == CANONICAL_BASELINE_WEIGHTS
    assert args.ca_weights == CANONICAL_CA_WEIGHTS
    assert args.checkpoint == CANONICAL_REFINE
    assert args.tests is True
    assert args.smoke is True
    assert args.h1h2 is True
    assert args.archive is True
    assert build_run_identity(args)["evidence_protocol_version"] == EVIDENCE_PROTOCOL_VERSION == 3


def test_collection_plan_calls_evidence_tools_but_not_training(tmp_path):
    args = build_collection_parser().parse_args([])
    stages = build_command_stages(args, tmp_path)
    names = {stage.name for stage in stages}
    assert {
        "main_validation",
        "reproduction_fp32_batch8",
        "reproduction_amp_batch8",
        "reproduction_fp32_batch1",
        "reproduction_audit",
        "complexity_baseline_ca",
        "profile_refine_fp32_batch1",
        "profile_comparative_fp32_batch1",
        "qualitative_predictions",
        "h1h2_baseline",
        "h1h2_ca",
    }.issubset(names)
    commands = "\n".join(" ".join(stage.command) for stage in stages)
    assert "train_refine" not in commands
    assert "--imgsz 640" in commands
    assert f"Baseline={CANONICAL_BASELINE_WEIGHTS}" in commands
    assert f"CA={CANONICAL_CA_WEIGHTS}" in commands
    assert "myscripts.V3_1_1.profile_comparative_v311" in commands
    assert "--repeats 3" in commands
    assert "--split val --passes 1" in commands
    assert "--epochs" not in commands
    formal_profile = next(stage for stage in stages if stage.name == "profile_refine_fp32_batch1")
    comparative_profile = next(stage for stage in stages if stage.name == "profile_comparative_fp32_batch1")
    assert "--warmup 500" in " ".join(formal_profile.command)
    assert "--warmup 500" in " ".join(comparative_profile.command)


def test_collection_can_skip_smoke_and_h1h2(tmp_path):
    args = build_collection_parser().parse_args(["--no-smoke", "--no-h1h2"])
    names = {stage.name for stage in build_command_stages(args, tmp_path)}
    assert not any(name.startswith("smoke_") for name in names)
    assert not any(name.startswith("h1h2_") for name in names)


def test_resume_skips_only_completed_stage_with_existing_output(tmp_path):
    expected = tmp_path / "result.json"
    expected.write_text("{}\n", encoding="utf-8")
    collector = EvidenceCollector(tmp_path, resume=True)
    collector.state["stages"]["example"] = {"status": "completed"}
    assert collector._can_skip("example", (expected,)) is True
    expected.unlink()
    assert collector._can_skip("example", (expected,)) is False


def test_resume_reprofiles_legacy_efficiency_protocol(tmp_path):
    collector = EvidenceCollector(tmp_path, resume=True)
    collector.state["stages"]["profile_refine_fp32_batch1"] = {"status": "completed"}
    summary = tmp_path / "profile_refine_fp32_batch1" / "profile_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text('{"protocol_version": 1, "warmup_passes": 500}\n', encoding="utf-8")
    assert collector._can_skip("profile_refine_fp32_batch1", (summary,)) is False
    summary.write_text('{"protocol_version": 2, "warmup_passes": 500}\n', encoding="utf-8")
    assert collector._can_skip("profile_refine_fp32_batch1", (summary,)) is False
    summary.write_text('{"protocol_version": 3, "warmup_passes": 20}\n', encoding="utf-8")
    assert collector._can_skip("profile_refine_fp32_batch1", (summary,)) is False
    summary.write_text('{"protocol_version": 3, "warmup_passes": 500}\n', encoding="utf-8")
    assert collector._can_skip("profile_refine_fp32_batch1", (summary,)) is True


def test_resume_rejects_run_identity_drift(tmp_path):
    collector = EvidenceCollector(tmp_path, resume=True)
    collector.lock_run_identity({"checkpoint": "/weights/a.pt", "imgsz": 640})
    try:
        collector.lock_run_identity({"checkpoint": "/weights/b.pt", "imgsz": 640})
    except RuntimeError as error:
        assert "拒绝混合证据" in str(error)
    else:
        raise AssertionError("resume accepted a changed checkpoint")
