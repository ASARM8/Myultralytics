"""Torch-free tests for the isolated V3.1.1 low-gain tools."""

import argparse
from pathlib import Path

from myscripts.V3_1_1.evidence_runtime import CANONICAL_CA_WEIGHTS
from myscripts.V3_1_1_low_gain import DEFAULT_RESIDUAL_SCALE, TARGET_MAP_GAIN
from myscripts.V3_1_1_low_gain.audit_reproductions_low_gain_v311 import build_parser as build_audit_parser
from myscripts.V3_1_1_low_gain.collect_low_gain_evidence import build_parser as build_collector_parser
from myscripts.V3_1_1_low_gain.collect_low_gain_evidence import build_stages
from myscripts.V3_1_1_low_gain.export_qualitative_low_gain_v311 import build_parser as build_export_parser
from myscripts.V3_1_1_low_gain.profile_comparative_low_gain_v311 import (
    build_parser as build_comparative_parser,
)
from myscripts.V3_1_1_low_gain.profile_refine_low_gain_v311 import build_parser as build_profile_parser
from myscripts.V3_1_1_low_gain.runtime import ResidualScaledRefiner
from myscripts.V3_1_1_low_gain.sweep_residual_scale_v311 import build_parser as build_sweep_parser
from myscripts.V3_1_1_low_gain.sweep_residual_scale_v311 import parse_scales
from myscripts.V3_1_1_low_gain.validate_low_gain_v311 import build_parser as build_validation_parser


def test_low_gain_defaults_are_isolated_and_conservative():
    args = build_validation_parser().parse_args(
        ["--checkpoint", "/tmp/refine.pt", "--data", "/tmp/data.yaml", "--output-dir", "/tmp/out"]
    )
    assert args.ca_weights == CANONICAL_CA_WEIGHTS
    assert args.residual_scale == DEFAULT_RESIDUAL_SCALE == 0.42
    assert args.target_gain == TARGET_MAP_GAIN == 0.04
    assert args.imgsz == 640
    assert args.amp is False


def test_low_gain_sweep_grid_keeps_zero_full_and_target_region():
    args = build_sweep_parser().parse_args(
        ["--checkpoint", "/tmp/refine.pt", "--data", "/tmp/data.yaml", "--output-dir", "/tmp/out"]
    )
    assert args.scales[0] == 0.0
    assert args.scales[-1] == 1.0
    assert 0.42 in args.scales
    assert parse_scales("0.4,0.2,0.4") == (0.2, 0.4)


def test_runtime_scales_only_residual_and_delegates_methods():
    class DummyRefiner:
        marker = "delegated"

        def __call__(self, *_args, **_kwargs):
            return {"residual": 10.0, "quality": 7.0}

        def eval(self):
            return self

    wrapped = ResidualScaledRefiner(DummyRefiner(), 0.42)
    output = wrapped(None)
    assert output == {"residual": 4.2, "quality": 7.0}
    assert wrapped.marker == "delegated"
    assert wrapped.eval() is wrapped


def test_all_formal_wrappers_default_to_the_frozen_scale():
    profile = build_profile_parser().parse_args(
        ["--checkpoint", "/tmp/refine.pt", "--data", "/tmp/data.yaml", "--output-dir", "/tmp/profile"]
    )
    comparative = build_comparative_parser().parse_args(
        [
            "--refine-profile-summary", "/tmp/profile.json",
            "--data", "/tmp/data.yaml",
            "--output-dir", "/tmp/comparative",
        ]
    )
    exported = build_export_parser().parse_args(
        ["--checkpoint", "/tmp/refine.pt", "--data", "/tmp/data.yaml", "--output-dir", "/tmp/export"]
    )
    audit = build_audit_parser().parse_args(
        [
            "--fp32-batch8-dir", "/tmp/fp32b8",
            "--amp-batch8-dir", "/tmp/ampb8",
            "--fp32-batch1-dir", "/tmp/fp32b1",
            "--output-dir", "/tmp/audit",
        ]
    )
    collector = build_collector_parser().parse_args([])
    assert {
        profile.residual_scale,
        comparative.residual_scale,
        exported.residual_scale,
        audit.residual_scale,
        collector.residual_scale,
    } == {0.42}


def test_collector_calls_complete_low_gain_evidence_chain():
    args = argparse.Namespace(
        tests=True,
        qualitative=True,
        baseline_weights="/tmp/baseline.pt",
        ca_weights="/tmp/ca.pt",
        checkpoint="/tmp/refine.pt",
        data="/tmp/data.yaml",
        residual_scale=0.42,
        device="0",
        workers=8,
    )
    stages = build_stages(args, Path("/tmp/evidence"))
    names = [stage.name for stage in stages]
    assert names == [
        "tests",
        "baseline_ca_validation",
        "reproduction_fp32_batch8",
        "reproduction_amp_batch8",
        "reproduction_fp32_batch1",
        "assemble_main_results",
        "reproduction_audit",
        "profile_refine_fp32_batch1",
        "profile_comparative_fp32_batch1",
        "qualitative_predictions",
        "qualitative_figure",
    ]
    commands = "\n".join(" ".join(stage.command) for stage in stages)
    assert "myscripts.V3_1_1_low_gain.validate_low_gain_v311" in commands
    assert "myscripts.V3_1_1_low_gain.profile_refine_low_gain_v311" in commands
    assert "myscripts.V3_1_1_low_gain.profile_comparative_low_gain_v311" in commands
    assert "myscripts.V3_1_1_low_gain.export_qualitative_low_gain_v311" in commands
