"""Torch-free tests for the isolated V3.1.1 low-gain tools."""

from myscripts.V3_1_1.evidence_runtime import CANONICAL_CA_WEIGHTS
from myscripts.V3_1_1_low_gain import DEFAULT_RESIDUAL_SCALE, TARGET_MAP_GAIN
from myscripts.V3_1_1_low_gain.sweep_residual_scale_v311 import build_parser as build_sweep_parser
from myscripts.V3_1_1_low_gain.sweep_residual_scale_v311 import parse_scales
from myscripts.V3_1_1_low_gain.validate_low_gain_v311 import build_parser as build_validation_parser


def test_low_gain_defaults_are_isolated_and_conservative():
    args = build_validation_parser().parse_args(
        ["--checkpoint", "/tmp/refine.pt", "--data", "/tmp/data.yaml", "--output-dir", "/tmp/out"]
    )
    assert args.ca_weights == CANONICAL_CA_WEIGHTS
    assert args.residual_scale == DEFAULT_RESIDUAL_SCALE == 0.37
    assert args.target_gain == TARGET_MAP_GAIN == 0.04
    assert args.imgsz == 640
    assert args.amp is False


def test_low_gain_sweep_grid_keeps_zero_full_and_target_region():
    args = build_sweep_parser().parse_args(
        ["--checkpoint", "/tmp/refine.pt", "--data", "/tmp/data.yaml", "--output-dir", "/tmp/out"]
    )
    assert args.scales[0] == 0.0
    assert args.scales[-1] == 1.0
    assert 0.37 in args.scales
    assert parse_scales("0.4,0.2,0.4") == (0.2, 0.4)
