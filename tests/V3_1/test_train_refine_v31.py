"""Torch-light policy tests for Refine V3.1 training."""

import inspect

from myscripts.V3_1 import train_refine_v31
from myscripts.V3_1.runtime import evaluate_refiner_v31


def test_v31_parser_freezes_original_split_and_conservative_defaults():
    parser = train_refine_v31.build_parser()
    args = parser.parse_args(
        ["--experiment", "geometry_only", "--data", "/data/dataset.yaml", "--output-dir", "/tmp/out"]
    )
    assert args.holdout_fraction == 0.20
    assert args.seed == 0
    assert args.imgsz == 640
    assert args.short_negative_limit == 0.50
    assert args.short_positive_limit == 0.20
    assert args.long_negative_limit == 0.08
    assert args.long_positive_limit == 0.08
    assert args.minimum_map_gain == 0.03


def test_v31_holdout_key_is_metric_first_and_prefers_earlier_exact_tie():
    earlier = {"epoch": 5, "row": {"map50_95": 0.50, "ap75": 0.55, "ap90": 0.20}}
    later = {"epoch": 10, "row": {"map50_95": 0.50, "ap75": 0.55, "ap90": 0.20}}
    better_ap75 = {"epoch": 10, "row": {"map50_95": 0.50, "ap75": 0.56, "ap90": 0.19}}
    assert train_refine_v31._candidate_key(earlier) > train_refine_v31._candidate_key(later)
    assert train_refine_v31._candidate_key(better_ap75) > train_refine_v31._candidate_key(earlier)


def test_v31_evaluator_has_no_second_nms_or_quality_gate():
    source = inspect.getsource(evaluate_refiner_v31)
    assert "rerun_rotated_nms" not in source
    assert "quality >=" not in source
    assert 'variants = ("coarse", "identity", "refined")' in source
