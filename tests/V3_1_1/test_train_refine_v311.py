"""Torch-free protocol tests for Refine V3.1.1."""

from myscripts.V3_1.train_refine_v31 import build_parser, model_config, validate_args


def test_v311_protocol_builds_smooth_target_config():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--refiner-version",
            "v311",
            "--experiment",
            "geometry_only",
            "--target-margin",
            "0.80",
            "--epochs",
            "15",
            "--eval-interval",
            "1",
            "--data",
            "/data/dataset.yaml",
            "--output-dir",
            "/tmp/out",
        ]
    )
    validate_args(parser, args)
    config = model_config(args, 64, 128)
    assert config["supervision_margin"] == 0.80
    assert "target_margin" not in config
    assert "use_quality_aux" not in config
    assert args.epochs == 15
    assert args.eval_interval == 1
    assert args.max_ap95_drop == 0.002
