"""TTPLA instance segmentation downstream of a frozen CA + V3.1.1 detector.

Cache once per split, then reuse that split's cache across all mask-head ablations.
Image and JSON folders are sibling tile splits, e.g. images/train and jsons/train.
Use `python train_wire_seg.py COMMAND --help` for the required paths.
"""
from __future__ import annotations

import argparse
import json


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("cache", help="Run frozen CA + low-gain Refine on an image folder")
    cache.add_argument("--images", required=True)
    cache.add_argument("--ca", required=True, help="CA best.pt used by the Refine training run")
    cache.add_argument("--refine", required=True, help="V3.1.1 refiner checkpoint")
    cache.add_argument("--wire-class", type=int, help="Required for multi-class detectors")
    cache.add_argument("--output", required=True, help="Destination JSON for this split")
    cache.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu")
    for name, desc in [("audit", "Measure available geometry and interference without training"),
                       ("train", "Fit only the instance mask head on frozen proposals"),
                       ("eval", "Evaluate all proposals and the matched, exposed subset"),
                       ("paired", "Compare CA vs Refine geometry with the same frozen axes head and swap pairs"),
                       ("neighbor-pairs", "Score four frozen heads on all ground-truth neighboring wire pairs"),
                       ("predict", "Export per-instance binary PNG masks and their detections")]:
        cmd = commands.add_parser(name, help=desc)
        cmd.add_argument("--images", required=True)
        cmd.add_argument("--limit", type=int, help="Use evenly spaced tiles for a quick probe")
        if name == "paired":
            cmd.set_defaults(boxes="coarse")
        elif name == "neighbor-pairs":
            cmd.set_defaults(boxes="refined")
        else:
            cmd.add_argument("--boxes", choices=("coarse", "refined"), default="refined" if name in {"train","audit"} else None,
                             help="CA or refined proposal geometry; inference defaults to checkpoint setting")
        cmd.add_argument("--match-iou", type=float, default=0.30, help="Minimum coarse OBB IoU for instance pairing")
        if name == "audit":
            source = cmd.add_mutually_exclusive_group(required=True)
            source.add_argument("--cache")
            source.add_argument("--truth-only", action="store_true", help="Geometry feasibility only, no detector claims")
        else:
            cmd.add_argument("--cache", required=True)
        if name != "predict":
            cmd.add_argument("--jsons", required=True, help="LabelMe per-instance polygon directory")
        if name in {"train", "audit"}:
            cmd.add_argument("--margin", type=float, default=12, help="Context per short side in native pixels")
        if name != "audit":
            cmd.add_argument("--device", default="cuda:0", help="PyTorch device, e.g. cuda:0 or cpu")
        cmd.add_argument("--output", required=name != "audit", help="Report, model checkpoint or prediction folder")
        if name == "train":
            cmd.add_argument("--prompt", choices=("target", "points", "boxes", "axes"), default="axes")
            cmd.add_argument("--batch-size", type=int, default=1, help="Maximum proposals per optimizer update")
            cmd.add_argument("--max-batch-pixels", type=int, default=2097152,
                             help="Maximum padded crop pixels per batch; a larger single crop runs alone")
            cmd.add_argument("--workers", type=int, default=0, help="CPU processes preparing native-resolution crops")
            cmd.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False,
                             help="CUDA mixed precision, preferring BF16 on supported GPUs")
            cmd.add_argument("--epochs", type=int, default=15)
            cmd.add_argument("--lr", type=float, default=0.0003)
            cmd.add_argument("--seed", type=int, default=0)
        if name in {"predict", "eval", "paired", "neighbor-pairs"}:
            cmd.add_argument("--threshold", type=float, default=0.5)
            if name == "neighbor-pairs":
                cmd.add_argument("--weights-dir", required=True, help="Parent of target/points/boxes/axes mask.pt directories")
            else:
                cmd.add_argument("--weights", required=True)
        if name == "eval":
            cmd.add_argument("--swap-target", action="store_true",
                             help="Also switch identity on the very same crop when a matched neighbor is visible")
    return cli


def main(argv=None):
    import math
    cli = parser()
    args = cli.parse_args(argv)
    if args.command == "cache":
        from myscripts.wire_seg.proposals import cache_proposals
        cache_proposals(args)
        return
    if args.limit is not None and args.limit < 1:
        cli.error("--limit must be positive")
    if not 0 < args.match_iou <= 1:
        cli.error("--match-iou must be in (0, 1]")
    if hasattr(args, "margin") and not (math.isfinite(args.margin) and args.margin >= 0):
        cli.error("--margin must be finite and nonnegative")
    if args.command == "audit":
        from myscripts.wire_seg.data import audit
        from myscripts.wire_seg.proposals import save_json
        report = audit(args)
        print(json.dumps(report, indent=2))
        if args.output:
            save_json(args.output, report)
    elif args.command == "train":
        if args.batch_size < 1 or args.max_batch_pixels < 1 or args.workers < 0:
            cli.error("--batch-size and --max-batch-pixels must be positive; --workers must be nonnegative")
        if args.epochs < 1 or not (math.isfinite(args.lr) and args.lr > 0):
            cli.error("--epochs and --lr must be positive")
        from myscripts.wire_seg.learning import train
        train(args)
    else:
        if not 0 < args.threshold < 1:
            cli.error("--threshold must be in (0, 1)")
        if args.command == "paired":
            from myscripts.wire_seg.paired import run_paired
            run_paired(args)
            return
        if args.command == "neighbor-pairs":
            from myscripts.wire_seg.neighbor_pairs import run_neighbor_pairs
            run_neighbor_pairs(args)
            return
        from myscripts.wire_seg.learning import run_masks
        run_masks(args)


if __name__ == "__main__":
    main()
