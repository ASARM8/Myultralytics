"""Evaluate coarse-only and normal-refine mAP50-95 over a checkpoint sequence."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_weight_spec(value: str) -> tuple[int, Path]:
    """Parse ``EPOCH=PATH``."""
    if "=" not in value:
        raise ValueError(f"--weights 必须使用 EPOCH=PATH: {value}")
    epoch, path = value.split("=", 1)
    return int(epoch), Path(path)


def discover_weights(pattern: str, epoch_regex: str) -> list[tuple[int, Path]]:
    """Discover checkpoints and extract epoch numbers from their filenames."""
    regex = re.compile(epoch_regex)
    path = Path(pattern)
    paths = sorted(Path().glob(pattern)) if not path.is_absolute() else sorted(path.parent.glob(path.name))
    discovered = []
    for path in paths:
        match = regex.search(path.stem)
        if not match:
            raise ValueError(f"无法从权重文件名提取 epoch: {path}; 请调整 --epoch-regex")
        discovered.append((int(match.group(1)), path))
    return discovered


def set_refine_mode(model, *, coarse_only: bool) -> int:
    """Toggle every implemented OBBRefine-like module without hard-coding a class import."""
    changed = 0
    for module in model.model.modules():
        if hasattr(module, "disable_refine_inference"):
            module.disable_refine_inference = bool(coarse_only)
            changed += 1
    if changed == 0:
        raise RuntimeError("权重中未找到带 disable_refine_inference 属性的 Refine 模块")
    return changed


def metrics_row(epoch: int, method: str, results) -> dict:
    """Convert Ultralytics validation output to the tidy curve schema."""
    values = results.results_dict
    metric = results.box
    all_ap = metric.all_ap
    ap90 = float(all_ap[:, 8].mean()) if len(all_ap) else float("nan")
    return {
        "epoch": epoch,
        "method": method,
        "precision": float(values["metrics/precision(B)"]),
        "recall": float(values["metrics/recall(B)"]),
        "map50": float(values["metrics/mAP50(B)"]),
        "map50_95": float(values["metrics/mAP50-95(B)"]),
        "ap75": float(metric.map75),
        "ap90": ap90,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", action="append", default=[], metavar="EPOCH=PATH")
    parser.add_argument("--weights-glob", help="例如 work-dirs/exp/weights/epoch*.pt")
    parser.add_argument("--epoch-regex", default=r"epoch(\d+)")
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--project", default="work-dirs/paper_refine_ab_curve")
    args = parser.parse_args()

    checkpoints = [parse_weight_spec(value) for value in args.weights]
    if args.weights_glob:
        checkpoints.extend(discover_weights(args.weights_glob, args.epoch_regex))
    checkpoints = sorted(dict(checkpoints).items())
    if not checkpoints:
        raise ValueError("未提供任何 checkpoint")

    from ultralytics import YOLO

    rows = []
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    for epoch, path in checkpoints:
        if not path.exists():
            raise FileNotFoundError(path)
        model = YOLO(str(path))
        common = {
            "data": args.data,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "project": args.project,
            "plots": False,
            "save_json": False,
            "verbose": False,
        }
        set_refine_mode(model, coarse_only=True)
        coarse = model.val(name=f"epoch{epoch:04d}_coarse", exist_ok=True, **common)
        rows.append(metrics_row(epoch, "CA+Refine (coarse-only)", coarse))

        set_refine_mode(model, coarse_only=False)
        normal = model.val(name=f"epoch{epoch:04d}_normal", exist_ok=True, **common)
        rows.append(metrics_row(epoch, "CA+Refine (normal refine)", normal))
        pd.DataFrame(rows).to_csv(args.output_csv, index=False, encoding="utf-8-sig")
        print(f"epoch={epoch}: coarse={rows[-2]['map50_95']:.5f}, normal={rows[-1]['map50_95']:.5f}")

    print(args.output_csv)


if __name__ == "__main__":
    main()
