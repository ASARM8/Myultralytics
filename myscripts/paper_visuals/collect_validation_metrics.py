"""Collect P/R/mAP50/mAP50-95/AP75/AP90 on the cloud server for paper Tables 6 and 9."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_model_spec(value: str) -> tuple[str, Path]:
    """Parse ``LABEL=WEIGHTS``."""
    if "=" not in value:
        raise ValueError(f"--model must use LABEL=WEIGHTS: {value}")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise ValueError(f"Invalid --model value: {value}")
    return label.strip(), Path(path.strip())


def parse_mode_spec(value: str) -> tuple[str, str]:
    """Parse ``LABEL=coarse|normal``."""
    if "=" not in value:
        raise ValueError(f"--refine-mode must use LABEL=coarse|normal: {value}")
    label, mode = (item.strip() for item in value.split("=", 1))
    if mode not in {"coarse", "normal"}:
        raise ValueError(f"Unknown refine mode for {label}: {mode}")
    return label, mode


def set_refine_mode(model, mode: str | None) -> int:
    """Toggle every Refine-capable module when a mode is specified."""
    if mode is None:
        return 0
    changed = 0
    for module in model.model.modules():
        if hasattr(module, "disable_refine_inference"):
            module.disable_refine_inference = mode == "coarse"
            changed += 1
    if changed == 0:
        raise RuntimeError(f"Refine mode '{mode}' requested, but no Refine module was found")
    return changed


def metrics_row(label: str, weights: Path, mode: str | None, results) -> dict:
    """Extract standard metrics plus AP75/AP90 from the full IoU-threshold array."""
    values = results.results_dict
    metric = results.box
    all_ap = metric.all_ap
    ap90 = float(all_ap[:, 8].mean()) if len(all_ap) else float("nan")
    return {
        "method": label,
        "weights": str(weights),
        "refine_mode": mode or "not_applicable",
        "precision": float(values["metrics/precision(B)"]),
        "recall": float(values["metrics/recall(B)"]),
        "map50": float(values["metrics/mAP50(B)"]),
        "map50_95": float(values["metrics/mAP50-95(B)"]),
        "ap75": float(metric.map75),
        "ap90": ap90,
        "preprocess_ms_per_image": float(results.speed.get("preprocess", float("nan"))),
        "inference_ms_per_image": float(results.speed.get("inference", float("nan"))),
        "postprocess_ms_per_image": float(results.speed.get("postprocess", float("nan"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="LABEL=WEIGHTS")
    parser.add_argument("--refine-mode", action="append", default=[], metavar="LABEL=coarse|normal")
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--project", default="work-dirs/paper_validation_metrics")
    args = parser.parse_args()

    from ultralytics import YOLO

    modes = dict(parse_mode_spec(value) for value in args.refine_mode)
    rows = []
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    for label, weights in (parse_model_spec(value) for value in args.model):
        if not weights.exists():
            raise FileNotFoundError(weights)
        model = YOLO(str(weights))
        mode = modes.get(label)
        set_refine_mode(model, mode)
        result = model.val(
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=args.project,
            name=re.sub(r"[^0-9A-Za-z_.-]+", "_", label).strip("_") or "method",
            exist_ok=True,
            plots=False,
            save_json=False,
            verbose=False,
        )
        row = metrics_row(label, weights, mode, result)
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.output_csv, index=False, encoding="utf-8-sig")
        print(
            f"{label}: mAP50-95={row['map50_95']:.5f}, AP75={row['ap75']:.5f}, "
            f"AP90={row['ap90']:.5f}"
        )
    print(args.output_csv)


if __name__ == "__main__":
    main()
