"""Collect parameters, GFLOPs, validation latency, FPS, and peak GPU memory for paper Table 11."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_model_spec(value: str) -> tuple[str, Path]:
    """Parse ``LABEL=MODEL`` where MODEL is a YAML file or checkpoint."""
    if "=" not in value:
        raise ValueError(f"--model must use LABEL=MODEL: {value}")
    label, source = value.split("=", 1)
    if not label.strip() or not source.strip():
        raise ValueError(f"Invalid --model value: {value}")
    return label.strip(), Path(source.strip())


def parse_mode_spec(value: str) -> tuple[str, str]:
    """Parse ``LABEL=coarse|normal``."""
    if "=" not in value:
        raise ValueError(f"--refine-mode must use LABEL=coarse|normal: {value}")
    label, mode = (item.strip() for item in value.split("=", 1))
    if mode not in {"coarse", "normal"}:
        raise ValueError(f"Unknown refine mode for {label}: {mode}")
    return label, mode


def set_refine_mode(model, mode: str | None) -> int:
    """Apply the requested inference mode to every Refine-capable module."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, metavar="LABEL=MODEL")
    parser.add_argument("--refine-mode", action="append", default=[], metavar="LABEL=coarse|normal")
    parser.add_argument("--data", required=True, help="Validation dataset YAML; use the same file for every method")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1, help="Use batch=1 for deployment-oriented latency")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--project", default="work-dirs/paper_complexity")
    args = parser.parse_args()

    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    modes = dict(parse_mode_spec(value) for value in args.refine_mode)
    rows = []
    for label, source in (parse_model_spec(value) for value in args.model):
        if not source.exists():
            raise FileNotFoundError(source)
        model = YOLO(str(source))
        mode = modes.get(label)
        set_refine_mode(model, mode)

        torch_model = model.model
        parameters_m = get_num_params(torch_model) / 1e6
        gflops = float(get_flops(torch_model, imgsz=args.imgsz))

        try:
            import torch

            if torch.cuda.is_available() and str(args.device).lower() != "cpu":
                torch.cuda.reset_peak_memory_stats()
        except (RuntimeError, ValueError):
            pass

        result = model.val(
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=args.project,
            name=label.replace(" ", "_"),
            exist_ok=True,
            plots=False,
            save_json=False,
            verbose=False,
        )
        inference_ms = float(result.speed.get("inference", float("nan")))
        fps = 1000.0 / inference_ms if inference_ms > 0 else float("nan")

        peak_memory_gb = float("nan")
        try:
            import torch

            if torch.cuda.is_available() and str(args.device).lower() != "cpu":
                peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
                torch.cuda.reset_peak_memory_stats()
        except (RuntimeError, ValueError):
            pass

        rows.append(
            {
                "method": label,
                "model": str(source),
                "refine_mode": mode or "not_applicable",
                "parameters_m": parameters_m,
                "gflops": gflops,
                "inference_ms_per_image": inference_ms,
                "fps": fps,
                "peak_gpu_memory_gb": peak_memory_gb,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "device": args.device,
                "latency_scope": "Ultralytics validator inference only; excludes preprocess/postprocess",
            }
        )
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output_csv, index=False, encoding="utf-8-sig")
        print(f"{label}: params={parameters_m:.3f}M, GFLOPs={gflops:.3f}, inference={inference_ms:.3f} ms")

    print(args.output_csv)


if __name__ == "__main__":
    main()
