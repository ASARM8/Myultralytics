"""Train a clean YOLO26-OBB comparison model on the same dataset protocol.

Default settings mirror the 640/811 YOLO11-OBB comparison protocol:
    python train_yolo26_obb.py

Common overrides:
    python train_yolo26_obb.py --data /root/autodl-tmp/datasets/TTPLA-640-811/dataset.yaml --device 0
    python train_yolo26_obb.py --imgsz 1024 --batch 8 --name yolo26_obb_1024_l_scratch
"""

from __future__ import annotations

import argparse
import os

os.environ["OMP_NUM_THREADS"] = "8"

from ultralytics import YOLO


MODEL_YAML = "ultralytics/cfg/models/26/yolo26l-obb.yaml"

DEFAULT_CONFIG = {
    # Dataset and output protocol.
    "data": "/root/autodl-tmp/datasets/TTPLA-640-811/dataset.yaml",
    "project": "/root/autodl-tmp/work-dirs",
    "name": "yolo26_obb_640_811_l_scratch",
    "exist_ok": False,
    # Training scale.
    "epochs": 300,
    "batch": 16,
    "imgsz": 640,
    "device": 0,
    "workers": 16,
    "patience": 0,
    # Checkpoints and plots.
    "save": True,
    "save_period": 10,
    "val": True,
    "plots": True,
    # Optimization. Keep this aligned with YOLO11 baseline/CA/refine comparisons.
    "pretrained": False,
    "optimizer": "AdamW",
    "lr0": 0.0003,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 10.0,
    "cos_lr": True,
    "close_mosaic": 100,
    # Augmentation. Keep this aligned with YOLO11 baseline/CA/refine comparisons.
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.05,
    "scale": 0.3,
    "fliplr": 0.5,
    "flipud": 0.0,
    "mosaic": 0.5,
    "mixup": 0.0,
    # Misc.
    "amp": False,
    "cache": "disk",
    "resume": False,
    "seed": 0,
    "verbose": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO26-OBB as a clean comparison baseline.")
    parser.add_argument("--model", default=MODEL_YAML, help="YOLO26 OBB YAML, e.g. yolo26l-obb.yaml.")
    parser.add_argument("--data", default=DEFAULT_CONFIG["data"], help="Dataset YAML path.")
    parser.add_argument("--project", default=DEFAULT_CONFIG["project"], help="Output project directory.")
    parser.add_argument("--name", default=DEFAULT_CONFIG["name"], help="Run name.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch", type=int, default=DEFAULT_CONFIG["batch"])
    parser.add_argument("--imgsz", type=int, default=DEFAULT_CONFIG["imgsz"])
    parser.add_argument("--device", default=str(DEFAULT_CONFIG["device"]))
    parser.add_argument("--workers", type=int, default=DEFAULT_CONFIG["workers"])
    parser.add_argument("--exist-ok", action="store_true", help="Allow writing into an existing run directory.")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted training.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    config = DEFAULT_CONFIG.copy()
    config.update(
        {
            "data": args.data,
            "project": args.project,
            "name": args.name,
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "device": args.device,
            "workers": args.workers,
            "exist_ok": args.exist_ok,
            "resume": args.resume,
        }
    )
    return config


def main():
    args = parse_args()
    config = build_config(args)

    print("=" * 72)
    print("YOLO26-OBB clean comparison training")
    print(f"Model:   {args.model}")
    print(f"Data:    {config['data']}")
    print(f"Epochs:  {config['epochs']}")
    print(f"Batch:   {config['batch']}")
    print(f"Image:   {config['imgsz']}")
    print(f"Device:  {config['device']}")
    print(f"Output:  {config['project']}/{config['name']}")
    print("=" * 72)

    model = YOLO(args.model)
    return model.train(**config)


if __name__ == "__main__":
    main()
