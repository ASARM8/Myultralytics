"""
WAG 诊断训练脚本
================
只跑 2-3 个 epoch，收集 aux_geo 损失的门控激活率、量级等统计信息。
用于判断 WAG 无效的根因。

用法: python train_wag_diag.py
"""
import os
import sys
os.environ["OMP_NUM_THREADS"] = "8"

from ultralytics import YOLO

# 导入诊断 patch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "myscripts"))
from wag_diag_patch import apply_wag_diag_patch, save_diag_results, print_diag_summary


CONFIG = {
    # 使用 WAG 训练的最新权重，或 CA best.pt
    "model": "/root/autodl-tmp/work-dirs/yolo11_obb-ca-wag/weights/last.pt",

    "data": "/root/autodl-tmp/dataset/TTPLA-1024/dataset.yaml",

    # 只跑 2 个 epoch 做诊断
    "epochs": 2,
    "batch": 16,
    "imgsz": 1024,
    "device": 0,
    "workers": 16,

    "project": "/root/autodl-tmp/work-dirs",
    "name": "wag_diag",
    "exist_ok": False,

    "save": False,
    "val": True,
    "plots": False,

    "optimizer": "AdamW",
    "lr0": 0.0001,
    "cos_lr": True,

    # WAG 参数（与正式训练一致）
    "aux_geo": 1.5,
    "aux_geo_lp": 1.0,
    "aux_geo_lw": 2.0,
    "aux_geo_lt": 0.5,
    "aux_geo_ar": 30.0,
    "aux_geo_ws": 12.0,

    "amp": True,
    "cache": "disk",
    "resume": False,
    "verbose": True,
}


def main():
    model = YOLO(CONFIG["model"])

    # 应用诊断 patch（每 10 步打印一次）
    apply_wag_diag_patch(model, log_every=10)

    # 注册回调：训练结束后保存诊断结果
    def on_train_end(trainer):
        save_path = os.path.join(trainer.save_dir, "wag_diag_log.csv")
        save_diag_results(save_path)
        print_diag_summary()

    model.add_callback("on_train_end", on_train_end)

    model.train(**CONFIG)


if __name__ == "__main__":
    main()
