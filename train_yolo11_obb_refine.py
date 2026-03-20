"""
YOLOv11-OBB + Refine Head 训练脚本
基于 CA (Coverage-Aware) 模型，新增轻量宽高精修分支（Δw, Δh）。
使用方法:
    直接运行: python train_yolo11_obb_refine.py
    修改下方 CONFIG 字典中的参数即可自定义训练配置

关键设计：
    - 从 yolo11-obb-ca-refine.yaml 构建模型（含 OBBRefine 头 + cv5 精修分支）
    - 通过回调加载 CA best.pt 预训练权重（cv5 零初始化保持 identity）
    - aux_geo 设为 0.2（作为 refine residual 宽高监督增益）
"""

import argparse
import os
import sys
import datetime
import re
os.environ["OMP_NUM_THREADS"] = "8"

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import intersect_dicts

# ========================== 终端日志保存配置 ==========================
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
TQDM_BAR = re.compile(
    r'(\d+/\d+\s+[\d.]+G\s+[\d.]+)'
    r'|(━━━━━|━━━╸|━━╸|━╸|━)'
    r'|(\d+%\s*\|?[█▉▊▋▌▍▎▏]+)'
    r'|(Class\s+Images\s+Instances.*?:\s*\d+%)'
)

class Logger(object):
    """带缓冲与过滤的日志类"""
    def __init__(self, stream):
        self.terminal = stream
        self.log_file = None
        self.buffer = []

    def set_log_file(self, filepath):
        self.log_file = open(filepath, 'a', encoding='utf-8')
        if self.buffer:
            self.log_file.write("".join(self.buffer))
            self.log_file.flush()
            self.buffer = []

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        clean_msg = ANSI_ESCAPE.sub('', message)
        if TQDM_BAR.search(clean_msg):
            return
        if '\r' in clean_msg:
            parts = [p for p in clean_msg.split('\r') if p]
            if parts:
                clean_msg = parts[-1]
            else:
                return
        if not clean_msg.strip('\n'):
            return
        if self.log_file is not None:
            self.log_file.write(clean_msg)
            self.log_file.flush()
        else:
            self.buffer.append(clean_msg)

    def flush(self):
        self.terminal.flush()
        if self.log_file is not None:
            self.log_file.flush()

# ========================== 路径配置 ==========================
# CA 预训练权重路径（用于初始化，cv5 不在其中，将保持零初始化）
PRETRAIN_WEIGHTS = "/root/autodl-tmp/work-dirs/yolo11_obb-ca/weights/best.pt"

# OBBRefine YAML 配置路径（含 cv5 精修分支定义）
MODEL_YAML = "ultralytics/cfg/models/11/yolo11l-obb-ca-refine.yaml"
RUN_NAME = "yolo11_obb-ca-refine-decouple"

VAL_WEIGHTS = f"/root/autodl-tmp/work-dirs/{RUN_NAME}/weights/best.pt"

# ========================== 训练配置 ==========================
CONFIG = {
    # ---------- 模型配置 ----------
    # 从 YAML 构建新架构（不从 .pt 加载，权重通过回调注入）
    "model": MODEL_YAML,

    # ---------- 数据集配置 ----------
    "data": "/root/autodl-tmp/dataset/TTPLA-1024/dataset.yaml",

    # ---------- 训练基本参数 ----------
    "epochs": 150,
    "batch": 16,
    "imgsz": 1024,
    "device": 0,
    "workers": 16,  # 临时设为 0 排查多进程崩溃根因；确认无误后改回 8
    "patience": 0,

    # ---------- 输出目录配置 ----------
    "project": "/root/autodl-tmp/work-dirs",
    "name": RUN_NAME,
    "exist_ok": False,

    # ---------- 模型保存配置 ----------
    "save": True,
    "save_period": 50,

    # ---------- 验证与可视化 ----------
    "val": True,
    "plots": True,

    # ---------- 训练策略 ----------
    "pretrained": False,
    "optimizer": "AdamW",
    "lr0": 0.0001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 5.0,
    "cos_lr": True,
    "close_mosaic": 10,

    # ---------- 数据增强配置 ----------
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "fliplr": 0.5,
    "flipud": 0.0,
    "mosaic": 1.0,
    "mixup": 0.0,

    # ---------- 损失权重 ----------
    # aux_geo: 辅助几何损失增益（梯度现在通过 cv5 连续参数有效流回）
    "aux_geo": 0.2,
    "aux_geo_lp": 0.0,    # L_perp（法向偏移）：本次不做 Δn，关闭
    "aux_geo_lw": 2.0,
    "aux_geo_lt": 0.0,
    "aux_geo_ar": 30.0,             
    "aux_geo_ws": 16.0,

    # ---------- 其他 ----------
    "amp": True,
    "cache": 'disk',
    "resume": False,
    "seed": 0,
    "verbose": True,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "val_ab"], default="train")
    parser.add_argument("--weights", default=VAL_WEIGHTS)
    parser.add_argument("--data", default=CONFIG["data"])
    parser.add_argument("--imgsz", type=int, default=CONFIG["imgsz"])
    parser.add_argument("--batch", type=int, default=CONFIG["batch"])
    parser.add_argument("--device", default=str(CONFIG["device"]))
    return parser.parse_args()


def set_refine_inference_mode(yolo_model, disable_refine_inference: bool):
    from ultralytics.nn.modules.head import OBBRefine

    model_root = yolo_model.model if hasattr(yolo_model, "model") else yolo_model
    found = 0
    for m in model_root.modules():
        if isinstance(m, OBBRefine):
            m.disable_refine_inference = bool(disable_refine_inference)
            found += 1
    if found == 0:
        raise RuntimeError("未找到 OBBRefine 模块，无法切换 coarse-only 验证模式。")


def sync_obbrefine_runtime_attrs(target_model, refine_select_ar: float, refine_select_ws: float, disable_refine_inference: bool):
    from ultralytics.nn.modules.head import OBBRefine

    model_root = target_model.model if hasattr(target_model, "model") else target_model
    found = 0
    for m in model_root.modules():
        if isinstance(m, OBBRefine):
            m.refine_select_ar = float(refine_select_ar)
            m.refine_select_ws = float(refine_select_ws)
            m.disable_refine_inference = bool(disable_refine_inference)
            found += 1
    if found == 0:
        raise RuntimeError("未找到 OBBRefine 模块，无法同步运行时属性。")


def run_val_ab(args):
    sys.stdout = Logger(sys.stdout)
    sys.stderr = Logger(sys.stderr)

    model = YOLO(args.weights)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    common_kwargs = {
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": CONFIG["project"],
        "exist_ok": True,
        "plots": True,
    }
    metric_keys = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    metric_labels = {
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
        "metrics/mAP50(B)": "mAP50",
        "metrics/mAP50-95(B)": "mAP50-95",
    }

    print("=" * 60)
    print(f"  A/B 验证权重: {args.weights}")
    print(f"  数据集: {args.data}")
    print(f"  图片尺寸: {args.imgsz}")
    print(f"  批次大小: {args.batch}")
    print("=" * 60)

    set_refine_inference_mode(model, False)
    normal_metrics = model.val(name=f"{CONFIG['name']}-val-normal-{stamp}", **common_kwargs)
    normal_dict = normal_metrics.results_dict

    set_refine_inference_mode(model, True)
    coarse_metrics = model.val(name=f"{CONFIG['name']}-val-coarse-only-{stamp}", **common_kwargs)
    coarse_dict = coarse_metrics.results_dict

    print("\n" + "=" * 60)
    print("  A/B 验证结果对比")
    print("=" * 60)
    for key in metric_keys:
        normal_value = float(normal_dict[key])
        coarse_value = float(coarse_dict[key])
        delta = coarse_value - normal_value
        print(
            f"  {metric_labels[key]}: normal={normal_value:.5f}, coarse-only={coarse_value:.5f}, Δ={delta:+.5f}"
        )
    print("=" * 60)

    return {"normal": normal_dict, "coarse_only": coarse_dict}


def main():
    """主训练函数"""
    sys.stdout = Logger(sys.stdout)
    sys.stderr = Logger(sys.stderr)
    print("\n[*] 终端日志拦截器已启动\n")

    # 1. 从 YAML 构建模型（含 OBBRefine 头）
    model = YOLO(CONFIG["model"])

    # 2. 注册回调：训练开始前加载 CA 预训练权重
    def on_pretrain_routine_end(trainer):
        """在模型构建完成后、训练循环开始前，注入 CA 预训练权重。"""
        print(f"\n[*] 加载预训练权重: {PRETRAIN_WEIGHTS}")

        ckpt = torch.load(PRETRAIN_WEIGHTS, map_location=trainer.device, weights_only=False)
        if "model" in ckpt:
            src_state = ckpt["model"].float().state_dict()
        else:
            src_state = ckpt

        # intersect_dicts 只加载 key 和 shape 都匹配的参数
        # cv5 的参数不在 CA checkpoint 中，保持零初始化
        model_state = trainer.model.state_dict()
        loaded = intersect_dicts(src_state, model_state)
        trainer.model.load_state_dict(loaded, strict=False)

        n_loaded = len(loaded)
        n_total = len(model_state)
        n_new = n_total - n_loaded
        print(f"[*] 已加载 {n_loaded}/{n_total} 参数，{n_new} 个新参数（cv5 精修分支）保持零初始化")

        # 同步 EMA
        if hasattr(trainer, 'ema') and trainer.ema is not None:
            trainer.ema.ema.load_state_dict(trainer.model.state_dict())
            trainer.ema.updates = 0
            print("[*] EMA 已同步")

        sync_obbrefine_runtime_attrs(
            trainer.model,
            trainer.args.aux_geo_ar,
            trainer.args.aux_geo_ws,
            True,
        )
        if hasattr(trainer, 'ema') and trainer.ema is not None:
            sync_obbrefine_runtime_attrs(
                trainer.ema.ema,
                trainer.args.aux_geo_ar,
                trainer.args.aux_geo_ws,
                True,
            )
        print(
            f"[*] OBBRefine 运行时属性已同步: AR>{float(trainer.args.aux_geo_ar)}, "
            f"short<{float(trainer.args.aux_geo_ws)}px, 默认验证/推理=coarse-only"
        )

    model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)

    # 3. 注册回调：日志文件落地
    def on_train_start(trainer):
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_log_file = os.path.join(trainer.save_dir, f"train_terminal_{time_str}.log")
        print(f"\n[*] 日志文件: {final_log_file}\n")
        sys.stdout.set_log_file(final_log_file)
        sys.stderr.set_log_file(final_log_file)

    model.add_callback("on_train_start", on_train_start)

    # 4. 打印配置
    print("=" * 60)
    print(f"  模型 YAML: {MODEL_YAML}")
    print(f"  预训练权重: {PRETRAIN_WEIGHTS}")
    print(f"  数据集: {CONFIG['data']}")
    print(f"  训练轮数: {CONFIG['epochs']}")
    print(f"  批次大小: {CONFIG['batch']}")
    print(f"  图片尺寸: {CONFIG['imgsz']}")
    print(f"  输出目录: {CONFIG['project']}/{CONFIG['name']}")
    print(f"  aux_geo 增益: {CONFIG['aux_geo']}")
    print(f"  Refine Head: Δw + Δh (ne_refine=2, 完全解耦)")
    print("  验证/推理默认口径: coarse-only（默认禁用 refine inference）")
    print("  A/B 对照入口: python train_yolo11_obb_refine.py --mode val_ab --weights <ckpt>")
    print("=" * 60)

    # 5. 开始训练
    results = model.train(**CONFIG)

    # 6. 训练完成
    print("\n" + "=" * 60)
    print("  训练完成！")
    print(f"  结果保存在: {CONFIG['project']}/{CONFIG['name']}")
    print("  保存内容说明:")
    print("    - weights/best.pt     : 按 coarse-only 验证指标选出的最佳模型")
    print("    - weights/last.pt     : 最后一轮权重")
    print("    - results.csv         : 每轮训练指标（默认 coarse-only 验证口径）")
    print("    - results.png         : 训练曲线图")
    print("    - args.yaml           : 完整训练参数记录")
    print("=" * 60)

    return results


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.mode == "val_ab":
        run_val_ab(cli_args)
    else:
        main()
