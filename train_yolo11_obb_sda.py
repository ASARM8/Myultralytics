"""
YOLOv11-OBB-SDA 训练脚本（Scale-Decoupled Asymmetric Head）
基于 CA (reg_max=32) 预训练权重，加载到 AsymmetricOBB 架构：
  - 主干网络和 FPN 权重完美继承
  - 回归头（cv2）、分类头（cv3）、角度头（cv4）权重完美继承
  - 新增的 fusion_p4 / fusion_p5 层随机初始化（新模块）
  - unshuffle 层无参数，不影响权重加载

使用方法:
    直接运行: python train_yolo11_obb_sda.py
    修改下方 CONFIG 字典中的参数即可自定义训练配置
"""

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

# ========================== 训练配置 ==========================
# 预训练 CA 模型路径（用于权重迁移）
PRETRAIN_WEIGHTS = "/root/autodl-tmp/work-dirs/yolo11_obb-ca/weights/best.pt"

CONFIG = {
    # ---------- 模型配置 ----------
    # 使用 SDA YAML 从零构建架构，然后手动加载 CA 预训练权重
    "model": "yolo11l-obb-sda.yaml",

    # ---------- 数据集配置 ----------
    "data": "/root/autodl-tmp/dataset/TTPLA-1024/dataset.yaml",

    # ---------- 训练基本参数 ----------
    "epochs": 200,
    "batch": 16,
    "imgsz": 1024,
    "device": 0,
    "workers": 16,

    # ---------- 输出目录配置 ----------
    "project": "/root/autodl-tmp/work-dirs",
    "name": "yolo11_obb-sda",
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
    "lr0": 0.0005,              # 稍低的学习率：主干已收敛，新模块需要温和学习
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 5.0,       # 较短预热：大部分权重已预训练
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

    # ---------- 其他 ----------
    "amp": True,
    "cache": 'disk',
    "resume": False,
    "seed": 0,
    "verbose": True,
}


def load_pretrained_weights(model, pretrain_path):
    """从 CA 预训练模型加载权重到 SDA 模型。

    AsymmetricOBB 继承自 OBB，新增的层：
      - unshuffle_p4 / unshuffle_p5: PixelUnshuffle，无可学习参数
      - fusion_p4 / fusion_p5: 1x1 Conv，随机初始化

    其余层（backbone, FPN, cv2, cv3, cv4, dfl）均可从 CA 模型完美迁移。
    """
    print(f"\n[*] 加载预训练权重: {pretrain_path}")

    ckpt = torch.load(pretrain_path, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        # .pt 文件包含完整 checkpoint
        pretrain_sd = ckpt["model"].float().state_dict()
    else:
        pretrain_sd = ckpt

    model_sd = model.model.state_dict()

    # 使用 intersect_dicts 匹配形状兼容的权重
    matched = intersect_dicts(pretrain_sd, model_sd, exclude=[])
    n_total = len(model_sd)
    n_matched = len(matched)

    model.model.load_state_dict(matched, strict=False)

    # 统计新增层（未匹配的参数）
    new_keys = [k for k in model_sd if k not in matched]
    new_param_count = sum(model_sd[k].numel() for k in new_keys)

    print(f"  总参数组: {n_total}")
    print(f"  匹配加载: {n_matched}")
    print(f"  新增层数: {len(new_keys)} (共 {new_param_count:,} 参数，随机初始化)")

    if new_keys:
        print(f"  新增层明细:")
        for k in new_keys:
            print(f"    - {k}  shape={list(model_sd[k].shape)}")

    return model


def main():
    """主训练函数"""
    sys.stdout = Logger(sys.stdout)
    sys.stderr = Logger(sys.stderr)
    print("\n[*] 终端日志拦截器已启动\n")

    # 1. 从 YAML 构建 SDA 模型架构
    print("=" * 60)
    print("  YOLOv11-OBB-SDA 训练")
    print("=" * 60)
    print(f"  模型架构: {CONFIG['model']}")
    print(f"  预训练权重: {PRETRAIN_WEIGHTS}")
    print(f"  数据集: {CONFIG['data']}")
    print(f"  训练轮数: {CONFIG['epochs']}")
    print(f"  批次大小: {CONFIG['batch']}")
    print(f"  图片尺寸: {CONFIG['imgsz']}")
    print(f"  输出目录: {CONFIG['project']}/{CONFIG['name']}")
    print("=" * 60)

    model = YOLO(CONFIG["model"])

    # 2. 加载 CA 预训练权重
    model = load_pretrained_weights(model, PRETRAIN_WEIGHTS)

    # 3. 注册日志回调
    def on_train_start(trainer):
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_log_file = os.path.join(trainer.save_dir, f"train_terminal_{time_str}.log")
        print(f"\n[*] 日志文件: {final_log_file}\n")
        sys.stdout.set_log_file(final_log_file)
        sys.stderr.set_log_file(final_log_file)

    model.add_callback("on_train_start", on_train_start)

    # 4. 开始训练
    results = model.train(**CONFIG)

    # 5. 完成
    print("\n" + "=" * 60)
    print("  训练完成！")
    print(f"  结果保存在: {CONFIG['project']}/{CONFIG['name']}")
    print("  保存内容说明:")
    print("    - weights/best.pt     : 最佳模型权重")
    print("    - weights/last.pt     : 最后一轮权重")
    print("    - results.csv         : 每轮训练指标")
    print("    - results.png         : 训练曲线图")
    print("    - args.yaml           : 完整训练参数记录")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
