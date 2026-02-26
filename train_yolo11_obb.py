"""
YOLOv11-OBB 自定义数据集训练脚本
使用方法:
    直接运行: python train_yolo11_obb.py
    修改下方 CONFIG 字典中的参数即可自定义训练配置
"""

import os
os.environ["OMP_NUM_THREADS"] = "8"  # 修复 libgomp 警告

from ultralytics import YOLO

# ========================== 训练配置 ==========================
CONFIG = {
    # ---------- 模型配置 ----------
    # 使用 .yaml 文件从零开始训练（不使用预训练权重）
    # 可选规模: yolo11n-obb.yaml / yolo11s-obb.yaml / yolo11m-obb.yaml / yolo11l-obb.yaml / yolo11x-obb.yaml
    "model": "yolo11l-obb.yaml",

    # ---------- 数据集配置 ----------
    # 指向你的自定义数据集 yaml 文件
    "data": "/root/autodl-tmp/dataset/TTPLA-1024/dataset.yaml",

    # ---------- 训练基本参数 ----------
    "epochs": 300,              # 训练总轮数
    "batch": 16,                # 批次大小（RTX 5090 32GB 显存，yolo11l + 1024 尝试 32，OOM 则降回 16）
    "imgsz": 1024,              # 输入图片尺寸
    "device": 0,                # GPU 设备编号
    "workers": 8,               # 数据加载线程数（5090 性能强，可用更多线程）

    # ---------- 输出目录配置 ----------
    "project": "/root/autodl-tmp/work-dirs",          # 输出主目录
    "name": "yolo11_obb",        # 实验名称（子目录）
    "exist_ok": False,                      # 是否允许覆盖已有同名目录

    # ---------- 模型保存配置 ----------
    "save": True,               # 保存训练检查点（best.pt 和 last.pt）
    "save_period": 50,          # 每隔 N 个 epoch 额外保存一次权重（-1 表示不额外保存）

    # ---------- 验证与可视化 ----------
    "val": True,                # 训练过程中进行验证
    "plots": True,              # 保存训练曲线图（loss、mAP 等）

    # ---------- 训练策略 ----------
    "pretrained": False,        # 不使用预训练权重（从零训练）
    "optimizer": "AdamW",       # AdamW 优化器（从零训练时收敛更稳定）
    "lr0": 0.001,               # 初始学习率（AdamW 推荐 1e-3）
    "lrf": 0.01,                # 最终学习率 = lr0 * lrf
    "momentum": 0.937,          # Adam beta1
    "weight_decay": 0.0005,     # 权重衰减（L2 正则化）
    "warmup_epochs": 5.0,       # 预热轮数（从零训练建议更长的预热）
    "cos_lr": True,             # 余弦退火学习率调度（平滑降低学习率，提升收敛质量）
    "patience": 100,            # 早停：连续 N 个 epoch 验证指标不提升则停止
    "close_mosaic": 15,         # 最后 15 个 epoch 关闭 mosaic（让模型更好适应真实图片）

    # ---------- 数据增强配置 ----------
    "hsv_h": 0.015,             # HSV 色调增强
    "hsv_s": 0.7,               # HSV 饱和度增强
    "hsv_v": 0.4,               # HSV 亮度增强
    "degrees": 0.0,             # 旋转角度范围（±度）
    "translate": 0.1,           # 平移比例
    "scale": 0.5,               # 缩放比例
    "fliplr": 0.5,              # 水平翻转概率
    "flipud": 0.0,              # 垂直翻转概率
    "mosaic": 1.0,              # Mosaic 增强概率
    "mixup": 0.0,               # MixUp 增强概率

    # ---------- 其他 ----------
    "amp": True,                # 自动混合精度训练（FP16 加速，5090 原生支持）
    "cache": True,              # 缓存数据集到 RAM（加速训练，内存不够可改为 'disk' 或 False）
    "resume": False,            # 是否从上次中断处恢复训练
    "seed": 0,                  # 随机种子（保证可复现性）
    "verbose": True,            # 输出详细日志
}


def main():
    """主训练函数"""
    # 1. 创建模型（从 yaml 配置文件构建，随机初始化权重）
    model = YOLO(CONFIG["model"])

    # 2. 开始训练
    print("=" * 60)
    print(f"  模型: {CONFIG['model']}")
    print(f"  数据集: {CONFIG['data']}")
    print(f"  训练轮数: {CONFIG['epochs']}")
    print(f"  批次大小: {CONFIG['batch']}")
    print(f"  图片尺寸: {CONFIG['imgsz']}")
    print(f"  输出目录: {CONFIG['project']}/{CONFIG['name']}")
    print("=" * 60)

    results = model.train(**CONFIG)

    # 3. 训练完成后输出结果路径
    print("\n" + "=" * 60)
    print("  训练完成！")
    print(f"  结果保存在: {CONFIG['project']}/{CONFIG['name']}")
    print("  保存内容说明:")
    print("    - weights/best.pt     : 最佳模型权重")
    print("    - weights/last.pt     : 最后一轮权重")
    print("    - results.csv         : 每轮训练指标（loss, mAP等）")
    print("    - results.png         : 训练曲线图")
    print("    - confusion_matrix.png: 混淆矩阵")
    print("    - args.yaml           : 完整训练参数记录")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
