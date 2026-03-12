"""
YOLOv11-OBB + Width-Aware Geometric Loss 训练脚本
基于 CA (Coverage-Aware) 模型，额外启用宽度-法向-角度辅助损失。
使用方法:
    直接运行: python train_yolo11_obb_wag.py
    修改下方 CONFIG 字典中的参数即可自定义训练配置
"""

import os
import sys
import datetime
import re
os.environ["OMP_NUM_THREADS"] = "8"  # 修复 libgomp 警告

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import intersect_dicts

# ========================== 终端日志保存配置 ==========================
# 匹配 ANSI 转义序列（如颜色代码、光标清除命令等）
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
# 匹配类似 "1/150      30.4G     2.124" 这种含有进度更新特征的行
# 以及 "Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 32% ━━━" 这样的验证进度条
TQDM_BAR = re.compile(
    r'(\d+/\d+\s+[\d.]+G\s+[\d.]+)'          # 训练进度条特征
    r'|(━━━━━|━━━╸|━━╸|━╸|━)'                # 进度条专属符号
    r'|(\d+%\s*\|?[█▉▊▋▌▍▎▏]+)'              # 其他形式的进度条 (可选)
    r'|(Class\s+Images\s+Instances.*?:\s*\d+%)' # 验证期特定的类目+百分比前缀
)

class Logger(object):
    """
    带缓冲与过滤的日志类：
    1. 过滤 ANSI 颜色代码与 tqdm 进度条
    2. 初期缓存在内存中，直到 `set_log_file` 被调用才真正写入文件
    """
    def __init__(self, stream):
        self.terminal = stream
        self.log_file = None
        self.buffer = []  # 缓存列表，未指定文件前输出的所有内容

    def set_log_file(self, filepath):
        """设定最终的日志文件，并将之前缓冲的文本一次性写入"""
        self.log_file = open(filepath, 'a', encoding='utf-8')
        if self.buffer:
            self.log_file.write("".join(self.buffer))
            self.log_file.flush()
            self.buffer = []  # 清空缓冲

    def write(self, message):
        # 1. 始终正常打印到终端（不修改终端外观）
        self.terminal.write(message)
        self.terminal.flush()

        # 2. 对文件内容进行剥离和过滤
        # 去除 ANSI 转义
        clean_msg = ANSI_ESCAPE.sub('', message)

        # 核心逻辑：拦截任何被识别为"进度刷新"的片段
        # 很多时候 tqdm 每刷新一下都会带 \r，我们直接丢弃带有 tqdm 痕迹的文本
        if TQDM_BAR.search(clean_msg):
            return

        # 对于剩下的包含 \r 的文本：
        # 如果不是进度条，\r 的出现意味着终端前面的内容被当前内容覆写。
        # 为了避免日志文件中积累 `旧内容\r新内容` 导致排版混乱，我们将其替换为换行，或者提取最后一部分有效内容。
        # 这里最安全的做法是将 \r 替换为空字符串（如果它只是用来刷新单行提示）或使用正则只取 \r 后的最后一段
        if '\r' in clean_msg:
            # 取 \r 分割后的最后一段非空内容（即最终在屏幕上显示的最终态）
            parts = [p for p in clean_msg.split('\r') if p]
            if parts:
                clean_msg = parts[-1]
            else:
                return

        # 空消息跳过
        if not clean_msg.strip('\n'):
            return

        # 写入文件或缓冲
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
CONFIG = {
    # ---------- 模型配置 ----------
    # CA 预训练权重 (reg_max=32 + Coverage-Aware TAL)
    "model": "/root/autodl-tmp/work-dirs/yolo11_obb-ca/weights/best.pt",

    # ---------- 数据集配置 ----------
    "data": "/root/autodl-tmp/dataset/TTPLA-1024/dataset.yaml",

    # ---------- 训练基本参数 ----------
    "epochs": 150,              # 训练总轮数
    "batch": 16,                # 批次大小
    "imgsz": 1024,              # 输入图片尺寸
    "device": 0,                # GPU 设备编号
    "workers": 16,              # 数据加载线程数

    # ---------- 输出目录配置 ----------
    "project": "/root/autodl-tmp/work-dirs",
    "name": "yolo11_obb-ca-wag",        # WAG = Width-Aware Geometric
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
    # Width-Aware Geometric Loss 配置（所有参数均可在此调节）
    "aux_geo": 1.5,       # 全局增益
    "aux_geo_lp": 1.0,    # 法向偏移分量权重 λ_perp
    "aux_geo_lw": 2.0,    # 宽度分量权重 λ_w
    "aux_geo_lt": 0.5,    # 角度分量权重 λ_theta
    "aux_geo_ar": 30.0,   # 门控：长宽比阈值（AR > 此值时启用）
    "aux_geo_ws": 12.0,   # 门控：短边像素阈值（短边 < 此值时启用）

    # ---------- 其他 ----------
    "amp": True,
    "cache": 'disk',
    "resume": False,
    "seed": 0,
    "verbose": True,
}


def main():
    """主训练函数"""
    # --- 拦截控制台输出（暂存在内存） ---
    sys.stdout = Logger(sys.stdout)
    sys.stderr = Logger(sys.stderr)
    print("\n[*] 终端日志拦截器已启动（将在验证最终目录后落地存盘，并已开启进度条剥离保护）\n")
    # ----------------------------------

    # 1. 创建模型
    model = YOLO(CONFIG["model"])

    # 2. 开始训练
    print("=" * 60)
    print(f"  模型: {CONFIG['model']}")
    print(f"  数据集: {CONFIG['data']}")
    print(f"  训练轮数: {CONFIG['epochs']}")
    print(f"  批次大小: {CONFIG['batch']}")
    print(f"  图片尺寸: {CONFIG['imgsz']}")
    print(f"  输出目录: {CONFIG['project']}/{CONFIG['name']}")
    print(f"  辅助损失增益 (aux_geo): {CONFIG['aux_geo']}")
    print("=" * 60)

    # 注册回调函数：在 Trainer 确定了 save_dir 后，将暂存的日志吐出到由系统建立的目录中
    def on_train_start(trainer):
        # 此时已经存在类似 work-dirs/yolo11_obb 等最终准确确定的训练目录
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_log_file = os.path.join(trainer.save_dir, f"train_terminal_{time_str}.log")
        print(f"\n[*] YOLO 目录已确定，开始将暂存日志刷入文件: {final_log_file}\n")
        # 为两个流设置真实的文件句柄
        sys.stdout.set_log_file(final_log_file)
        sys.stderr.set_log_file(final_log_file)

    model.add_callback("on_train_start", on_train_start)

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
