"""
YOLOv11-OBB 自定义数据集训练脚本
使用方法:
    直接运行: python train_yolo11_obb.py
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

        # 核心逻辑：拦截任何被识别为“进度刷新”的片段
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
    # 使用 .yaml 文件从零开始训练（不使用预训练权重）
    # 可选规模: yolo11n-obb.yaml / yolo11s-obb.yaml / yolo11m-obb.yaml / yolo11l-obb.yaml / yolo11x-obb.yaml
    # "model": "yolo11l-obb-ca.yaml",
    # "model": "/root/autodl-tmp/work-dirs/yolo11_obb2/weights/last.pt",
    "model": "/root/autodl-tmp/work-dirs/yolo11_obb-ca/weights/last.pt",

    # ---------- 数据集配置 ----------
    # 指向你的自定义数据集 yaml 文件
    "data": "/root/autodl-tmp/dataset/TTPLA-1024/dataset.yaml",

    # ---------- 训练基本参数 ----------
    "epochs": 50,              # 训练总轮数
    "batch": 16,                # 批次大小（RTX 5090 32GB 显存，yolo11l + 1024 尝试 32，OOM 则降回 16）
    "imgsz": 1024,              # 输入图片尺寸
    "device": 0,                # GPU 设备编号
    "workers": 16,               # 数据加载线程数（5090 性能强，可用更多线程）

    # ---------- 输出目录配置 ----------
    "project": "/root/autodl-tmp/work-dirs",          # 输出主目录
    "name": "yolo11_obb-ca-continue",        # 实验名称（子目录）
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
    "lr0": 0.0001,               # 初始学习率（AdamW 推荐 1e-3）
    "lrf": 0.01,                # 最终学习率 = lr0 * lrf
    "momentum": 0.937,          # Adam beta1
    "weight_decay": 0.0005,     # 权重衰减（L2 正则化）
    "warmup_epochs": 10.0,       # 预热轮数（从零训练建议更长的预热）
    "cos_lr": True,             # 余弦退火学习率调度（平滑降低学习率，提升收敛质量）
    # "patience":15,            # 早停：连续 N 个 epoch 验证指标不提升则停止
    "close_mosaic": 10,         # 最后 15 个 epoch 关闭 mosaic（让模型更好适应真实图片）

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
    "amp": True,               # 排雷期关闭混合精度，使用 Float32 排除数值不稳定（验证通过后可改回 True）
    "cache": 'disk',              # 缓存数据集到 RAM（加速训练，内存不够可改为 'disk' 或 False）
    "resume": False,            # 是否从上次中断处恢复训练
    "seed": 0,                  # 随机种子（保证可复现性）
    "verbose": True,            # 输出详细日志

#    "conf": 0.05,              # 验证阈值，解决 NMS 卡死
#    "max_det": 100,            # 最大检测数，第二道保险


}


def main():
    """主训练函数"""
    # --- 拦截控制台输出（暂存在内存） ---
    sys.stdout = Logger(sys.stdout)
    sys.stderr = Logger(sys.stderr)
    print("\n[*] 终端日志拦截器已启动（将在验证最终目录后落地存盘，并已开启进度条剥离保护）\n")
    # ----------------------------------

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
