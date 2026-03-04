"""
YOLOv11-OBB + StripPooling（条状池化）训练脚本
使用方法:
    直接运行: python train_yolo11_obb_sp.py
    修改下方 CONFIG 字典中的参数即可自定义训练配置

说明:
    本脚本基于 yolo11-obb-sp.yaml 架构（在 SPPF 后插入 StripPooling 注意力模块），
    支持两种训练模式：
    1. 从零训练：直接使用 YAML 配置
    2. 迁移训练（推荐）：先用 YAML 构建新架构，再加载已有权重（如 best.pt），
       已有层自动继承权重，StripPooling 层保持零初始化（Sigmoid(0)=0.5，平稳过渡）
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
    # 使用带条状池化的 YAML 配置文件
    # 可选规模: yolo11n-obb-sp.yaml / yolo11s-obb-sp.yaml / yolo11m-obb-sp.yaml / yolo11l-obb-sp.yaml / yolo11x-obb-sp.yaml
    "model": "yolo11l-obb-sp.yaml",

    # ---------- 迁移学习配置 ----------
    # 指定已有权重路径，用于迁移训练（留空或注释掉则从零训练）
    # StripPooling 层在已有权重中不存在，会自动随机初始化（零初始化保证平稳过渡）
    "pretrained_weights": "/root/autodl-tmp/work-dirs/yolo11_obb-v2.1full/weights/best.pt",

    # ---------- 数据集配置 ----------
    # 指向你的自定义数据集 yaml 文件
    "data": "/root/autodl-tmp/dataset/TTPLA-1024/dataset.yaml",

    # ---------- 训练基本参数 ----------
    "epochs": 150,              # 迁移训练建议 100~150 轮（主干已成熟，不需要 300 轮）
    "batch": 20,                # 批次大小（RTX 5090 32GB 显存，新增 StripPooling 参数量极小，不影响显存）
    "imgsz": 1024,              # 输入图片尺寸
    "device": 0,                # GPU 设备编号
    "workers": 16,              # 数据加载线程数

    # ---------- 输出目录配置 ----------
    "project": "/root/autodl-tmp/work-dirs",          # 输出主目录
    "name": "yolo11_obb_sp",    # 实验名称（子目录）—— 区分于原版
    "exist_ok": False,                      # 是否允许覆盖已有同名目录

    # ---------- 模型保存配置 ----------
    "save": True,               # 保存训练检查点（best.pt 和 last.pt）
    "save_period": 50,          # 每隔 N 个 epoch 额外保存一次权重（-1 表示不额外保存）

    # ---------- 验证与可视化 ----------
    "val": True,                # 训练过程中进行验证
    "plots": True,              # 保存训练曲线图（loss、mAP 等）

    # ---------- 训练策略 ----------
    "pretrained": False,        # 不使用 COCO 预训练权重（我们用自己的权重迁移）
    "optimizer": "AdamW",       # AdamW 优化器
    "lr0": 0.001,               # 初始学习率（迁移训练 1e-3 ~ 1e-4 均可）
    "lrf": 0.01,                # 最终学习率 = lr0 * lrf
    "momentum": 0.937,          # Adam beta1
    "weight_decay": 0.0005,     # 权重衰减（L2 正则化）
    "warmup_epochs": 5.0,       # 预热轮数（迁移训练必须开启预热，让新模块慢慢苏醒）
    "cos_lr": True,             # 余弦退火学习率调度
    # "patience": 15,           # 早停：连续 N 个 epoch 验证指标不提升则停止
    "close_mosaic": 10,         # 最后 10 个 epoch 关闭 mosaic

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
    "amp": True,                # 混合精度训练
    "cache": 'disk',            # 缓存数据集到磁盘（加速训练）
    "resume": False,            # 是否从上次中断处恢复训练
    "seed": 0,                  # 随机种子（保证可复现性）
    "verbose": True,            # 输出详细日志
}


def main():
    """主训练函数"""
    # --- 拦截控制台输出（暂存在内存） ---
    sys.stdout = Logger(sys.stdout)
    sys.stderr = Logger(sys.stderr)
    print("\n[*] 终端日志拦截器已启动（将在验证最终目录后落地存盘，并已开启进度条剥离保护）\n")
    # ----------------------------------

    # 1. 用新 YAML 构建带 StripPooling 的模型架构
    model = YOLO(CONFIG["model"])

    # 2. 如果指定了已有权重，重映射层索引后加载
    #    原因：插入 StripPooling 在第 10 层，导致后续所有层索引 +1，
    #    直接用 model.load() 会因 key 名不匹配而丢失 60% 的权重。
    #    解决：手动将旧权重中 index >= 10 的层重映射为 index + 1。
    pretrained_weights = CONFIG.pop("pretrained_weights", None)
    if pretrained_weights and os.path.exists(pretrained_weights):
        # 加载旧权重的 state_dict
        ckpt = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
        old_model = (ckpt.get("ema") or ckpt["model"]).float()
        old_sd = old_model.state_dict()

        # 重映射层索引：旧模型 index >= 10 的层在新模型中对应 index + 1
        INSERT_IDX = 10  # StripPooling 插入位置
        remapped_sd = {}
        for k, v in old_sd.items():
            parts = k.split(".")
            if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
                layer_idx = int(parts[1])
                if layer_idx >= INSERT_IDX:
                    parts[1] = str(layer_idx + 1)
            remapped_sd[".".join(parts)] = v

        # 与新模型做 key+shape 交集匹配，然后加载
        new_sd = model.model.state_dict()
        matched_sd = intersect_dicts(remapped_sd, new_sd)
        model.model.load_state_dict(matched_sd, strict=False)
        n_matched = len(matched_sd)
        n_total = len(new_sd)
        print(f"  已加载迁移权重: {pretrained_weights}")
        print(f"  层索引重映射后匹配: {n_matched}/{n_total} 项（StripPooling 层保持零初始化）")
    else:
        print(f"  未加载迁移权重，从零开始训练")

    # 3. 开始训练
    print("=" * 60)
    print(f"  模型: {CONFIG['model']} (含 StripPooling 条状池化)")
    print(f"  数据集: {CONFIG['data']}")
    print(f"  训练轮数: {CONFIG['epochs']}")
    print(f"  批次大小: {CONFIG['batch']}")
    print(f"  图片尺寸: {CONFIG['imgsz']}")
    print(f"  输出目录: {CONFIG['project']}/{CONFIG['name']}")
    print("=" * 60)

    # 注册回调函数：在 Trainer 确定了 save_dir 后，将暂存的日志吐出到由系统建立的目录中
    def on_train_start(trainer):
        # 此时已经存在类似 work-dirs/yolo11_obb_sp 等最终准确确定的训练目录
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_log_file = os.path.join(trainer.save_dir, f"train_terminal_{time_str}.log")
        print(f"\n[*] YOLO 目录已确定，开始将暂存日志刷入文件: {final_log_file}\n")
        # 为两个流设置真实的文件句柄
        sys.stdout.set_log_file(final_log_file)
        sys.stderr.set_log_file(final_log_file)

    model.add_callback("on_train_start", on_train_start)

    results = model.train(**CONFIG)

    # 4. 训练完成后输出结果路径
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
