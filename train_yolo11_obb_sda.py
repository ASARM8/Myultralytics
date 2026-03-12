"""
YOLOv11-OBB-SDA v2 训练脚本（Scale-Decoupled Asymmetric Head）

分阶段冻结训练策略：
  Stage 1 (epoch 0~9):   仅训练新增模块（p3_reduce, fuse, alpha gates）
  Stage 2 (epoch 10~49):  解冻回归头(cv2) + 角度头(cv4)
  Stage 3 (epoch 50~199): 全模型小学习率微调

权重迁移：
  - backbone / FPN / cv2 / cv3 / cv4 / dfl → 从 CA best.pt 完美继承
  - p3_reduce / fuse_p4 / fuse_p5 → 随机初始化
  - alpha gates → 初始化为 0（训练开始时等价于原始 CA 模型）

使用方法:
    直接运行: python train_yolo11_obb_sda.py
"""

import os
import sys
import datetime
import re
os.environ["OMP_NUM_THREADS"] = "8"

import torch
import numpy as np
from collections import defaultdict
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
    "patience": 0,              # 0 表示完全关闭早停，强制跑完 epochs
    "batch": 16,
    "imgsz": 1024,
    "device": 0,
    "workers": 16,

    # ---------- 输出目录配置 ----------
    "project": "/root/autodl-tmp/work-dirs",
    "name": "check_yolo11_obb-sda-v2",
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
    "lr0": 0.0001,              # 保守全局 LR，配合分阶段冻结策略使用
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
    "amp": False,
    "cache": 'disk',
    "resume": False,
    "seed": 0,
    "verbose": True,
}


def load_pretrained_weights(nn_model, pretrain_path, device=None):
    """从 CA 预训练模型加载权重到 SDA v2 模型的 nn.Module。

    AsymmetricOBB v2 新增的层：
      - p3_reduce: 1x1 Conv，随机初始化
      - unshuffle_p4 / unshuffle_p5: PixelUnshuffle，无可学习参数
      - fuse_p4 / fuse_p5: 1x1 Conv，随机初始化
      - alpha_box_p4/p5, alpha_ang_p4/p5: 标量 gate，初始化为 0

    其余层（backbone, FPN, cv2, cv3, cv4, dfl）均从 CA 模型完美迁移。

    Args:
        nn_model (torch.nn.Module): 目标模型的 nn.Module（trainer.model）。
        pretrain_path (str): CA 预训练权重路径。
        device: 模型所在设备，用于加载权重到正确设备。
    """
    print(f"\n[*] 加载预训练权重: {pretrain_path}")

    map_location = device if device else "cpu"
    ckpt = torch.load(pretrain_path, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        pretrain_sd = ckpt["model"].float().state_dict()
    else:
        pretrain_sd = ckpt

    model_sd = nn_model.state_dict()

    # 使用 intersect_dicts 匹配形状兼容的权重
    matched = intersect_dicts(pretrain_sd, model_sd, exclude=[])
    n_total = len(model_sd)
    n_matched = len(matched)

    nn_model.load_state_dict(matched, strict=False)

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


# ========================== 分阶段冻结配置 ==========================
# Stage 1: 仅训练新增 SDA 模块，让新模块先学会"不添乱"
# Stage 2: 解冻回归头(cv2) + 角度头(cv4)，让它们适应注入后的特征
# Stage 3: 全模型小学习率微调
STAGE_CONFIG = {
    "stage1_end": 10,    # Stage 1 持续到第 10 个 epoch
    "stage2_end": 50,    # Stage 2 持续到第 50 个 epoch
    # Stage 3: epoch 50 ~ 结束
}

# SDA 新增模块的关键字（用于识别需要在 Stage 1 解冻的参数）
SDA_NEW_KEYWORDS = ("p3_reduce", "fuse_p4", "fuse_p5", "alpha_box", "alpha_ang",
                    "unshuffle")  # unshuffle 无参数，但保险起见也列出

# 诊断 2：需要监控梯度的模块关键字
GRAD_MONITOR_KEYWORDS = ("p3_reduce", "fuse_p4", "fuse_p5",
                         "alpha_box_p4", "alpha_box_p5",
                         "alpha_ang_p4", "alpha_ang_p5")

# 梯度范数累积器（闭包共享）
# 每个参数存储: {"norms": [...], "nan_count": int, "inf_count": int, "total": int}
_grad_norms = {}


def get_sda_freeze_list(nn_model, stage):
    """根据训练阶段返回需要冻结的参数名列表。

    Args:
        nn_model (torch.nn.Module): 模型的 nn.Module（trainer.model）。
        stage (int): 当前训练阶段 (1, 2, 3)。

    Returns:
        list[str]: 需要冻结的参数名列表。
    """
    freeze_names = []

    for name, param in nn_model.named_parameters():
        is_sda_new = any(kw in name for kw in SDA_NEW_KEYWORDS)
        is_cv2 = ".cv2." in name
        is_cv4 = ".cv4." in name

        if stage == 1:
            # Stage 1: 仅训练 SDA 新模块，冻结其他所有
            if not is_sda_new:
                freeze_names.append(name)
        elif stage == 2:
            # Stage 2: SDA 新模块 + cv2 + cv4 可训练，冻结其余
            if not is_sda_new and not is_cv2 and not is_cv4:
                freeze_names.append(name)
        # stage == 3: 全部解冻，返回空列表

    return freeze_names


def apply_freeze(nn_model, freeze_names, stage_label):
    """应用冻结策略：冻结指定参数，解冻其余。"""
    n_frozen = 0
    n_total = 0
    for name, param in nn_model.named_parameters():
        n_total += 1
        if name in freeze_names:
            param.requires_grad = False
            n_frozen += 1
        else:
            param.requires_grad = True

    print(f"\n[*] {stage_label}: 冻结 {n_frozen}/{n_total} 参数组, "
          f"可训练 {n_total - n_frozen}/{n_total}")


def main():
    """主训练函数：分阶段冻结训练"""
    sys.stdout = Logger(sys.stdout)
    sys.stderr = Logger(sys.stderr)
    print("\n[*] 终端日志拦截器已启动\n")

    # 1. 从 YAML 构建 SDA 模型架构
    print("=" * 60)
    print("  YOLOv11-OBB-SDA v2 训练（分阶段冻结策略）")
    print("=" * 60)
    print(f"  模型架构: {CONFIG['model']}")
    print(f"  预训练权重: {PRETRAIN_WEIGHTS}")
    print(f"  数据集: {CONFIG['data']}")
    print(f"  训练轮数: {CONFIG['epochs']}")
    print(f"  批次大小: {CONFIG['batch']}")
    print(f"  图片尺寸: {CONFIG['imgsz']}")
    print(f"  输出目录: {CONFIG['project']}/{CONFIG['name']}")
    print(f"  Stage 1 (epoch 0~{STAGE_CONFIG['stage1_end']-1}): 仅训练 SDA 新模块")
    print(f"  Stage 2 (epoch {STAGE_CONFIG['stage1_end']}~{STAGE_CONFIG['stage2_end']-1}): + cv2 + cv4")
    print(f"  Stage 3 (epoch {STAGE_CONFIG['stage2_end']}~{CONFIG['epochs']-1}): 全模型微调")
    print("=" * 60)

    model = YOLO(CONFIG["model"])

    # 2. 注意：不能在 model.train() 之前加载权重或冻结参数！
    # 原因：YOLO.train() 内部会通过 get_model() 重新构建 nn.Module（见 engine/model.py L774）：
    #   self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, ...)
    # 当从 YAML 构建时 self.ckpt=None → weights=None → 全新随机模型，手动加载的权重被丢弃！
    # 解决方案：在 on_pretrain_routine_end 回调中加载权重（trainer 已完成模型构建后）。
    current_stage = [1]  # 用列表包装以便闭包修改

    def on_train_start(trainer):
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        final_log_file = os.path.join(trainer.save_dir, f"train_terminal_{time_str}.log")
        print(f"\n[*] 日志文件: {final_log_file}\n")
        sys.stdout.set_log_file(final_log_file)
        sys.stderr.set_log_file(final_log_file)

    def on_pretrain_routine_end(trainer):
        """在 trainer._setup_train() 完成后、训练循环开始前：
        1. 加载 CA 预训练权重到 trainer.model
        2. 同步更新 EMA 模型
        3. 应用 Stage 1 冻结
        4. 注册梯度监控 hooks（诊断 2）
        """
        # (a) 加载 CA 预训练权重到 trainer 的实际模型
        load_pretrained_weights(trainer.model, PRETRAIN_WEIGHTS, device=trainer.device)

        # (b) 同步更新 EMA 模型（EMA 在 _setup_train 中已从随机模型初始化，需要覆盖）
        if hasattr(trainer, 'ema') and trainer.ema is not None:
            trainer.ema.ema.load_state_dict(trainer.model.state_dict())
            trainer.ema.updates = 0  # 重置 EMA 更新计数
            print("[*] EMA 模型已同步更新为 CA 预训练权重")

        # (c) 应用 Stage 1 冻结
        stage1_freeze = get_sda_freeze_list(trainer.model, stage=1)
        apply_freeze(trainer.model, stage1_freeze, "Stage 1")

        # (d) 诊断 2：注册梯度监控 hooks
        _register_grad_hooks(trainer.model)

    def on_train_epoch_start(trainer):
        """在每个 epoch 开始时检查是否需要切换训练阶段。"""
        epoch = trainer.epoch  # 当前 epoch（0-indexed）

        if epoch == STAGE_CONFIG["stage1_end"] and current_stage[0] == 1:
            # 切换到 Stage 2
            current_stage[0] = 2
            stage2_freeze = get_sda_freeze_list(trainer.model, stage=2)
            apply_freeze(trainer.model, stage2_freeze, "Stage 2")
            # 打印 gate 值监控
            _print_gate_values(trainer.model)

        elif epoch == STAGE_CONFIG["stage2_end"] and current_stage[0] == 2:
            # 切换到 Stage 3
            current_stage[0] = 3
            apply_freeze(trainer.model, [], "Stage 3 (全部解冻)")
            _print_gate_values(trainer.model)

    def on_train_epoch_end(trainer):
        """每个 epoch 结束时打印 gate 值和梯度统计。"""
        epoch = trainer.epoch
        # 每 10 个 epoch 打印一次 gate 值
        if (epoch + 1) % 10 == 0:
            _print_gate_values(trainer.model)
        # 每个 epoch 打印梯度范数统计（诊断 2）
        _report_grad_norms(epoch)

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    # 4. 开始训练（权重加载和冻结都在回调中完成）
    results = model.train(**CONFIG)

    # 6. 完成
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

    _print_gate_values(model.model)
    return results


def _print_gate_values(nn_model):
    """打印当前 gate 参数值，监控注入强度。"""
    for name, param in nn_model.named_parameters():
        if "alpha_" in name:
            print(f"  [gate] {name} = {param.data.item():.6f}")


def _register_grad_hooks(nn_model):
    """诊断 2：为 SDA 关键模块的参数注册梯度 hooks，累积梯度范数。

    注意：
    - param.register_hook 在 backward() 期间触发，此时梯度尚未被 zero_grad() 清除
    - AMP 训练中梯度是 scaled 的，fp16 溢出会导致 inf/nan
    - 此处先统计 nan/inf 比例，再用 nan_to_num 过滤后计算有效 norm
    """
    _grad_norms.clear()
    hook_count = 0
    for name, param in nn_model.named_parameters():
        if any(kw in name for kw in GRAD_MONITOR_KEYWORDS):
            def make_hook(param_name):
                def hook_fn(grad):
                    if param_name not in _grad_norms:
                        _grad_norms[param_name] = {"norms": [], "nan_count": 0, "inf_count": 0, "total": 0}
                    entry = _grad_norms[param_name]
                    entry["total"] += 1
                    has_nan = torch.isnan(grad).any().item()
                    has_inf = torch.isinf(grad).any().item()
                    if has_nan:
                        entry["nan_count"] += 1
                    if has_inf:
                        entry["inf_count"] += 1
                    # 过滤 nan/inf 后计算有效 norm
                    clean_grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                    entry["norms"].append(clean_grad.norm().item())
                return hook_fn
            param.register_hook(make_hook(name))
            hook_count += 1
    print(f"[*] 诊断 2：已为 {hook_count} 个 SDA 参数注册梯度监控 hooks")


def _report_grad_norms(epoch):
    """诊断 2：汇总并打印当前 epoch 的 SDA 模块梯度范数统计。"""
    if not _grad_norms:
        return

    print(f"\n  [诊断2] Epoch {epoch} SDA 模块梯度范数 (nan/inf 已过滤后统计):")
    print(f"  {'参数名':>45s}  {'batches':>7s}  {'nan%':>6s}  {'inf%':>6s}  "
          f"{'mean':>10s}  {'max':>10s}  {'min':>10s}  {'状态':>8s}")
    print(f"  {'-'*45}  {'-'*7}  {'-'*6}  {'-'*6}  "
          f"{'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")

    for name in sorted(_grad_norms.keys()):
        entry = _grad_norms[name]
        norms = entry["norms"]
        total = entry["total"]
        if total == 0:
            continue

        nan_pct = entry["nan_count"] / total * 100
        inf_pct = entry["inf_count"] / total * 100
        arr = np.array(norms)
        mean_val = arr.mean()
        max_val = arr.max()
        min_val = arr.min()

        # 判断梯度状态（优先检测 nan/inf）
        if nan_pct > 50:
            status = "⚠️ 全nan"
        elif nan_pct > 10:
            status = "⚠️ 多nan"
        elif inf_pct > 10:
            status = "⚠️ 多inf"
        elif mean_val < 1e-8:
            status = "⚠️ 零"
        elif mean_val < 1e-5:
            status = "极小"
        elif mean_val > 100:
            status = "⚠️ 爆"
        else:
            status = "正常"

        short_name = name.replace("model.", "") if len(name) > 45 else name
        print(f"  {short_name:>45s}  {total:7d}  {nan_pct:5.1f}%  {inf_pct:5.1f}%  "
              f"{mean_val:10.2e}  {max_val:10.2e}  {min_val:10.2e}  {status:>8s}")

    # 诊断建议
    any_nan = any(e["nan_count"] > 0 for e in _grad_norms.values())
    any_all_nan = any(e["nan_count"] == e["total"] for e in _grad_norms.values() if e["total"] > 0)
    if any_all_nan:
        print(f"  [!] 所有 batch 梯度均含 nan → 可能原因:")
        print(f"      1. AMP fp16 溢出（随机初始化的 SDA 模块输出值域与预训练特征不匹配）")
        print(f"      2. loss 本身为 nan（检查训练日志中的 loss 值）")
        print(f"      3. 尝试: 关闭 AMP (amp=False) 或降低 lr0")
    elif any_nan:
        print(f"  [!] 部分 batch 梯度含 nan → AMP 偶发溢出，通常无害（scaler 会自动跳过）")

    # 清空累积器，为下一个 epoch 准备
    _grad_norms.clear()


if __name__ == "__main__":
    main()
