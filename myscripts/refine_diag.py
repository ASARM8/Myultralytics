"""
Refine Head v1 诊断脚本
验证三个假设：
1. cv5 输出 Δw/Δh/Δθ 是否接近零（Refine Head 未学到有效修正）
2. cv5 梯度量级是否远小于 cv2（梯度被 DFL 头吸收）
3. DFL 粗框在薄目标上是否有残差（cv5 是否有修正空间）

使用方法：
    python myscripts/refine_diag.py
"""

import math
import torch
import numpy as np
from collections import defaultdict

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_KEYS
from ultralytics.utils.tal import dist2rbox, make_anchors
from ultralytics.utils.metrics import probiou


# ========================== 配置 ==========================
# v1 训练的 checkpoint 路径（含 OBBRefine 头）
MODEL_PATH = "/root/autodl-tmp/work-dirs/yolo11_obb-ca-refine/weights/best.pt"
DATA_PATH = "/root/autodl-tmp/dataset/TTPLA-1024/dataset.yaml"
NUM_BATCHES = 100  # 诊断用的 batch 数
DEVICE = 0


def collect_diagnostics():
    """收集诊断数据"""
    model = YOLO(MODEL_PATH)
    model.model.float()
    for p in model.model.parameters():
        p.requires_grad = True
    model.model.to(f"cuda:{DEVICE}")
    model.model.train()

    # 规范化 model.args：checkpoint 中常为 dict，而 loss/criterion 期望可属性访问的配置对象
    train_args = getattr(model.model, "args", {})
    model_args_overrides = dict(train_args) if isinstance(train_args, dict) else vars(train_args)
    model_args_overrides = {k: v for k, v in model_args_overrides.items() if k in DEFAULT_CFG_KEYS}
    model_args_overrides["task"] = "obb"
    model.model.args = get_cfg(DEFAULT_CFG, overrides=model_args_overrides)

    # 获取检测头（OBBRefine）
    detect_head = model.model.model[-1]
    print(f"检测头类型: {type(detect_head).__name__}")

    # 确认 cv5 存在
    if not hasattr(detect_head, 'cv5') or detect_head.cv5 is None:
        print("错误: 模型没有 cv5 分支！")
        return
    print(f"cv5 层数: {len(detect_head.cv5)}")

    # ---- 假设 1：cv5 参数和输出统计 ----
    print("\n" + "=" * 60)
    print("假设 1：cv5 参数统计（权重是否仍为零初始化附近）")
    print("=" * 60)
    for i, seq in enumerate(detect_head.cv5):
        last_conv = seq[-1]  # 最后一层 Conv2d
        w = last_conv.weight
        b = last_conv.bias
        print(f"  cv5[{i}] last Conv2d:")
        print(f"    weight: mean={w.mean().item():.6f}, std={w.std().item():.6f}, "
              f"abs_max={w.abs().max().item():.6f}, norm={w.norm().item():.6f}")
        print(f"    bias:   mean={b.mean().item():.6f}, std={b.std().item():.6f}, "
              f"abs_max={b.abs().max().item():.6f}")

    # 构建 dataloader
    overrides = {
        "data": DATA_PATH, "batch": 8, "imgsz": 1024,
        "device": DEVICE, "workers": 4, "cache": "disk", "task": "obb",
    }
    args = get_cfg(DEFAULT_CFG, overrides=overrides)

    # 直接用 train loader
    from ultralytics.data.utils import check_det_dataset
    data_dict = check_det_dataset(DATA_PATH)

    from ultralytics.data import build_dataloader, build_yolo_dataset
    dataset = build_yolo_dataset(
        args, data_dict["train"], batch=8, data=data_dict, mode="train", rect=False
    )
    loader = build_dataloader(dataset, batch=8, workers=4, rank=-1)

    # 收集统计
    stats = defaultdict(list)
    criterion = model.model.criterion
    if criterion is None:
        criterion = model.model.init_criterion()
        model.model.criterion = criterion

    # 启动前自检：确认 loss 所需超参已就位，避免中途因 dict/缺字段报错
    required_hyp_keys = ["box", "cls", "dfl", "angle", "aux_geo", "aux_geo_ar", "aux_geo_ws"]
    missing_hyp_keys = [k for k in required_hyp_keys if not hasattr(criterion.hyp, k)]
    if missing_hyp_keys:
        raise RuntimeError(f"criterion.hyp 缺少必要字段: {missing_hyp_keys}")
    print(
        "[*] criterion 已初始化: "
        f"aux_geo={criterion.hyp.aux_geo}, "
        f"aux_geo_ar={criterion.hyp.aux_geo_ar}, "
        f"aux_geo_ws={criterion.hyp.aux_geo_ws}"
    )

    print(f"\n开始跑 {NUM_BATCHES} 个 batch 收集诊断数据...")

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= NUM_BATCHES:
            break

        # 移到 GPU
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(f"cuda:{DEVICE}")
        imgs = batch["img"].float() / 255.0

        # forward
        model.model.zero_grad()
        preds = model.model(imgs)

        # 提取 refine 输出
        if isinstance(preds, dict):
            train_preds = preds
        elif isinstance(preds, tuple):
            train_preds = preds[1] if len(preds) > 1 else preds[0]
        else:
            train_preds = preds

        # 从 train_preds 中提取 refine
        if isinstance(train_preds, dict) and "refine" in train_preds:
            refine = train_preds["refine"]  # (B, 3, H*W)
            dw = refine[:, 0, :]
            dh = refine[:, 1, :]
            dt = refine[:, 2, :] if refine.shape[1] >= 3 else None

            stats["dw_mean"].append(dw.mean().item())
            stats["dw_std"].append(dw.std().item())
            stats["dw_absmax"].append(dw.abs().max().item())
            stats["dh_mean"].append(dh.mean().item())
            stats["dh_std"].append(dh.std().item())
            stats["dh_absmax"].append(dh.abs().max().item())
            if dt is not None:
                stats["dt_mean"].append(dt.mean().item())
                stats["dt_std"].append(dt.std().item())
                stats["dt_absmax"].append(dt.abs().max().item())
        else:
            print(f"  batch {batch_idx}: 未找到 refine 输出！preds 类型={type(train_preds)}")
            if isinstance(train_preds, dict):
                print(f"    keys={list(train_preds.keys())}")
            if batch_idx == 0:
                print("  跳过后续 batch")
                break
            continue

        # backward 计算梯度
        loss_items, _ = criterion(train_preds, batch)
        total_loss = loss_items.sum()
        total_loss.backward()

        # 假设 2：梯度量级对比
        cv2_grad_norm = 0.0
        cv2_param_count = 0
        cv5_grad_norm = 0.0
        cv5_param_count = 0

        for name, param in detect_head.named_parameters():
            if param.grad is None:
                continue
            g_norm = param.grad.norm().item()
            if "cv2" in name:
                cv2_grad_norm += param.grad.norm().item() ** 2
                cv2_param_count += param.numel()
            elif "cv5" in name:
                cv5_grad_norm += param.grad.norm().item() ** 2
                cv5_param_count += param.numel()

        cv2_grad_norm = cv2_grad_norm ** 0.5
        cv5_grad_norm = cv5_grad_norm ** 0.5
        stats["cv2_grad_norm"].append(cv2_grad_norm)
        stats["cv5_grad_norm"].append(cv5_grad_norm)
        if cv5_grad_norm > 0:
            stats["grad_ratio_cv2_cv5"].append(cv2_grad_norm / cv5_grad_norm)

        # 假设 3：DFL 残差分析
        # 重新 forward 但这次分别看 coarse vs refined
        with torch.no_grad():
            preds2 = model.model(imgs)
            if isinstance(preds2, tuple):
                train_preds2 = preds2[1] if len(preds2) > 1 else preds2[0]
            else:
                train_preds2 = preds2

            if isinstance(train_preds2, dict):
                pred_distri = train_preds2["boxes"]  # (B, 4*reg_max, H*W)
                pred_angle = train_preds2["angle"]  # (B, 1, H*W)
                pred_refine = train_preds2.get("refine")
                feats = train_preds2["feats"]

                stride = detect_head.stride
                anchor_points, stride_tensor = make_anchors(feats, stride, 0.5)

                # 解码粗框（无 refine）
                pd = pred_distri.permute(0, 2, 1).contiguous()
                pa = pred_angle.permute(0, 2, 1).contiguous()
                coarse_bboxes = criterion.bbox_decode(anchor_points, pd, pa)

                # 解码精修框
                if pred_refine is not None:
                    rf = pred_refine.permute(0, 2, 1)
                    tau = 5 * math.pi / 180
                    clamp_val = 1.0
                    dw_val = rf[..., 0:1].clamp(-clamp_val, clamp_val)
                    dh_val = rf[..., 1:2].clamp(-clamp_val, clamp_val)
                    refined_bboxes = torch.cat([
                        coarse_bboxes[..., 0:2],
                        coarse_bboxes[..., 2:3] * torch.exp(dw_val),
                        coarse_bboxes[..., 3:4] * torch.exp(dh_val),
                        coarse_bboxes[..., 4:5] + tau * torch.tanh(rf[..., 2:3])
                        if rf.shape[-1] >= 3 else coarse_bboxes[..., 4:5],
                    ], dim=-1)

                    # 对比 coarse vs refined 的差异
                    w_diff = (refined_bboxes[..., 2] - coarse_bboxes[..., 2]).abs()
                    h_diff = (refined_bboxes[..., 3] - coarse_bboxes[..., 3]).abs()
                    a_diff = (refined_bboxes[..., 4] - coarse_bboxes[..., 4]).abs()
                    stats["refine_w_diff_mean"].append(w_diff.mean().item())
                    stats["refine_h_diff_mean"].append(h_diff.mean().item())
                    stats["refine_a_diff_mean"].append(a_diff.mean().item())

        if (batch_idx + 1) % 20 == 0:
            print(f"  已处理 {batch_idx + 1}/{NUM_BATCHES} batches")

    # ========================== 输出报告 ==========================
    report_lines = []
    
    def log_and_print(text):
        print(text)
        report_lines.append(text)

    log_and_print("\n" + "=" * 60)
    log_and_print("假设 1：cv5 输出（Δw/Δh/Δθ）是否接近零？")
    log_and_print("=" * 60)
    for key in ["dw", "dh", "dt"]:
        if f"{key}_mean" in stats:
            means = np.array(stats[f"{key}_mean"])
            stds = np.array(stats[f"{key}_std"])
            maxs = np.array(stats[f"{key}_absmax"])
            label = {"dw": "Δw", "dh": "Δh", "dt": "Δθ"}[key]
            log_and_print(f"  {label}: mean={means.mean():.6f}±{means.std():.6f}, "
                  f"std={stds.mean():.6f}, abs_max={maxs.mean():.6f} (max={maxs.max():.6f})")

    log_and_print(f"\n  判定：")
    dw_max = np.array(stats.get("dw_absmax", [0])).max()
    if dw_max < 0.01:
        log_and_print("✅ 确认 — cv5 输出接近零，Refine Head 未学到有效修正")
    else:
        log_and_print(f"❌ 否定 — cv5 输出非零 (abs_max={dw_max:.4f})，问题可能在别处")

    log_and_print("\n" + "=" * 60)
    log_and_print("假设 2：cv5 梯度是否远小于 cv2？")
    log_and_print("=" * 60)
    cv2_norms = np.array(stats.get("cv2_grad_norm", [0]))
    cv5_norms = np.array(stats.get("cv5_grad_norm", [0]))
    ratios = np.array(stats.get("grad_ratio_cv2_cv5", [0]))
    log_and_print(f"  cv2 梯度 L2 norm: mean={cv2_norms.mean():.6f}")
    log_and_print(f"  cv5 梯度 L2 norm: mean={cv5_norms.mean():.6f}")
    log_and_print(f"  比值 cv2/cv5:     mean={ratios.mean():.2f}x")

    log_and_print(f"\n  判定：")
    if ratios.mean() > 10:
        log_and_print(f"✅ 确认 — cv2 梯度是 cv5 的 {ratios.mean():.1f}x，梯度竞争严重")
    elif ratios.mean() > 3:
        log_and_print(f"⚠️ 部分确认 — cv2 梯度是 cv5 的 {ratios.mean():.1f}x")
    else:
        log_and_print(f"❌ 否定 — 梯度量级相近 ({ratios.mean():.1f}x)")

    log_and_print("\n" + "=" * 60)
    log_and_print("假设 3：Refine 对预测框的实际修正量")
    log_and_print("=" * 60)
    for key, label in [("refine_w_diff_mean", "w 修正量(grid)"),
                       ("refine_h_diff_mean", "h 修正量(grid)"),
                       ("refine_a_diff_mean", "θ 修正量(rad)")]:
        if key in stats:
            vals = np.array(stats[key])
            log_and_print(f"  {label}: mean={vals.mean():.6f}")

    log_and_print("\n" + "=" * 60)
    log_and_print("总结")
    log_and_print("=" * 60)
    log_and_print("如果假设 1+2 均确认：冻结除 cv5 外所有参数 + 提高 lr 是正确的修复方向")
    log_and_print("如果假设 1 否定：cv5 有输出但未改善指标，需要检查 loss 计算路径")
    
    # 写入 md 文件
    md_path = "/root/autodl-tmp/work-dirs/refine_diag/refine_diag_results.md"
    try:
        import os
        os.makedirs(os.path.dirname(md_path), exist_ok=False)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Refine Head 诊断报告\n\n```text\n")
            f.write("\n".join(report_lines))
            f.write("\n```\n")
        print(f"\n[*] 诊断报告已保存到 {md_path}")
    except Exception as e:
        print(f"\n[!] 保存 md 报告失败: {e}")


if __name__ == "__main__":
    collect_diagnostics()
