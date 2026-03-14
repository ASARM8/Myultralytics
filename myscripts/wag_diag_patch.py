"""
WAG (Width-Aware Geometric Loss) 诊断 Patch
=============================================
功能：monkey-patch v8OBBLoss.loss()，在每 N 步记录以下统计量：
  1. 门控激活率：gate.sum() / fg_mask.sum()
  2. 各损失分量量级（L_perp, L_w, L_theta, aux_geo 总计 vs box/cls/dfl/angle）
  3. 门控样本的 GT 短边分布（pixel 单位）
  4. 门控样本的 GT AR 分布
  5. 各 FPN 层的门控激活率

用法：
  在训练脚本的 model.train() 之前，import 并调用 apply_wag_diag_patch(model)
  或者直接运行此脚本进行单次 forward pass 诊断（需指定模型和数据集路径）

输出文件：<save_dir>/wag_diag_log.csv
"""

import math
import os
import csv
import torch
from collections import defaultdict


# 全局存储
_diag_state = {
    "step": 0,
    "log_every": 10,  # 每 N 个 batch 记录一次
    "csv_file": None,
    "csv_writer": None,
    "records": [],
}


def _patched_loss(self_loss, preds, batch, _original_loss=None):
    """替换 v8OBBLoss.loss()，增加诊断日志记录。"""
    from ultralytics.utils.tal import make_anchors

    device = self_loss.device
    loss = torch.zeros(5, device=device)  # box, cls, dfl, angle, aux_geo
    pred_distri, pred_scores, pred_angle = (
        preds["boxes"].permute(0, 2, 1).contiguous(),
        preds["scores"].permute(0, 2, 1).contiguous(),
        preds["angle"].permute(0, 2, 1).contiguous(),
    )
    anchor_points, stride_tensor = make_anchors(preds["feats"], self_loss.stride, 0.5)
    batch_size = pred_angle.shape[0]

    dtype = pred_scores.dtype
    imgsz = torch.tensor(preds["feats"][0].shape[2:], device=device, dtype=dtype) * self_loss.stride[0]

    # targets
    try:
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
        rw, rh = targets[:, 4] * float(imgsz[1]), targets[:, 5] * float(imgsz[0])
        targets = targets[(rw >= 2) & (rh >= 2)]
        targets = self_loss.preprocess(targets.to(device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 5), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
    except RuntimeError as e:
        raise TypeError("OBB dataset format error") from e

    # Pboxes
    pred_bboxes = self_loss.bbox_decode(anchor_points, pred_distri, pred_angle)

    bboxes_for_assigner = pred_bboxes.clone().detach()
    bboxes_for_assigner[..., :4] *= stride_tensor
    self_loss.assigner._stride_tensor = stride_tensor
    _, target_bboxes, target_scores, fg_mask, _ = self_loss.assigner(
        pred_scores.detach().sigmoid(),
        bboxes_for_assigner.type(gt_bboxes.dtype),
        anchor_points * stride_tensor,
        gt_labels,
        gt_bboxes,
        mask_gt,
    )

    target_scores_sum = max(target_scores.sum(), 1)

    # Cls loss
    loss[1] = self_loss.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

    # ====== 诊断：收集统计量 ======
    diag = {}
    n_fg = int(fg_mask.sum())
    diag["n_fg"] = n_fg
    diag["batch_size"] = batch_size

    if fg_mask.sum():
        weight = target_scores.sum(-1)[fg_mask]

        # === 辅助损失诊断（在 stride 归一化前） ===
        eps = 1e-6
        tgt = target_bboxes[fg_mask]  # (N_fg, 5) pixel coords
        pred_px = pred_bboxes.clone()
        pred_px[..., :4] *= stride_tensor
        pred_fg = pred_px[fg_mask]  # (N_fg, 5) pixel coords

        w_gt, h_gt = tgt[:, 2], tgt[:, 3]
        w_pred, h_pred = pred_fg[:, 2], pred_fg[:, 3]
        theta_gt, theta_pred = tgt[:, 4], pred_fg[:, 4]

        w_is_long = w_gt >= h_gt
        short_gt = torch.where(w_is_long, h_gt, w_gt)
        long_gt = torch.where(w_is_long, w_gt, h_gt)
        theta_long = theta_gt + torch.where(
            w_is_long, torch.zeros_like(theta_gt), torch.full_like(theta_gt, math.pi / 2)
        )

        ar = long_gt / (short_gt + eps)
        ar_thresh = self_loss.hyp.aux_geo_ar
        ws_thresh = self_loss.hyp.aux_geo_ws
        gate = ((ar > ar_thresh) | (short_gt < ws_thresh)).float()

        n_gated = int(gate.sum())
        diag["n_gated"] = n_gated
        diag["gate_ratio"] = n_gated / max(n_fg, 1)

        # GT 短边统计（仅门控样本）
        if n_gated > 0:
            gated_mask = gate.bool()
            gated_short = short_gt[gated_mask]
            diag["gated_short_mean"] = float(gated_short.mean())
            diag["gated_short_min"] = float(gated_short.min())
            diag["gated_short_max"] = float(gated_short.max())
            diag["gated_ar_mean"] = float(ar[gated_mask].mean())
            diag["gated_ar_max"] = float(ar[gated_mask].max())

            # 各损失分量（门控样本的均值）
            center_offset = pred_fg[:, :2] - tgt[:, :2]
            n_x = -torch.sin(theta_long)
            n_y = torch.cos(theta_long)
            d_perp = (center_offset[:, 0] * n_x + center_offset[:, 1] * n_y).abs()
            loss_perp_raw = d_perp / (short_gt + eps)

            short_pred = torch.min(w_pred, h_pred)
            loss_w_raw = torch.abs(torch.log((short_pred + eps) / (short_gt + eps)))

            loss_theta_raw = 1.0 - torch.cos(theta_pred - theta_gt)

            diag["L_perp_gated_mean"] = float(loss_perp_raw[gated_mask].mean())
            diag["L_w_gated_mean"] = float(loss_w_raw[gated_mask].mean())
            diag["L_theta_gated_mean"] = float(loss_theta_raw[gated_mask].mean())

            # 加权后的总 aux loss（未乘 hyp.aux_geo）
            lp = self_loss.hyp.aux_geo_lp
            lw = self_loss.hyp.aux_geo_lw
            lt = self_loss.hyp.aux_geo_lt
            aux_raw = gate * (lp * loss_perp_raw + lw * loss_w_raw + lt * loss_theta_raw)
            aux_weighted = (aux_raw * weight).sum() / target_scores_sum
            diag["aux_geo_raw"] = float(aux_weighted)  # 未乘 hyp.aux_geo

            # FPN 层门控分析
            strides_fg = stride_tensor.squeeze(-1).expand(batch_size, -1)[fg_mask]
            for s in [8, 16, 32]:
                s_mask = (strides_fg == s)
                n_s = int(s_mask.sum())
                n_s_gated = int((gate * s_mask.float()).sum()) if n_s > 0 else 0
                diag[f"P{int(math.log2(s))-1}_n_fg"] = n_s
                diag[f"P{int(math.log2(s))-1}_n_gated"] = n_s_gated
                if n_s > 0:
                    diag[f"P{int(math.log2(s))-1}_short_mean"] = float(short_gt[s_mask].mean())
        else:
            diag["gated_short_mean"] = 0
            diag["gated_ar_mean"] = 0
            diag["L_perp_gated_mean"] = 0
            diag["L_w_gated_mean"] = 0
            diag["L_theta_gated_mean"] = 0
            diag["aux_geo_raw"] = 0

        # === 计算正常损失 ===
        loss[4] = self_loss.calculate_aux_geo_loss(
            pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum, stride_tensor
        )
        target_bboxes[..., :4] /= stride_tensor
        loss[0], loss[2] = self_loss.bbox_loss(
            pred_distri, pred_bboxes, anchor_points, target_bboxes,
            target_scores, target_scores_sum, fg_mask, imgsz, stride_tensor,
        )
        loss[3] = self_loss.calculate_angle_loss(
            pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum
        )
    else:
        loss[0] += (pred_angle * 0).sum()
        diag["n_gated"] = 0
        diag["gate_ratio"] = 0

    # 记录各 loss 分量（乘 gain 前）
    diag["box_raw"] = float(loss[0])
    diag["cls_raw"] = float(loss[1])
    diag["dfl_raw"] = float(loss[2])
    diag["angle_raw"] = float(loss[3])
    diag["aux_geo_after_gain"] = float(loss[4] * self_loss.hyp.aux_geo)

    loss[0] *= self_loss.hyp.box
    loss[1] *= self_loss.hyp.cls
    loss[2] *= self_loss.hyp.dfl
    loss[3] *= self_loss.hyp.angle
    loss[4] *= self_loss.hyp.aux_geo

    diag["box_final"] = float(loss[0])
    diag["cls_final"] = float(loss[1])
    diag["dfl_final"] = float(loss[2])
    diag["angle_final"] = float(loss[3])
    diag["aux_geo_final"] = float(loss[4])
    diag["total"] = float(loss.sum())
    diag["aux_geo_ratio"] = float(loss[4]) / max(float(loss.sum()), 1e-8)

    # 周期性打印
    step = _diag_state["step"]
    if step % _diag_state["log_every"] == 0:
        print(f"\n[WAG-DIAG step={step}] "
              f"n_fg={diag['n_fg']}, n_gated={diag['n_gated']}, "
              f"gate_ratio={diag['gate_ratio']:.3f}")
        if diag['n_gated'] > 0:
            print(f"  短边(px): mean={diag.get('gated_short_mean',0):.1f}, "
                  f"min={diag.get('gated_short_min',0):.1f}, max={diag.get('gated_short_max',0):.1f}")
            print(f"  AR: mean={diag.get('gated_ar_mean',0):.1f}, max={diag.get('gated_ar_max',0):.1f}")
            print(f"  L_perp={diag['L_perp_gated_mean']:.4f}, L_w={diag['L_w_gated_mean']:.4f}, "
                  f"L_theta={diag['L_theta_gated_mean']:.6f}")
        print(f"  loss: box={diag['box_final']:.4f}, cls={diag['cls_final']:.4f}, "
              f"dfl={diag['dfl_final']:.4f}, angle={diag['angle_final']:.4f}, "
              f"aux_geo={diag['aux_geo_final']:.4f}")
        print(f"  aux_geo占比: {diag['aux_geo_ratio']*100:.2f}%")

    _diag_state["records"].append(diag)
    _diag_state["step"] += 1

    return loss * batch_size, loss.detach()


def apply_wag_diag_patch(model, log_every=10):
    """
    对 YOLO model 的 loss 函数应用诊断 patch。
    在 model.train() 之前调用。

    Args:
        model: YOLO model 对象
        log_every: 每 N 步打印一次诊断信息
    """
    from ultralytics.utils.loss import v8OBBLoss

    _diag_state["log_every"] = log_every
    _diag_state["step"] = 0
    _diag_state["records"] = []

    # Monkey-patch loss method
    original_loss = v8OBBLoss.loss
    v8OBBLoss.loss = _patched_loss

    print(f"[WAG-DIAG] 已 patch v8OBBLoss.loss, 每 {log_every} 步记录诊断信息")

    return original_loss


def save_diag_results(save_path="wag_diag_log.csv"):
    """保存诊断记录到 CSV。"""
    records = _diag_state["records"]
    if not records:
        print("[WAG-DIAG] 无记录可保存")
        return

    keys = list(records[0].keys())
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"[WAG-DIAG] 已保存 {len(records)} 条记录到 {save_path}")


def print_diag_summary():
    """打印诊断汇总统计。"""
    records = _diag_state["records"]
    if not records:
        print("[WAG-DIAG] 无记录")
        return

    import statistics

    n = len(records)
    print(f"\n{'='*70}")
    print(f"WAG 诊断汇总（共 {n} 步）")
    print(f"{'='*70}")

    # 门控统计
    gate_ratios = [r["gate_ratio"] for r in records]
    n_gated_list = [r["n_gated"] for r in records]
    n_fg_list = [r["n_fg"] for r in records]
    print(f"\n--- 门控统计 ---")
    print(f"  平均前景样本数/batch: {statistics.mean(n_fg_list):.1f}")
    print(f"  平均门控样本数/batch: {statistics.mean(n_gated_list):.1f}")
    print(f"  门控激活率: mean={statistics.mean(gate_ratios):.4f}, "
          f"min={min(gate_ratios):.4f}, max={max(gate_ratios):.4f}")
    zero_gate = sum(1 for r in records if r["n_gated"] == 0)
    print(f"  门控全零的 batch 比例: {zero_gate}/{n} ({zero_gate/n*100:.1f}%)")

    # 损失量级
    print(f"\n--- 损失量级（乘 gain 后均值）---")
    for key in ["box_final", "cls_final", "dfl_final", "angle_final", "aux_geo_final"]:
        vals = [r[key] for r in records]
        print(f"  {key}: mean={statistics.mean(vals):.4f}, std={statistics.stdev(vals) if len(vals)>1 else 0:.4f}")

    aux_ratios = [r["aux_geo_ratio"] for r in records if r["aux_geo_ratio"] > 0]
    if aux_ratios:
        print(f"\n  aux_geo 占总 loss 比例: mean={statistics.mean(aux_ratios)*100:.2f}%, "
              f"max={max(aux_ratios)*100:.2f}%")

    # 分量统计（仅门控非零的 batch）
    gated_records = [r for r in records if r["n_gated"] > 0]
    if gated_records:
        print(f"\n--- 门控样本的损失分量均值（{len(gated_records)} 个非零 batch）---")
        for key in ["L_perp_gated_mean", "L_w_gated_mean", "L_theta_gated_mean"]:
            vals = [r[key] for r in gated_records]
            print(f"  {key}: {statistics.mean(vals):.4f}")

        short_means = [r["gated_short_mean"] for r in gated_records]
        ar_means = [r["gated_ar_mean"] for r in gated_records]
        print(f"\n--- 门控样本 GT 属性 ---")
        print(f"  短边(px): mean={statistics.mean(short_means):.1f}")
        print(f"  AR: mean={statistics.mean(ar_means):.1f}")

    # FPN 层分析
    print(f"\n--- FPN 层门控分析 ---")
    for p in ["P2", "P3", "P4"]:
        fg_key = f"{p}_n_fg"
        gated_key = f"{p}_n_gated"
        short_key = f"{p}_short_mean"
        if fg_key in records[0]:
            fg_vals = [r.get(fg_key, 0) for r in records]
            gated_vals = [r.get(gated_key, 0) for r in records]
            short_vals = [r.get(short_key, 0) for r in records if r.get(short_key, 0) > 0]
            total_fg = sum(fg_vals)
            total_gated = sum(gated_vals)
            ratio = total_gated / max(total_fg, 1)
            label = f"stride={2**(int(p[1])+1)}"
            print(f"  {p} ({label}): 总fg={total_fg}, 总gated={total_gated}, "
                  f"比例={ratio:.3f}", end="")
            if short_vals:
                print(f", 短边均值={statistics.mean(short_vals):.1f}px")
            else:
                print()

    print(f"{'='*70}\n")
