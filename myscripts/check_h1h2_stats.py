"""
DFL 回归截断 & 正样本层级分配 验证统计脚本 (H1 + H2)

根据 mydocs/check.md 的验证方案，在训练过程中收集以下统计信息：
  H1: 每个正样本的所需回归距离 D_req 是否超出 DFL 理论上限 D_max = stride × (reg_max - 1)
  H2: 不同长边长度的 GT 框，其正样本被分配到了 FPN 的哪一层 (P3/P4/P5)

使用方法: python myscripts/check_h1h2_stats.py
只需跑 1~3 个 Epoch 即可收集足够的统计学证据。

原理:
  - 通过 monkey-patch v8OBBLoss.loss()，在每个 batch 的损失计算前
    额外调用一次 assigner 以获取 target_gt_idx（原始代码将其丢弃），
    然后对所有正样本计算无截断的旋转回归距离 (l,t,r,b)，统计溢出情况。
  - 统计完成后正常执行原始 loss，不影响训练过程。
"""

import os
import sys
import re
import json
import datetime
import math
import numpy as np
from collections import defaultdict
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "8"

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.tal import make_anchors


# ========================== 配置 ==========================
CONFIG = {
    # ---------- 模型 ----------
    # 可选 "yolo11l-obb.yaml"（原版）或 "yolo11l-obb-sp.yaml"（含 StripPooling）
    "model": "yolo11l-obb.yaml",
    # 迁移权重（留空则从零开始；SP 模型需要层索引重映射，脚本自动处理）
    "pretrained_weights": "/root/autodl-tmp/work-dirs/yolo11_obb-v2.1full/weights/best.pt",

    # ---------- 数据 ----------
    "data": "/root/autodl-tmp/dataset/TTPLA-1024/dataset.yaml",

    # ---------- 训练参数（仅用于统计，跑 1~3 个 epoch 即可） ----------
    "epochs": 3,
    "batch": 20,
    "imgsz": 1024,
    "device": 0,
    "workers": 16,

    # ---------- 输出 ----------
    "project": "/root/autodl-tmp/work-dirs",
    "name": "check_h1h2-2",
    "exist_ok": True,

    # ---------- 节省资源：关闭不需要的功能 ----------
    "save": False,
    "val": True,
    "plots": False,
    "pretrained": False,
    "amp": True,
    "cache": "disk",
    "verbose": True,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "warmup_epochs": 1.0,
    "cos_lr": True,

    # ---------- 必须关闭的空间与拼接增强 ----------
    "mosaic": 0.0,      # 必须关闭！防止目标被拼接截断
    "scale": 0.0,       # 必须关闭！防止目标物理长度被缩放
    "degrees": 0.0,     # 关闭旋转，防止 OBB 角度和长宽由于旋转发生变异
    "translate": 0.0,   # 关闭平移，防止长目标被移出图片边缘导致截断
    "shear": 0.0,       # 关闭错切
    "perspective": 0.0, # 关闭透视变换
    "mixup": 0.0,       # 关闭 Mixup
    "copy_paste": 0.0,  # 关闭复制粘贴

    # ---------- 统计参数 ----------
    # 长边分桶边界（像素），可根据数据集实际分布调整
    "long_edge_bins": [0, 50, 100, 200, 300, 500, float("inf")],
    "long_edge_labels": ["<50", "50-100", "100-200", "200-300", "300-500", ">500"],
}

# StripPooling 层索引重映射插入位置（仅 SP 模型使用）
SP_INSERT_IDX = 10


# ========================== 统计收集器 ==========================
class H1H2Collector:
    """收集 H1 (DFL 回归截断) 和 H2 (正样本层级分配) 统计数据。"""

    def __init__(self, bins, labels):
        self.bins = bins
        self.labels = labels
        self.n_bins = len(labels)

        # H1: 每个分桶的溢出数 / 总正样本数 / D_req 值列表
        self.h1_overflow = defaultdict(int)
        self.h1_total = defaultdict(int)
        self.h1_dreq_values = defaultdict(list)

        # H2: h2_counts[bin_idx][stride_value] = count
        self.h2_counts = defaultdict(lambda: defaultdict(int))

        self.batch_count = 0

    def _get_bin(self, long_edge_px):
        """根据长边像素值返回分桶索引。"""
        for i in range(self.n_bins):
            if long_edge_px < self.bins[i + 1]:
                return i
        return self.n_bins - 1

    @staticmethod
    def _rbox2dist_unclamped(target_xywh, anchor_points, target_angle):
        """
        计算旋转框回归距离 (l, t, r, b)，**不做 clamp**。
        与 tal.py 中的 rbox2dist 逻辑一致，但不截断到 reg_max。

        Args:
            target_xywh: (N, 4) 目标框 [x, y, w, h]（grid 坐标）
            anchor_points: (N, 2) 锚点 [cx, cy]（grid 坐标）
            target_angle: (N, 1) 旋转角度（弧度）

        Returns:
            (N, 4) 回归距离 [l, t, r, b]（grid 单位，未截断）
        """
        xy, wh = target_xywh.split(2, dim=-1)
        offset = xy - anchor_points
        offset_x, offset_y = offset.split(1, dim=-1)
        cos, sin = torch.cos(target_angle), torch.sin(target_angle)
        xf = offset_x * cos + offset_y * sin
        yf = -offset_x * sin + offset_y * cos
        w, h = wh.split(1, dim=-1)
        target_l = w / 2 - xf
        target_t = h / 2 - yf
        target_r = w / 2 + xf
        target_b = h / 2 + yf
        return torch.cat([target_l, target_t, target_r, target_b], dim=-1)

    @torch.no_grad()
    def collect(self, gt_bboxes, target_bboxes, target_gt_idx, fg_mask,
                anchor_points, stride_tensor, reg_max, mask_gt):
        """
        在每个 batch 的 loss 计算中调用，收集 H1 & H2 统计数据。

        Args:
            gt_bboxes:      (B, N_max, 5) xywhr 像素坐标
            target_bboxes:  (B, H*W, 5)   分配给每个 anchor 的 GT bbox（像素坐标）
            target_gt_idx:  (B, H*W)       每个 anchor 分配到的 GT 索引
            fg_mask:        (B, H*W)       前景掩码（bool）
            anchor_points:  (H*W, 2)       锚点（grid 坐标）
            stride_tensor:  (H*W, 1)       每个 anchor 的 stride
            reg_max:        int            DFL 最大回归值
            mask_gt:        (B, N_max, 1)  GT 有效掩码
        """
        self.batch_count += 1
        bs = gt_bboxes.shape[0]
        d_max_grid = float(reg_max - 1)  # DFL 理论上限（grid 单位）

        for b in range(bs):
            fg = fg_mask[b]  # (H*W,)
            if not fg.any():
                continue

            fg_indices = fg.nonzero(as_tuple=True)[0]
            gt_idx = target_gt_idx[b][fg_indices].long()
            strides = stride_tensor[fg_indices, 0]

            valid_mask = mask_gt[b].squeeze(-1).bool()
            gt_wh = gt_bboxes[b, :, 2:4]  # (N_max, 2) — w, h 像素
            gt_long_edge = gt_wh.max(dim=-1).values

            # === H1: 计算无截断回归距离 ===
            fg_target = target_bboxes[b][fg_indices]  # (n_fg, 5) 像素坐标
            fg_target_grid = fg_target.clone()
            fg_target_grid[..., :4] /= strides.unsqueeze(-1)  # 转 grid 坐标
            fg_anchor = anchor_points[fg_indices]              # (n_fg, 2) grid
            fg_angle = fg_target_grid[..., 4:5]                # (n_fg, 1)

            dist = self._rbox2dist_unclamped(
                fg_target_grid[..., :4], fg_anchor, fg_angle
            )  # (n_fg, 4)
            d_req = dist.max(dim=-1).values  # (n_fg,)

            # === 逐正样本统计 ===
            for i in range(len(fg_indices)):
                gidx = gt_idx[i].item()
                if gidx >= gt_long_edge.shape[0] or not valid_mask[gidx]:
                    continue

                le = gt_long_edge[gidx].item()
                bin_idx = self._get_bin(le)
                s = strides[i].item()
                dreq_val = d_req[i].item()

                # H1
                self.h1_total[bin_idx] += 1
                if dreq_val > d_max_grid:
                    self.h1_overflow[bin_idx] += 1
                self.h1_dreq_values[bin_idx].append(dreq_val)

                # H2
                self.h2_counts[bin_idx][int(s)] += 1

    def report(self, reg_max, strides, save_dir=None):
        """生成并打印统计报告，可选保存到文件。"""
        d_max_grid = reg_max - 1
        all_strides = sorted(set(s for counts in self.h2_counts.values() for s in counts))
        if not all_strides:
            all_strides = [int(s) for s in strides]

        # ==================== 控制台报告 ====================
        print("\n" + "=" * 80)
        print("  H1: DFL 回归截断统计 — D_req > D_max 的正样本比例")
        print(f"  reg_max = {reg_max}, D_max (grid) = {d_max_grid}")
        for s in all_strides:
            print(f"  P{int(math.log2(s))} (stride={s}): D_max (pixel) = {s * d_max_grid}")
        print("-" * 80)
        header = f"{'长边分桶':<12} {'正样本数':>10} {'溢出数':>10} {'溢出比例':>10} {'D_req均值':>10} {'D_req P95':>10} {'D_req最大':>10}"
        print(header)
        print("-" * 80)

        for i in range(self.n_bins):
            total = self.h1_total.get(i, 0)
            overflow = self.h1_overflow.get(i, 0)
            ratio = overflow / total * 100 if total > 0 else 0
            dreqs = self.h1_dreq_values.get(i, [])
            mean_d = np.mean(dreqs) if dreqs else 0
            p95_d = np.percentile(dreqs, 95) if dreqs else 0
            max_d = max(dreqs) if dreqs else 0
            print(f"{self.labels[i]:<12} {total:>10d} {overflow:>10d} {ratio:>9.1f}% {mean_d:>10.1f} {p95_d:>10.1f} {max_d:>10.1f}")

        # H2 控制台
        print("\n" + "=" * 80)
        print("  H2: 正样本层级分配统计 — 各长边分桶的 FPN 层分布")
        print("-" * 80)
        stride_labels = [f"P{int(math.log2(s))}(s={s})" for s in all_strides]
        header2 = f"{'长边分桶':<12}" + "".join(f"{sl:>18}" for sl in stride_labels) + f"{'合计':>10}"
        print(header2)
        print("-" * 80)

        for i in range(self.n_bins):
            counts = self.h2_counts.get(i, {})
            total = sum(counts.values())
            parts = []
            for s in all_strides:
                c = counts.get(s, 0)
                pct = c / total * 100 if total > 0 else 0
                parts.append(f"{c:>6d}({pct:>5.1f}%)")
            print(f"{self.labels[i]:<12}" + "".join(f"{p:>18}" for p in parts) + f"{total:>10d}")

        print("=" * 80)

        # ==================== 保存报告 ====================
        if save_dir is None:
            return

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # ---------- Markdown 报告 ----------
        md = []
        md.append("# H1 & H2 验证统计报告\n")
        md.append(f"- **统计批次数**: {self.batch_count}")
        md.append(f"- **生成时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        md.append("## H1: DFL 回归截断统计\n")
        md.append(f"- `reg_max` = {reg_max}, `D_max` (grid) = {d_max_grid}")
        for s in all_strides:
            md.append(f"- P{int(math.log2(s))} (stride={s}): D_max (pixel) = {s * d_max_grid}")
        md.append("")
        md.append("| 长边分桶 | 正样本数 | 溢出数 | 溢出比例 | D_req 均值 | D_req P95 | D_req 最大 |")
        md.append("|:------:|-------:|------:|-------:|----------:|----------:|----------:|")
        for i in range(self.n_bins):
            total = self.h1_total.get(i, 0)
            overflow = self.h1_overflow.get(i, 0)
            ratio = overflow / total * 100 if total > 0 else 0
            dreqs = self.h1_dreq_values.get(i, [])
            mean_d = np.mean(dreqs) if dreqs else 0
            p95_d = np.percentile(dreqs, 95) if dreqs else 0
            max_d = max(dreqs) if dreqs else 0
            md.append(f"| {self.labels[i]} | {total} | {overflow} | {ratio:.1f}% | {mean_d:.1f} | {p95_d:.1f} | {max_d:.1f} |")

        md.append("\n## H2: 正样本层级分配统计\n")
        h_parts = ["| 长边分桶 |"] + [f" {sl} |" for sl in stride_labels] + [" 合计 |"]
        md.append("".join(h_parts))
        sep_parts = ["|:------:|"] + ["-------:|" for _ in stride_labels] + ["-----:|"]
        md.append("".join(sep_parts))
        for i in range(self.n_bins):
            counts = self.h2_counts.get(i, {})
            total = sum(counts.values())
            row = [f"| {self.labels[i]} |"]
            for s in all_strides:
                c = counts.get(s, 0)
                pct = c / total * 100 if total > 0 else 0
                row.append(f" {c} ({pct:.1f}%) |")
            row.append(f" {total} |")
            md.append("".join(row))

        # ---------- 自动判定 ----------
        md.append("\n## 自动判定\n")

        # H1 判定：取长边 >= 200px 的分桶
        long_bins = [i for i in range(self.n_bins) if self.bins[i] >= 200]
        long_total = sum(self.h1_total.get(i, 0) for i in long_bins)
        long_overflow = sum(self.h1_overflow.get(i, 0) for i in long_bins)
        long_ratio = long_overflow / long_total * 100 if long_total > 0 else 0

        if long_ratio > 30:
            md.append(f"### H1: **成立** (长线缆溢出比例 {long_ratio:.1f}% > 30%)")
            md.append("DFL 回归上限是主要瓶颈，建议增大 `reg_max` 或引导长目标分配到大 stride 层。\n")
        elif long_ratio > 5:
            md.append(f"### H1: **部分成立** (长线缆溢出比例 {long_ratio:.1f}%，介于 5%~30%)")
            md.append("存在一定截断但非全面瓶颈，可能与层级分配偏差共同作用。\n")
        else:
            md.append(f"### H1: **不成立** (长线缆溢出比例 {long_ratio:.1f}% < 5%)")
            md.append("`reg_max` 设定充足，长度受限于其他原因（如特征提取能力）。\n")

        # H2 判定：取长边 >= 200px 的分桶中 P3 层比例
        min_stride = min(all_strides) if all_strides else 8
        p3_count_long = sum(self.h2_counts.get(i, {}).get(min_stride, 0) for i in long_bins)
        p3_ratio_long = p3_count_long / long_total * 100 if long_total > 0 else 0

        if p3_ratio_long > 50:
            md.append(f"### H2: **成立** (长线缆 P3 层分配比例 {p3_ratio_long:.1f}% > 50%)")
            md.append("长目标被大量分配到小感受野的浅层，存在严重层级错配。\n")
        elif p3_ratio_long > 30:
            md.append(f"### H2: **部分成立** (长线缆 P3 层分配比例 {p3_ratio_long:.1f}%，介于 30%~50%)")
            md.append("存在一定层级偏移，但非全面错配。\n")
        else:
            md.append(f"### H2: **不成立** (长线缆 P3 层分配比例 {p3_ratio_long:.1f}% < 30%)")
            md.append("分配逻辑基本正确，长目标主要落在深层 FPN。\n")

        md_path = save_dir / "h1h2_report.md"
        md_path.write_text("\n".join(md), encoding="utf-8")
        print(f"\nMarkdown 报告已保存至: {md_path}")

        # ---------- JSON 原始数据 ----------
        json_data = {
            "config": {
                "reg_max": reg_max,
                "strides": [int(s) for s in all_strides],
                "bins": [b for b in self.bins if b != float("inf")] + ["inf"],
                "labels": self.labels,
                "batch_count": self.batch_count,
            },
            "h1": {},
            "h2": {},
        }
        for i in range(self.n_bins):
            label = self.labels[i]
            dreqs = self.h1_dreq_values.get(i, [])
            json_data["h1"][label] = {
                "total": self.h1_total.get(i, 0),
                "overflow": self.h1_overflow.get(i, 0),
                "dreq_mean": float(np.mean(dreqs)) if dreqs else 0,
                "dreq_p95": float(np.percentile(dreqs, 95)) if dreqs else 0,
                "dreq_max": float(max(dreqs)) if dreqs else 0,
            }
            json_data["h2"][label] = {
                str(s): self.h2_counts.get(i, {}).get(s, 0)
                for s in all_strides
            }

        json_path = save_dir / "h1h2_data.json"
        json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON 原始数据已保存至: {json_path}")

        # ---------- 堆叠柱状图（H2 可视化） ----------
        self._plot_h2_bar(all_strides, save_dir)

    def _plot_h2_bar(self, all_strides, save_dir):
        """生成 H2 正样本层级分配的堆叠柱状图。"""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[警告] 未安装 matplotlib，跳过图表生成。")
            return

        # 准备数据
        x = np.arange(self.n_bins)
        width = 0.6
        bottom = np.zeros(self.n_bins)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#4FC3F7", "#FFB74D", "#EF5350", "#66BB6A", "#AB47BC"]

        for idx, s in enumerate(all_strides):
            layer_label = f"P{int(math.log2(s))} (stride={s})"
            counts = []
            for i in range(self.n_bins):
                total = sum(self.h2_counts.get(i, {}).values())
                c = self.h2_counts.get(i, {}).get(s, 0)
                pct = c / total * 100 if total > 0 else 0
                counts.append(pct)
            counts = np.array(counts)
            color = colors[idx % len(colors)]
            ax.bar(x, counts, width, bottom=bottom, label=layer_label, color=color, edgecolor="white")

            # 在柱体中心标注百分比
            for j in range(self.n_bins):
                if counts[j] > 3:  # 太小的不标注
                    ax.text(x[j], bottom[j] + counts[j] / 2, f"{counts[j]:.1f}%",
                            ha="center", va="center", fontsize=8, fontweight="bold")
            bottom += counts

        ax.set_xlabel("GT 长边分桶 (像素)", fontsize=12)
        ax.set_ylabel("正样本占比 (%)", fontsize=12)
        ax.set_title("H2: 正样本层级分配分布", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(self.labels, fontsize=10)
        ax.legend(loc="upper right", fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        fig_path = save_dir / "h2_level_distribution.png"
        fig.savefig(str(fig_path), dpi=150)
        plt.close(fig)
        print(f"H2 堆叠柱状图已保存至: {fig_path}")


# ========================== Monkey-Patch ==========================
# 全局收集器实例（在 main 中初始化）
_collector: H1H2Collector | None = None
_original_obb_loss = None


def _install_loss_hook():
    """
    对 v8OBBLoss.loss() 进行 monkey-patch。

    在每个 batch 的 loss 计算前，额外调用一次 assigner 以获取
    target_gt_idx（原始代码将其丢弃），并收集 H1/H2 统计数据。
    然后正常执行原始 loss 函数。
    """
    from ultralytics.utils.loss import v8OBBLoss

    global _original_obb_loss
    _original_obb_loss = v8OBBLoss.loss

    def _hooked_loss(self, preds, batch):
        """带统计收集的 OBB 损失函数。"""
        global _collector
        if _collector is not None:
            try:
                _collect_stats(self, preds, batch)
            except Exception as e:
                # 统计收集不应影响训练
                print(f"[H1H2 统计] 收集异常（已跳过）: {e}")

        return _original_obb_loss(self, preds, batch)

    v8OBBLoss.loss = _hooked_loss


@torch.no_grad()
def _collect_stats(loss_self, preds, batch):
    """
    从 OBB 损失函数中提取必要数据并调用收集器。

    复刻 v8OBBLoss.loss() 中的目标预处理和 assigner 调用逻辑，
    但额外保留 target_gt_idx 用于统计。
    """
    global _collector

    pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
    pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
    pred_angle = preds["angle"].permute(0, 2, 1).contiguous()

    anchor_points, stride_tensor = make_anchors(preds["feats"], loss_self.stride, 0.5)
    batch_size = pred_angle.shape[0]
    dtype = pred_scores.dtype
    imgsz = torch.tensor(
        preds["feats"][0].shape[2:], device=loss_self.device, dtype=dtype
    ) * loss_self.stride[0]

    # 目标预处理（与原始 loss 一致）
    batch_idx = batch["batch_idx"].view(-1, 1)
    targets = torch.cat(
        (batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1
    )
    rw = targets[:, 4] * float(imgsz[1])
    rh = targets[:, 5] * float(imgsz[0])
    targets = targets[(rw >= 2) & (rh >= 2)]
    targets = loss_self.preprocess(
        targets.to(loss_self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]
    )
    gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr（像素坐标）
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

    # 预测框解码
    pred_bboxes = loss_self.bbox_decode(anchor_points, pred_distri, pred_angle)
    bboxes_for_assigner = pred_bboxes.clone().detach()
    bboxes_for_assigner[..., :4] *= stride_tensor

    # 调用 assigner —— 关键：获取 target_gt_idx
    _, target_bboxes, _, fg_mask, target_gt_idx = loss_self.assigner(
        pred_scores.detach().sigmoid(),
        bboxes_for_assigner.type(gt_bboxes.dtype),
        anchor_points * stride_tensor,
        gt_labels,
        gt_bboxes,
        mask_gt,
    )

    # 收集统计
    _collector.collect(
        gt_bboxes=gt_bboxes,
        target_bboxes=target_bboxes,
        target_gt_idx=target_gt_idx,
        fg_mask=fg_mask,
        anchor_points=anchor_points,
        stride_tensor=stride_tensor,
        reg_max=loss_self.reg_max,
        mask_gt=mask_gt,
    )


# ========================== 权重加载辅助 ==========================
def load_weights_with_remap(model, weights_path, insert_idx=None):
    """
    加载预训练权重，支持可选的层索引重映射（用于 SP 模型）。

    Args:
        model: YOLO 模型实例
        weights_path: 权重文件路径
        insert_idx: StripPooling 插入位置（None 表示不重映射）
    """
    if not weights_path or not os.path.exists(weights_path):
        print("  未指定或找不到预训练权重，从随机初始化开始。")
        return

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    old_model = (ckpt.get("ema") or ckpt["model"]).float()
    old_sd = old_model.state_dict()

    if insert_idx is not None:
        # 层索引重映射：旧模型 index >= insert_idx 的层在新模型中对应 index + 1
        remapped_sd = {}
        for k, v in old_sd.items():
            parts = k.split(".")
            if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
                layer_idx = int(parts[1])
                if layer_idx >= insert_idx:
                    parts[1] = str(layer_idx + 1)
            remapped_sd[".".join(parts)] = v
        old_sd = remapped_sd

    new_sd = model.model.state_dict()
    matched_sd = intersect_dicts(old_sd, new_sd)
    model.model.load_state_dict(matched_sd, strict=False)
    print(f"  已加载预训练权重: {weights_path}")
    print(f"  匹配: {len(matched_sd)}/{len(new_sd)} 项")


# ========================== 主函数 ==========================
def main():
    global _collector

    print("=" * 80)
    print("  H1 & H2 验证统计脚本")
    print("  H1: DFL 回归截断分析 (D_req vs D_max)")
    print("  H2: 正样本层级分配分析 (长边分桶 vs FPN 层)")
    print("=" * 80)

    # 1. 初始化收集器
    _collector = H1H2Collector(
        bins=CONFIG["long_edge_bins"],
        labels=CONFIG["long_edge_labels"],
    )

    # 2. 安装 loss hook（必须在 model.train() 之前）
    _install_loss_hook()
    print("[*] 已安装 v8OBBLoss 统计 hook")

    # 3. 创建模型
    model_path = CONFIG["model"]
    model = YOLO(model_path)

    # 4. 加载权重
    pretrained_weights = CONFIG.pop("pretrained_weights", None)
    is_sp_model = "-sp" in str(model_path).lower()
    load_weights_with_remap(
        model,
        pretrained_weights,
        insert_idx=SP_INSERT_IDX if is_sp_model else None,
    )

    # 5. 提取训练参数（去除统计专用配置）
    train_config = {k: v for k, v in CONFIG.items()
                    if k not in ("long_edge_bins", "long_edge_labels")}

    print(f"\n  模型: {model_path}")
    print(f"  数据集: {CONFIG['data']}")
    print(f"  统计轮数: {CONFIG['epochs']} epochs")
    print(f"  长边分桶: {CONFIG['long_edge_labels']}")
    print("=" * 80)

    # 6. 注册回调：在训练结束时输出报告
    report_save_dir = [None]  # 用列表包装以便闭包修改

    def on_train_start(trainer):
        report_save_dir[0] = str(trainer.save_dir)
        print(f"[*] 统计报告将保存至: {report_save_dir[0]}")

    model.add_callback("on_train_start", on_train_start)

    # 7. 开始训练（统计收集在 loss hook 中自动进行）
    model.train(**train_config)

    # 8. 输出报告
    print("\n" + "=" * 80)
    print("  训练完成，正在生成 H1 & H2 统计报告...")
    print("=" * 80)

    # 获取 reg_max 和 strides
    try:
        m = model.model.model[-1]  # Detect 模块
        reg_max = m.reg_max
        strides = m.stride.tolist()
    except Exception:
        reg_max = 16
        strides = [8, 16, 32]
        print(f"[警告] 无法从模型获取 reg_max/stride，使用默认值: reg_max={reg_max}, strides={strides}")

    _collector.report(
        reg_max=reg_max,
        strides=strides,
        save_dir=report_save_dir[0] or f"{CONFIG['project']}/{CONFIG['name']}",
    )

    # 9. 恢复原始 loss 函数
    from ultralytics.utils.loss import v8OBBLoss
    if _original_obb_loss is not None:
        v8OBBLoss.loss = _original_obb_loss
        print("\n[*] 已恢复原始 v8OBBLoss.loss()")

    print("\n统计完成！")


if __name__ == "__main__":
    main()
