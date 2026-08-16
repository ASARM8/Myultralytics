"""
DFL 回归截断 & 正样本层级分配 验证统计脚本 (H1 + H2)

根据 mydocs/check.md 的验证方案，在固定 checkpoint 的只读前向过程中收集：
  H1: 每个正样本的所需回归距离 D_req 是否超出 DFL 理论上限 D_max = stride × (reg_max - 1)
  H2: 不同长边长度的 GT 框，其正样本被分配到了 FPN 的哪一层 (P3/P4/P5)

正式统计推荐直接加载对应方法的完整 checkpoint，例如：
  python myscripts/check_h1h2_stats.py \
    --method ca-refine \
    --model /root/autodl-tmp/work-dirs/yolo11_obb_640_811_ca_refine_scratch/weights/best.pt \
    --data /root/autodl-tmp/datasets/TTPLA-640-811/dataset.yaml

原始模型、CA和CA-Refine必须分别使用自己的checkpoint运行，不能把一种结构的
YAML与另一种结构的权重做参数交集后作为正式统计结果。
正式统计固定使用一个声明的数据 split；可重复若干 pass，但不会执行反向传播、
优化器更新、验证回调或 checkpoint 保存。

原理:
  - 模型始终保持 eval/frozen，只利用检测头返回的原始多尺度预测；
  - 复用 v8OBBLoss 的预处理、解码器和 assigner，但不计算 loss、不更新权重；
  - 对正样本计算未截断的旋转回归距离 (l,t,r,b)，统计溢出与层级分配。
"""

import argparse
import datetime
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

# 必须在导入 NumPy/PyTorch 之前修正，避免非法值触发libgomp警告；尊重用户的合法设置。
_omp_threads = os.environ.get("OMP_NUM_THREADS", "")
if not _omp_threads.isdigit() or int(_omp_threads) <= 0:
    os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.torch_utils import unwrap_model


# ========================== 配置 ==========================
CONFIG = {
    # ---------- 纯统计参数 ----------
    "passes": 1,
    "batch": 8,
    "imgsz": 640,
    "device": 0,
    "workers": 16,

    # ---------- 输出 ----------
    "project": "runs/h1h2",
    "exist_ok": False,

    # ---------- 统计参数 ----------
    # 长边分桶边界（像素），可根据数据集实际分布调整
    # 与论文表1/表2、图3/图5保持一致；这样 <100 桶的 P95 可由原始样本精确计算。
    "long_edge_bins": [0, 100, 200, 300, 500, float("inf")],
    "long_edge_labels": ["<100", "100-200", "200-300", "300-500", ">500"],
}

METHOD_EXPECTATIONS = {
    "baseline": {"head": "OBB", "reg_max": 16},
    "ca": {"head": "OBB", "reg_max": 32},
    "ca-refine": {"head": "OBBRefine", "reg_max": 32},
    "custom": {"head": None, "reg_max": None},
}


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
        self.image_count = 0

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
        self.image_count += int(bs)
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

    def report(self, reg_max, strides, save_dir=None, protocol=None):
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
            print(
                f"{self.labels[i]:<12} {total:>10d} {overflow:>10d} {ratio:>9.1f}% "
                f"{mean_d:>10.1f} {p95_d:>10.1f} {max_d:>10.1f}"
            )

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
        if protocol:
            md.append(f"- **统计模式**: {protocol['mode']}")
            md.append(f"- **数据划分**: `{protocol['split']}`")
            md.append(f"- **只读遍历次数**: {protocol['passes']}")
            md.append(f"- **优化器更新次数**: {protocol['optimization_steps']}")
        md.append(f"- **统计批次数**: {self.batch_count}")
        md.append(f"- **处理图像数（含重复 pass）**: {self.image_count}")
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
            md.append(
                f"| {self.labels[i]} | {total} | {overflow} | {ratio:.1f}% | "
                f"{mean_d:.1f} | {p95_d:.1f} | {max_d:.1f} |"
            )

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
                "image_count": self.image_count,
            },
            "protocol": protocol or {},
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


# ========================== 只读统计 ==========================
# 全局收集器实例（由主函数初始化）
_collector: H1H2Collector | None = None


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

    # 调用 assigner —— 关键：获取 target_gt_idx。
    # 纯统计路径不会调用原始loss，因此先写入当前stride，确保首个batch也启用CA覆盖掩码。
    loss_self.assigner._stride_tensor = stride_tensor
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


def _raw_predictions(outputs):
    """Extract the raw training-style prediction dictionary from an eval forward."""
    if isinstance(outputs, dict):
        return outputs.get("one2many", outputs)
    if isinstance(outputs, (tuple, list)):
        for item in reversed(outputs):
            if isinstance(item, dict):
                return item.get("one2many", item)
    raise TypeError(f"模型前向没有返回原始预测字典，实际类型为 {type(outputs)}")


def _state_tensor_sha256(torch_model) -> str:
    """Hash every in-memory parameter and buffer to prove the frozen pass is side-effect free."""
    digest = hashlib.sha256()
    for name, tensor in sorted(unwrap_model(torch_model).state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _file_sha256(path: str | Path) -> str | None:
    """Hash a checkpoint file; YAML scratch diagnostics do not have a checkpoint hash."""
    source = Path(path)
    if source.suffix.lower() != ".pt":
        return None
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ========================== 模型与命令行校验 ==========================
def parse_args():
    """解析正式H1/H2统计参数。"""
    parser = argparse.ArgumentParser(
        description="收集OBB模型的DFL溢出率与P3/P4/P5正样本分配统计。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--method",
        choices=tuple(METHOD_EXPECTATIONS),
        required=True,
        help="论文方法身份；baseline/ca/ca-refine会严格校验检测头和reg_max。",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="正式统计应传对应方法的完整.pt checkpoint；YAML仅允许随机初始化诊断。",
    )
    parser.add_argument("--data", required=True, help="TTPLA-640-811数据集YAML。")
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="val",
        help="只统计一个明确的数据划分；正式证据默认使用val。",
    )
    parser.add_argument("--passes", type=int, default=CONFIG["passes"], help="只读遍历指定split的次数。")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--batch", type=int, default=CONFIG["batch"])
    parser.add_argument("--imgsz", type=int, default=CONFIG["imgsz"])
    parser.add_argument("--device", default=str(CONFIG["device"]))
    parser.add_argument("--workers", type=int, default=CONFIG["workers"])
    parser.add_argument("--project", default=CONFIG["project"])
    parser.add_argument("--name", default=None, help="默认使用check_h1h2_<method>_640。")
    parser.add_argument("--exist-ok", action="store_true", default=CONFIG["exist_ok"])
    parser.add_argument(
        "--allow-yaml-scratch",
        action="store_true",
        help="允许从YAML随机初始化；正式论文统计不应启用。",
    )
    parser.add_argument(
        "--allow-stat-skip",
        action="store_true",
        help="统计异常时跳过当前batch；正式统计默认遇到异常立即失败。",
    )
    args = parser.parse_args()

    if args.imgsz != 640:
        parser.error("本论文H1/H2统计固定使用imgsz=640，拒绝非640输入。")
    if args.epochs is not None:
        if args.passes != CONFIG["passes"]:
            parser.error("--epochs是旧版兼容参数，不能与--passes同时使用。")
        args.passes = args.epochs
        print("[兼容提示] --epochs现仅表示只读passes，不会启动训练。建议改用--passes。")
    if args.passes < 1:
        parser.error("--passes必须大于等于1。")
    if args.batch < 1:
        parser.error("--batch必须大于等于1。")

    model_suffix = Path(args.model).suffix.lower()
    if model_suffix == ".pt":
        if not Path(args.model).is_file():
            parser.error(f"checkpoint不存在: {args.model}")
    elif model_suffix in {".yaml", ".yml"}:
        if not args.allow_yaml_scratch:
            parser.error(
                "正式统计必须直接传.pt checkpoint。若仅做随机初始化诊断，请显式增加--allow-yaml-scratch。"
            )
    else:
        parser.error("--model必须是.pt checkpoint或.yaml模型配置。")

    args.name = args.name or f"check_h1h2_{args.method.replace('-', '_')}_640"
    return args


def model_head_metadata(torch_model) -> dict:
    """读取实际检测头元数据。"""
    base_model = unwrap_model(torch_model)
    if not hasattr(base_model, "model") or not base_model.model:
        raise RuntimeError("无法从模型中定位检测头。")
    head = base_model.model[-1]
    reg_max = int(getattr(head, "reg_max", -1))
    strides = getattr(head, "stride", None)
    if hasattr(strides, "tolist"):
        strides = strides.tolist()
    elif strides is not None:
        strides = list(strides)
    model_yaml = getattr(base_model, "yaml", {})
    return {
        "head": type(head).__name__,
        "reg_max": reg_max,
        "nc": int(getattr(head, "nc", -1)),
        "has_cv5": hasattr(head, "cv5") and getattr(head, "cv5") is not None,
        "coverage_aware": reg_max > 16,
        "strides": strides,
        "yaml_file": model_yaml.get("yaml_file") if isinstance(model_yaml, dict) else None,
    }


def validate_model_identity(torch_model, method: str, *, expected_nc: int | None = None, stage: str) -> dict:
    """校验checkpoint、方法标签和实际冻结模型是否一致。"""
    metadata = model_head_metadata(torch_model)
    expected = METHOD_EXPECTATIONS[method]
    errors = []
    if expected["head"] and metadata["head"] != expected["head"]:
        errors.append(f"检测头应为{expected['head']}，实际为{metadata['head']}")
    if expected["reg_max"] is not None and metadata["reg_max"] != expected["reg_max"]:
        errors.append(f"reg_max应为{expected['reg_max']}，实际为{metadata['reg_max']}")
    if method == "ca-refine" and not metadata["has_cv5"]:
        errors.append("CA-Refine检测头缺少cv5精修分支")
    if method in {"baseline", "ca"} and metadata["has_cv5"]:
        errors.append(f"{method}不应包含cv5精修分支")
    normalized_strides = [int(round(float(value))) for value in metadata["strides"] or []]
    if method != "custom" and normalized_strides != [8, 16, 32]:
        errors.append(f"FPN stride应为[8, 16, 32]，实际为{metadata['strides']}")
    if expected_nc is not None and metadata["nc"] != int(expected_nc):
        errors.append(f"类别数应与数据集一致({expected_nc})，实际为{metadata['nc']}")
    if errors:
        raise RuntimeError(f"{stage}模型身份校验失败: " + "；".join(errors))

    print(
        f"  [{stage}] head={metadata['head']}, reg_max={metadata['reg_max']}, "
        f"nc={metadata['nc']}, cv5={metadata['has_cv5']}, "
        f"CA={metadata['coverage_aware']}, strides={metadata['strides']}, "
        f"yaml={metadata['yaml_file']}"
    )
    return metadata


def write_run_manifest(
    save_dir: str | Path,
    args,
    metadata: dict,
    dataset_nc: int,
    *,
    dataset_images: int,
    model_state_sha256_before: str,
    model_state_sha256_after: str,
    checkpoint_sha256_before: str | None,
    checkpoint_sha256_after: str | None,
) -> Path:
    """保存可复现的模型身份与统计配置。"""
    path = Path(save_dir) / "h1h2_run_config.json"
    payload = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "method": args.method,
        "model": str(Path(args.model).resolve()),
        "data": str(Path(args.data).resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "passes": args.passes,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "dataset_nc": int(dataset_nc),
        "dataset_images": int(dataset_images),
        "model_metadata": metadata,
        "long_edge_bins": CONFIG["long_edge_bins"],
        "long_edge_labels": CONFIG["long_edge_labels"],
        "strict_stats": not args.allow_stat_skip,
        "mode": "frozen_eval_assigner_statistics",
        "model_training": False,
        "gradient_enabled": False,
        "optimization_steps": 0,
        "checkpoint_saving": False,
        "validation_callback": False,
        "augmentation": False,
        "model_state_sha256_before": model_state_sha256_before,
        "model_state_sha256_after": model_state_sha256_after,
        "model_state_unchanged": model_state_sha256_before == model_state_sha256_after,
        "checkpoint_sha256_before": checkpoint_sha256_before,
        "checkpoint_sha256_after": checkpoint_sha256_after,
        "checkpoint_file_unchanged": (
            checkpoint_sha256_before == checkpoint_sha256_after
            if checkpoint_sha256_before is not None
            else None
        ),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[*] 运行配置已保存至: {path}")
    return path


# ========================== 主函数 ==========================
def main():
    global _collector
    args = parse_args()

    print("=" * 80)
    print("  H1 & H2 固定权重只读统计脚本")
    print("  H1: DFL 回归截断分析 (D_req vs D_max)")
    print("  H2: 正样本层级分配分析 (长边分桶 vs FPN 层)")
    print("=" * 80)

    _collector = H1H2Collector(
        bins=CONFIG["long_edge_bins"],
        labels=CONFIG["long_edge_labels"],
    )

    # checkpoint模式会恢复精确的检测头、类别数和所有分支；YAML只用于显式scratch诊断。
    model = YOLO(args.model)
    from ultralytics.utils.files import increment_path
    from ultralytics.utils.torch_utils import select_device

    from myscripts.V3.runtime import build_dataset, full_loader

    device = select_device(args.device)
    core_model = model.model.to(device).float().eval()
    for parameter in core_model.parameters():
        parameter.requires_grad_(False)
    dataset, data = build_dataset(
        args.data,
        args.split,
        args.imgsz,
        args.batch,
        args.workers,
        rect=True,
    )
    metadata = validate_model_identity(
        core_model,
        args.method,
        expected_nc=int(data["nc"]),
        stage="冻结模型",
    )
    loader = full_loader(dataset, args.batch, args.workers, shuffle=False)
    criterion = core_model.init_criterion()
    save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok, mkdir=True)
    state_before = _state_tensor_sha256(core_model)
    checkpoint_before = _file_sha256(args.model)

    print(f"\n  方法: {args.method}")
    print(f"  模型: {args.model}")
    print(f"  数据集: {args.data}")
    print(f"  数据划分: {args.split} ({len(dataset)} images)")
    print(f"  输入尺寸: {args.imgsz}")
    print(f"  只读遍历: {args.passes} passes")
    print("  训练/反传/优化器/checkpoint: 全部禁用")
    print(f"  长边分桶: {CONFIG['long_edge_labels']}")
    print(f"  输出目录: {save_dir}")
    print("=" * 80)

    with torch.inference_mode():
        for pass_index in range(args.passes):
            if hasattr(loader, "reset"):
                loader.reset()
            for batch_index, batch in enumerate(loader, start=1):
                try:
                    images = batch["img"].to(device, non_blocking=True).float().div_(255.0)
                    outputs = core_model(images)
                    _collect_stats(criterion, _raw_predictions(outputs), batch)
                except Exception as error:
                    if not args.allow_stat_skip:
                        raise RuntimeError(
                            f"H1/H2统计失败: pass={pass_index + 1}, batch={batch_index}: {error}"
                        ) from error
                    print(f"[H1/H2 统计] 已跳过 pass={pass_index + 1}, batch={batch_index}: {error}")
            print(f"[*] 完成只读 pass {pass_index + 1}/{args.passes}")

    if _collector.batch_count == 0:
        raise RuntimeError("没有成功收集任何batch，拒绝生成空的H1/H2报告。")
    state_after = _state_tensor_sha256(core_model)
    checkpoint_after = _file_sha256(args.model)
    if state_after != state_before:
        raise RuntimeError("冻结统计前后模型参数或buffer发生变化，拒绝输出证据。")
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("统计前后输入checkpoint文件发生变化，拒绝输出证据。")

    strides = metadata["strides"]
    if not strides or any(float(value) <= 0 for value in strides):
        raise RuntimeError(f"无法获取有效stride: {strides}")
    protocol = {
        "mode": "frozen_eval_assigner_statistics",
        "split": args.split,
        "passes": args.passes,
        "dataset_images": len(dataset),
        "model_training": False,
        "gradient_enabled": False,
        "optimization_steps": 0,
        "checkpoint_saving": False,
        "validation_callback": False,
        "augmentation": False,
        "model_state_unchanged": True,
    }
    write_run_manifest(
        save_dir,
        args,
        metadata,
        int(data["nc"]),
        dataset_images=len(dataset),
        model_state_sha256_before=state_before,
        model_state_sha256_after=state_after,
        checkpoint_sha256_before=checkpoint_before,
        checkpoint_sha256_after=checkpoint_after,
    )
    _collector.report(
        reg_max=metadata["reg_max"],
        strides=strides,
        save_dir=save_dir,
        protocol=protocol,
    )

    print("\n固定权重只读统计完成！")


if __name__ == "__main__":
    main()
