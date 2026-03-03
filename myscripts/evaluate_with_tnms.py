"""
T-NMS 后处理评估脚本

功能：
    1. 读取导出的预测结果和 GT 标注
    2. 对每张图的预测框应用 T-NMS 融合
    3. 计算融合前后的 mAP@0.5 和 mAP@0.5:0.95
    4. 输出对比表格和统计信息

使用方法：
    python evaluate_with_tnms.py \
        --pred-dir ./tnms_data/predictions \
        --gt-dir ./tnms_data/ground_truths \
        --tau-theta 10 \
        --tau-perp 8 \
        --tau-gap 20 \
        --output-dir ./tnms_results
"""

import argparse
import math
import os
import sys
from pathlib import Path
os.environ["OMP_NUM_THREADS"] = "8"  # 修复 libgomp 警告
import numpy as np
from tqdm import tqdm

# 将项目根目录和脚本目录加入 sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from tnms import (
    topology_nms,
    topology_nms_batch,
    load_obbs_from_txt,
    load_gt_from_txt,
    save_obbs_to_txt,
    DEFAULT_TAU_THETA,
    DEFAULT_TAU_PERP,
    DEFAULT_TAU_GAP,
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="T-NMS 后处理评估")
    parser.add_argument(
        "--pred-dir",
        type=str,
        required=True,
        help="预测结果目录（txt 文件，格式: x y w h theta conf class_id）",
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        required=True,
        help="GT 标注目录（txt 文件，格式: x y w h theta class_id）",
    )
    parser.add_argument(
        "--tau-theta",
        type=float,
        default=DEFAULT_TAU_THETA,
        help=f"角度容忍阈值（度，默认: {DEFAULT_TAU_THETA}）",
    )
    parser.add_argument(
        "--tau-perp",
        type=float,
        default=DEFAULT_TAU_PERP,
        help=f"法向偏移容忍阈值（像素，默认: {DEFAULT_TAU_PERP}）",
    )
    parser.add_argument(
        "--tau-gap",
        type=float,
        default=DEFAULT_TAU_GAP,
        help=f"轴向断点间隙容忍（像素，默认: {DEFAULT_TAU_GAP}）",
    )
    parser.add_argument(
        "--iou-thresholds",
        type=float,
        nargs="+",
        default=None,
        help="IoU 阈值列表（默认: 0.5:0.05:0.95）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./tnms_results",
        help="结果输出目录（默认: ./tnms_results）",
    )
    parser.add_argument(
        "--save-merged",
        action="store_true",
        help="是否保存融合后的预测结果到 txt",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="是否打印详细日志",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="图像尺寸，用于将归一化坐标缩放到像素空间计算 ProbIoU（默认: 1024）",
    )
    return parser.parse_args()


# ======================== mAP 计算 ========================

def match_predictions_to_gt(pred_obbs, gt_obbs, iouv, imgsz=1024):
    """
    将预测框与 GT 框进行匹配。
    使用 ultralytics 内部的匹配逻辑。

    注意: 导出的坐标为归一化格式 (0~1)，ProbIoU 的 eps 保护项在极小尺度下
    会主导协方差矩阵的计算，导致 ProbIoU 值严重偏高。因此必须先将坐标
    缩放到像素空间再计算 ProbIoU，与 YOLO 内部验证保持一致。

    Args:
        pred_obbs: 预测框 (N, 7)，[x, y, w, h, theta, conf, class_id]
        gt_obbs: GT 框 (M, 6)，[x, y, w, h, theta, class_id]
        iouv: IoU 阈值向量 (10,)
        imgsz: 图像尺寸，用于将归一化坐标缩放到像素空间

    Returns:
        correct: (N, 10) bool 数组，标记每个预测框在各个阈值下是否匹配
    """
    import torch
    from ultralytics.utils.metrics import batch_probiou
    
    n_pred = len(pred_obbs)
    if n_pred == 0:
        return np.zeros((0, len(iouv)), dtype=bool)
    if len(gt_obbs) == 0:
        return np.zeros((n_pred, len(iouv)), dtype=bool)

    # 提取信息并缩放到像素坐标空间
    # 归一化坐标 (0~1) → 像素坐标 (0~imgsz)，仅缩放 x, y, w, h，角度 theta 不变
    pred_bboxes = torch.tensor(pred_obbs[:, :5], dtype=torch.float32)
    pred_bboxes[:, :4] *= imgsz  # x, y, w, h 缩放到像素空间
    # 角度归一化到 [0, π)：YOLO 内部使用 [0, π)，但 arctan2 返回 [-π, π]
    # 这里用 mod(π) 统一角度范围，避免预测和GT角度差 π 导致 ProbIoU 失配
    pred_bboxes[:, 4] = pred_bboxes[:, 4] % math.pi
    pred_cls = torch.tensor(pred_obbs[:, 6], dtype=torch.float32)
    
    gt_bboxes = torch.tensor(gt_obbs[:, :5], dtype=torch.float32)
    gt_bboxes[:, :4] *= imgsz  # x, y, w, h 缩放到像素空间
    gt_bboxes[:, 4] = gt_bboxes[:, 4] % math.pi  # 同样归一化角度
    gt_cls = torch.tensor(gt_obbs[:, 5], dtype=torch.float32)

    # 计算 ProbIoU (M, N)
    iou = batch_probiou(gt_bboxes, pred_bboxes)
    
    # 构建 correct 矩阵 (N, 10)
    correct = np.zeros((n_pred, len(iouv)), dtype=bool)
    correct_class = (gt_cls[:, None] == pred_cls) # (M, N)
    iou = iou * correct_class # 过滤掉类别不匹配的
    iou = iou.cpu().numpy()
    
    iouv_np = iouv.cpu().numpy()
    for i, threshold in enumerate(iouv_np):
        matches = np.nonzero(iou >= threshold)
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
            
    return correct


def evaluate_map(pred_dict, gt_dict, iou_thresholds=None, verbose=False, imgsz=1024):
    """
    计算 mAP@各个 IoU 阈值。使用 ultralytics 官方评估函数。

    Args:
        pred_dict: dict，键为图片名，值为预测框 numpy array (N, 7)
        gt_dict: dict，键为图片名，值为 GT 框 numpy array (M, 6)
        iou_thresholds: IoU 阈值列表，实际上未使用，强制使用0.5-0.95
        verbose: 是否打印详细信息
        imgsz: 图像尺寸，用于将归一化坐标缩放到像素空间

    Returns:
        results: dict，包含各阈值下的 mAP 和详细指标
    """
    import torch
    from ultralytics.utils.metrics import ap_per_class
    
    iouv = torch.linspace(0.5, 0.95, 10)
    
    all_tp = []
    all_conf = []
    all_pred_cls = []
    all_gt_cls = []

    all_images = set(list(pred_dict.keys()) + list(gt_dict.keys()))

    total_gt = 0
    total_pred = 0

    for img_name in all_images:
        preds = pred_dict.get(img_name, np.empty((0, 7)))
        gts = gt_dict.get(img_name, np.empty((0, 6)))

        if isinstance(preds, list):
            preds = np.array(preds) if preds else np.empty((0, 7))
        if isinstance(gts, list):
            gts = np.array(gts) if gts else np.empty((0, 6))

        total_gt += len(gts)
        total_pred += len(preds)

        if len(gts) > 0:
            all_gt_cls.extend(gts[:, 5].tolist())

        if len(preds) > 0:
            tp = match_predictions_to_gt(preds, gts, iouv, imgsz=imgsz)
            all_tp.append(tp)
            all_conf.extend(preds[:, 5].tolist())
            all_pred_cls.extend(preds[:, 6].tolist())

    if all_tp:
        tp = np.concatenate(all_tp, axis=0) # (N, 10)
    else:
        tp = np.zeros((0, 10), dtype=bool)
        
    conf = np.array(all_conf)
    pred_cls = np.array(all_pred_cls)
    target_cls = np.array(all_gt_cls)

    map50 = 0.0
    map50_95 = 0.0

    if len(tp) > 0 and len(target_cls) > 0:
        # 调用 ultralytics 官方 ap 计算核心
        # ap_per_class 返回 12 个值
        (
            tp_res, fp_res, p, r, f1, ap,
            ap_class, p_curve, r_curve, f1_curve, x, prec_values
        ) = ap_per_class(
            tp, conf, pred_cls, target_cls, plot=False, prefix="Box"
        )
        if len(ap) > 0:
            map50 = ap[:, 0].mean() if ap.ndim > 1 else 0.0
            map50_95 = ap.mean()

    results = {
        "total_predictions": total_pred,
        "total_gt": total_gt,
        "mAP50": map50,
        "mAP50_95": map50_95,
        "iou_thresholds": iouv.numpy().tolist()
    }
    
    return results


# ======================== 主流程 ========================

def main():
    args = parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  T-NMS 后处理评估")
    print("=" * 60)
    print(f"  预测结果目录: {pred_dir}")
    print(f"  GT 标注目录: {gt_dir}")
    print(f"  T-NMS 参数: τ_θ={args.tau_theta}°, τ_⊥={args.tau_perp}px, τ_gap={args.tau_gap}px")
    print(f"  输出目录: {output_dir}")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/4] 加载预测结果和 GT 标注...")
    pred_dict = {}
    gt_dict = {}

    pred_files = sorted(pred_dir.glob("*.txt"))
    gt_files = sorted(gt_dir.glob("*.txt"))

    for f in tqdm(pred_files, desc="  加载预测结果"):
        pred_dict[f.stem] = load_obbs_from_txt(f)

    for f in tqdm(gt_files, desc="  加载 GT 标注"):
        gt_dict[f.stem] = load_gt_from_txt(f)

    total_pred_boxes = sum(len(v) for v in pred_dict.values())
    total_gt_boxes = sum(len(v) for v in gt_dict.values())
    print(f"  已加载 {len(pred_dict)} 张图的 {total_pred_boxes} 个预测框")
    print(f"  已加载 {len(gt_dict)} 张图的 {total_gt_boxes} 个 GT 框")

    # 2. 计算融合前的 mAP
    print("\n[2/4] 计算融合前 mAP...")
    iou_thresholds = np.array(args.iou_thresholds) if args.iou_thresholds else None
    results_before = evaluate_map(pred_dict, gt_dict, iou_thresholds, verbose=args.verbose, imgsz=args.imgsz)
    print(f"  融合前 mAP@0.5:    {results_before['mAP50']:.4f}")
    print(f"  融合前 mAP@0.5:0.95: {results_before['mAP50_95']:.4f}")

    # 3. 应用 T-NMS
    print("\n[3/4] 应用 T-NMS 融合...")
    merged_pred_dict, merge_stats = topology_nms_batch(
        pred_dict,
        tau_theta=args.tau_theta,
        tau_perp=args.tau_perp,
        tau_gap=args.tau_gap,
        verbose=args.verbose,
    )
    print(f"  融合统计: {merge_stats['total_boxes_before']} → {merge_stats['total_boxes_after']} 个框 "
          f"(减少 {merge_stats['total_merged']} 个)")

    # 保存融合后的预测结果
    if args.save_merged:
        merged_dir = output_dir / "merged_predictions"
        merged_dir.mkdir(parents=True, exist_ok=True)
        for img_name, obbs in merged_pred_dict.items():
            save_obbs_to_txt(obbs, merged_dir / f"{img_name}.txt")
        print(f"  融合后预测结果已保存到: {merged_dir}")

    # 4. 计算融合后的 mAP
    print("\n[4/4] 计算融合后 mAP...")
    results_after = evaluate_map(merged_pred_dict, gt_dict, iou_thresholds, verbose=args.verbose, imgsz=args.imgsz)
    print(f"  融合后 mAP@0.5:    {results_after['mAP50']:.4f}")
    print(f"  融合后 mAP@0.5:0.95: {results_after['mAP50_95']:.4f}")

    # 5. 输出对比报告
    print("\n" + "=" * 60)
    print("  T-NMS 前后对比报告")
    print("=" * 60)
    print(f"  {'指标':<20} {'融合前':>10} {'融合后':>10} {'变化':>10}")
    print(f"  {'-'*50}")

    map50_before = results_before["mAP50"]
    map50_after = results_after["mAP50"]
    map50_diff = map50_after - map50_before

    map50_95_before = results_before["mAP50_95"]
    map50_95_after = results_after["mAP50_95"]
    map50_95_diff = map50_95_after - map50_95_before

    pred_before = results_before["total_predictions"]
    pred_after = results_after["total_predictions"]
    pred_diff = pred_after - pred_before

    print(f"  {'预测框数量':<20} {pred_before:>10} {pred_after:>10} {pred_diff:>+10}")
    print(f"  {'mAP@0.5':<20} {map50_before:>10.4f} {map50_after:>10.4f} {map50_diff:>+10.4f}")
    print(f"  {'mAP@0.5:0.95':<20} {map50_95_before:>10.4f} {map50_95_after:>10.4f} {map50_95_diff:>+10.4f}")

    print("=" * 60)

    # 6. 保存报告到文件
    report_file = output_dir / "tnms_evaluation_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("T-NMS 后处理评估报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"T-NMS 参数:\n")
        f.write(f"  τ_θ (角度容忍): {args.tau_theta}°\n")
        f.write(f"  τ_⊥ (法向偏移): {args.tau_perp}px\n")
        f.write(f"  τ_gap (轴向间隙): {args.tau_gap}px\n\n")
        f.write(f"融合统计:\n")
        f.write(f"  融合前总框数: {merge_stats['total_boxes_before']}\n")
        f.write(f"  融合后总框数: {merge_stats['total_boxes_after']}\n")
        f.write(f"  减少框数: {merge_stats['total_merged']}\n\n")
        f.write(f"mAP 对比:\n")
        f.write(f"  mAP@0.5     融合前: {map50_before:.4f}  融合后: {map50_after:.4f}  变化: {map50_diff:+.4f}\n")
        f.write(f"  mAP@0.5:0.95 融合前: {map50_95_before:.4f}  融合后: {map50_95_after:.4f}  变化: {map50_95_diff:+.4f}\n")

    print(f"\n  评估报告已保存到: {report_file}")


if __name__ == "__main__":
    main()
