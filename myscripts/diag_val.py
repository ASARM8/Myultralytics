"""
OBB 模型诊断验证脚本
功能：
  诊断 1 — 按 IoU 阈值拆分 AP（AP50 / AP60 / AP70 / AP75 / AP80 / AP90 / AP95）
  诊断 2 — 按目标长度 & 长宽比分桶统计 AP
  诊断 3 — 定位误差拆分（横向偏移 / 角度误差 / 宽度误差 / 长度误差）
  诊断 4 — DFL 回归距离饱和分析（四边距离分布 / bin 命中频率 / 末端堆积 / 长短边对比）
  诊断 5 — GT 扰动 IoU 敏感性曲线（中心偏移 / 角度扰动 / 宽度扰动 → IoU 下降）
输出：诊断 Markdown 报告

使用方法:
    修改下方 CONFIG 后运行: python myscripts/diag_val.py
"""

import os
import sys
import glob
import yaml
import numpy as np
import torch
import cv2
from pathlib import Path
from collections import defaultdict

# 添加项目根目录
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ultralytics import YOLO
from ultralytics.utils.ops import xyxyxyxy2xywhr
from ultralytics.utils.metrics import batch_probiou

# ========================== 配置 ==========================
CONFIG = {
    # ---------- 模型 ----------
    "model": "/root/autodl-tmp/work-dirs/yolo11_obb-ca/weights/best.pt",
    "model_label": "CA (reg_max=32)",  # 报告中显示的模型名

    # ---------- 数据集 ----------
    "data": "/root/autodl-tmp/dataset/TTPLA-1024/dataset.yaml",

    # ---------- 推理参数 ----------
    "conf": 0.001,              # 极低置信度以获得完整 P-R 曲线
    "iou_nms": 0.7,             # NMS IoU 阈值
    "imgsz": 1024,
    "device": 0,
    "max_det": 1000,            # 每张图最大检测数

    # ---------- 输出 ----------
    "output_dir": "/root/autodl-tmp/work-dirs/diag_val_output",
}

# ========================== 分析参数 ==========================
# 诊断 1：AP 拆分的 IoU 阈值
IOU_THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95]

# 诊断 2：分桶定义
LENGTH_BUCKETS = [
    (0, 128, "<128"),
    (128, 256, "128-256"),
    (256, 512, "256-512"),
    (512, float("inf"), ">512"),
]
AR_BUCKETS = [
    (0, 10, "<10"),
    (10, 30, "10-30"),
    (30, 60, "30-60"),
    (60, float("inf"), ">60"),
]

# 诊断 3：定位误差匹配阈值（使用较宽松的 IoU 获取更多匹配对）
ERROR_MATCH_IOU = 0.50

# 诊断 4：DFL 饱和分析的 stride 配置
MODEL_STRIDES = [8, 16, 32]  # P3, P4, P5

# 诊断 5：GT 扰动参数
PERTURB_CENTER_PX = [1, 2, 3, 4, 5]      # 中心横向偏移（像素）
PERTURB_ANGLE_DEG = [0.5, 1.0, 1.5, 2.0, 3.0]  # 角度扰动（度）
PERTURB_WIDTH_PX = [1, 2, 3, 4]           # 宽度扰动（像素）


# ========================== 工具函数 ==========================

def load_dataset_paths(data_yaml):
    """从 dataset.yaml 加载验证集图片和标注路径。"""
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # 支持相对路径
    data_root = data.get("path", os.path.dirname(data_yaml))
    val_path = data.get("val", "images/val")
    if not os.path.isabs(val_path):
        val_path = os.path.join(data_root, val_path)

    # 图片列表
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    image_files = []
    for ext in exts:
        image_files.extend(glob.glob(os.path.join(val_path, ext)))
    image_files.sort()

    # 标注目录：images → labels
    label_dir = val_path.replace("images", "labels")

    return image_files, label_dir, data.get("names", {})


def load_gt_for_image(label_path, img_w, img_h):
    """加载单张图片的 GT 标注，返回 (cls_array, xywhr_array)。

    标注格式：class x1 y1 x2 y2 x3 y3 x4 y4（归一化坐标）
    返回 xywhr 为像素坐标。
    """
    cls_list = []
    corners_list = []

    if not os.path.exists(label_path):
        return np.zeros((0,), dtype=np.int32), np.zeros((0, 5), dtype=np.float32)

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            cls_list.append(int(parts[0]))
            coords = list(map(float, parts[1:9]))
            # 归一化 → 像素
            pts = np.array(coords, dtype=np.float32).reshape(4, 2)
            pts[:, 0] *= img_w
            pts[:, 1] *= img_h
            corners_list.append(pts)

    if not cls_list:
        return np.zeros((0,), dtype=np.int32), np.zeros((0, 5), dtype=np.float32)

    cls_arr = np.array(cls_list, dtype=np.int32)
    # 使用 Ultralytics 官方函数转换 (N, 4, 2) → (N, 5) xywhr
    corners_flat = np.array(corners_list).reshape(-1, 8)  # (N, 8)
    xywhr = xyxyxyxy2xywhr(corners_flat)  # (N, 5)
    if isinstance(xywhr, torch.Tensor):
        xywhr = xywhr.numpy()

    return cls_arr, xywhr.astype(np.float32)


def extract_predictions(results):
    """从 Ultralytics 推理结果中提取 OBB 预测。

    Returns:
        cls: (N,) int
        conf: (N,) float
        xywhr: (N, 5) float, 像素坐标
    """
    if results is None or results[0].obb is None or len(results[0].obb) == 0:
        return (np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, 5), dtype=np.float32))

    obb = results[0].obb
    cls = obb.cls.cpu().numpy().astype(np.int32)
    conf = obb.conf.cpu().numpy().astype(np.float32)
    xywhr = obb.xywhr.cpu().numpy().astype(np.float32)
    return cls, conf, xywhr


def match_at_thresholds(iou_matrix, gt_cls, pred_cls, thresholds):
    """对单张图片，在多个 IoU 阈值下做贪心匹配。

    Args:
        iou_matrix: (N_gt, N_pred) IoU 矩阵
        gt_cls: (N_gt,) GT 类别
        pred_cls: (N_pred,) 预测类别
        thresholds: list of float

    Returns:
        tp: (N_pred, len(thresholds)) bool，每个预测在各阈值下是否为 TP
        matched_gt_idx: (N_pred,) 在最低阈值下匹配的 GT 索引（未匹配为 -1）
    """
    n_pred = len(pred_cls)
    n_gt = len(gt_cls)
    n_th = len(thresholds)
    tp = np.zeros((n_pred, n_th), dtype=bool)
    matched_gt_idx = np.full(n_pred, -1, dtype=np.int32)

    if n_gt == 0 or n_pred == 0:
        return tp, matched_gt_idx

    # 类别匹配掩码
    cls_match = gt_cls[:, None] == pred_cls[None, :]  # (N_gt, N_pred)
    iou_masked = iou_matrix * cls_match  # 类别不匹配的 IoU 置零

    for ti, th in enumerate(thresholds):
        matched_gt_set = set()
        # 找到所有 IoU >= threshold 的 (gt, pred) 对
        matches = np.argwhere(iou_masked >= th)  # (K, 2)
        if len(matches) == 0:
            continue

        # 按 IoU 降序排列
        match_ious = iou_masked[matches[:, 0], matches[:, 1]]
        order = match_ious.argsort()[::-1]
        matches = matches[order]

        matched_pred_set = set()
        for gt_i, pred_i in matches:
            if gt_i in matched_gt_set or pred_i in matched_pred_set:
                continue
            tp[pred_i, ti] = True
            matched_gt_set.add(gt_i)
            matched_pred_set.add(pred_i)

            # 记录最低阈值（第一个）的匹配关系，用于误差分析
            if ti == 0 and matched_gt_idx[pred_i] == -1:
                matched_gt_idx[pred_i] = gt_i

    return tp, matched_gt_idx


def compute_ap_from_tp(tp_col, conf, n_gt, eps=1e-16):
    """根据 TP 列和置信度计算 AP（COCO 101 点插值）。

    Args:
        tp_col: (N,) bool 数组，是否为 TP
        conf: (N,) 置信度
        n_gt: GT 总数

    Returns:
        ap: float
        precision: float (在最佳 F1 点)
        recall: float (在最佳 F1 点)
    """
    if n_gt == 0:
        return 0.0, 0.0, 0.0

    # 按置信度降序排列
    i = np.argsort(-conf)
    tp_sorted = tp_col[i].astype(np.float64)

    # 累积 TP 和 FP
    tpc = tp_sorted.cumsum()
    fpc = (1 - tp_sorted).cumsum()

    recall = tpc / (n_gt + eps)
    precision = tpc / (tpc + fpc + eps)

    # AP（101 点插值）
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)

    # 最佳 F1 点
    f1 = 2 * precision * recall / (precision + recall + eps)
    best_i = f1.argmax()
    return float(ap), float(precision[best_i]), float(recall[best_i])


def gt_properties(xywhr):
    """计算 GT 框的几何属性。

    Args:
        xywhr: (5,) [cx, cy, w, h, angle]

    Returns:
        length: 长边像素长度
        width: 短边像素长度
        aspect_ratio: 长宽比
    """
    w, h = xywhr[2], xywhr[3]
    length = max(w, h)
    width = min(w, h)
    ar = length / max(width, 1e-6)
    return float(length), float(width), float(ar)


def compute_loc_errors(gt_box, pred_box):
    """计算单对匹配框的定位误差分解。

    Args:
        gt_box: (5,) xywhr
        pred_box: (5,) xywhr

    Returns:
        dict with:
          perp_offset: 横向偏移（垂直于 GT 长边方向）
          para_offset: 纵向偏移（沿 GT 长边方向）
          angle_error: 角度误差（度）
          width_error: 宽度误差（像素）
          length_error: 长度误差（像素）
          width_error_rel: 宽度相对误差 (%)
          length_error_rel: 长度相对误差 (%)
    """
    gt_cx, gt_cy, gt_w, gt_h, gt_a = gt_box
    pd_cx, pd_cy, pd_w, pd_h, pd_a = pred_box

    # 长/短边（xyxyxyxy2xywhr 保证 w >= h）
    gt_length, gt_width = gt_w, gt_h
    pd_length, pd_width = pd_w, pd_h

    # 中心点偏移
    dx = pd_cx - gt_cx
    dy = pd_cy - gt_cy

    # 沿 GT 长边方向分解
    cos_a = np.cos(gt_a)
    sin_a = np.sin(gt_a)
    para_offset = abs(dx * cos_a + dy * sin_a)   # 沿长边
    perp_offset = abs(-dx * sin_a + dy * cos_a)  # 垂直于长边

    # 角度误差（考虑 π 周期性）
    angle_diff = (pd_a - gt_a) % np.pi
    if angle_diff > np.pi / 2:
        angle_diff = np.pi - angle_diff
    angle_error_deg = np.degrees(angle_diff)

    # 尺寸误差
    width_error = abs(pd_width - gt_width)
    length_error = abs(pd_length - gt_length)
    width_error_rel = width_error / max(gt_width, 1e-6) * 100
    length_error_rel = length_error / max(gt_length, 1e-6) * 100

    return {
        "perp_offset": float(perp_offset),
        "para_offset": float(para_offset),
        "angle_error": float(angle_error_deg),
        "width_error": float(width_error),
        "length_error": float(length_error),
        "width_error_rel": float(width_error_rel),
        "length_error_rel": float(length_error_rel),
        "gt_length": float(gt_length),
        "gt_width": float(gt_width),
        "gt_ar": float(gt_length / max(gt_width, 1e-6)),
    }


def bucket_name(value, buckets):
    """返回 value 所属桶的名称。"""
    for lo, hi, name in buckets:
        if lo <= value < hi:
            return name
    return buckets[-1][2]


# ========================== 主逻辑 ==========================

def collect_data(config):
    """收集所有图片的预测和 GT 数据。"""
    print("=" * 60)
    print("  OBB 诊断验证脚本")
    print("=" * 60)

    # 加载数据集路径
    image_files, label_dir, class_names = load_dataset_paths(config["data"])
    print(f"  数据集: {config['data']}")
    print(f"  验证图片数: {len(image_files)}")
    print(f"  标注目录: {label_dir}")
    print(f"  类别: {class_names}")

    # 加载模型
    print(f"\n  加载模型: {config['model']}")
    model = YOLO(config["model"])

    # 提取模型信息（reg_max / strides）用于诊断 4
    det_head = model.model.model[-1]  # 最后一层是检测头
    reg_max = int(det_head.reg_max)
    strides = [int(s) for s in det_head.stride.tolist()]
    model_info = {"reg_max": reg_max, "strides": strides}
    print(f"  reg_max = {reg_max}, strides = {strides}")

    all_data = []  # 每张图一个 dict

    for idx, img_path in enumerate(image_files):
        img_name = os.path.basename(img_path)
        stem = os.path.splitext(img_name)[0]

        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(image_files)}] 处理: {img_name}")

        # 读取图片尺寸
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # 加载 GT
        label_path = os.path.join(label_dir, stem + ".txt")
        gt_cls, gt_xywhr = load_gt_for_image(label_path, img_w, img_h)

        # 预测
        results = model.predict(
            img_path,
            conf=config["conf"],
            iou=config["iou_nms"],
            imgsz=config["imgsz"],
            device=config["device"],
            max_det=config["max_det"],
            verbose=False,
        )
        pred_cls, pred_conf, pred_xywhr = extract_predictions(results)

        # 计算 IoU 矩阵
        n_gt = len(gt_cls)
        n_pred = len(pred_cls)
        if n_gt > 0 and n_pred > 0:
            gt_t = torch.from_numpy(gt_xywhr).float()
            pd_t = torch.from_numpy(pred_xywhr).float()
            iou_matrix = batch_probiou(gt_t, pd_t).numpy()
            iou_matrix = np.clip(iou_matrix, 0, 1)
        else:
            iou_matrix = np.zeros((n_gt, n_pred), dtype=np.float32)

        # 在多个 IoU 阈值下匹配
        tp, matched_gt_idx = match_at_thresholds(
            iou_matrix, gt_cls, pred_cls, IOU_THRESHOLDS
        )

        # GT 几何属性
        gt_props = []
        for i in range(n_gt):
            length, width, ar = gt_properties(gt_xywhr[i])
            gt_props.append({"length": length, "width": width, "ar": ar})

        all_data.append({
            "image": img_name,
            "gt_cls": gt_cls,
            "gt_xywhr": gt_xywhr,
            "gt_props": gt_props,
            "pred_cls": pred_cls,
            "pred_conf": pred_conf,
            "pred_xywhr": pred_xywhr,
            "tp": tp,
            "matched_gt_idx": matched_gt_idx,
            "iou_matrix": iou_matrix,
        })

    print(f"\n  数据收集完成: {len(all_data)} 张图片")
    return all_data, class_names, model_info


def diag1_ap_by_threshold(all_data):
    """诊断 1：按 IoU 阈值拆分 AP。"""
    print("\n--- 诊断 1：按 IoU 阈值拆分 AP ---")

    # 汇总所有图片的 tp, conf
    all_tp = []
    all_conf = []
    total_gt = 0

    for d in all_data:
        all_tp.append(d["tp"])
        all_conf.append(d["pred_conf"])
        total_gt += len(d["gt_cls"])

    if not all_tp:
        return {}

    all_tp = np.concatenate(all_tp, axis=0)      # (N_total_pred, n_thresholds)
    all_conf = np.concatenate(all_conf, axis=0)   # (N_total_pred,)

    results = {}
    for ti, th in enumerate(IOU_THRESHOLDS):
        ap, p, r = compute_ap_from_tp(all_tp[:, ti], all_conf, total_gt)
        results[th] = {"ap": ap, "precision": p, "recall": r}
        print(f"  AP{int(th*100):02d} = {ap:.4f}  (P={p:.3f}, R={r:.3f})")

    return results


def diag2_ap_by_bucket(all_data):
    """诊断 2：按目标长度和长宽比分桶统计 AP。"""
    print("\n--- 诊断 2：按目标长度/长宽比分桶统计 AP ---")

    # 收集每个 GT 的属性，并标记每张图的 GT 索引范围
    # 我们需要重新做分桶匹配：对每个桶，只考虑该桶内的 GT

    # 对每张图，建立全局 GT 索引
    gt_global_info = []  # (image_idx, local_gt_idx, length, ar)
    for img_i, d in enumerate(all_data):
        for gt_i, props in enumerate(d["gt_props"]):
            gt_global_info.append((img_i, gt_i, props["length"], props["ar"]))

    def compute_bucket_ap(filter_fn, iou_th_idx):
        """对满足 filter_fn 的 GT 子集，重新匹配并计算 AP。"""
        all_tp_bucket = []
        all_conf_bucket = []
        n_gt_bucket = 0

        for img_i, d in enumerate(all_data):
            gt_cls = d["gt_cls"]
            pred_cls = d["pred_cls"]
            pred_conf = d["pred_conf"]
            iou_matrix = d["iou_matrix"]
            n_pred = len(pred_cls)

            # 筛选该桶内的 GT
            bucket_gt_mask = np.array([
                filter_fn(d["gt_props"][gi]) for gi in range(len(gt_cls))
            ], dtype=bool) if len(gt_cls) > 0 else np.array([], dtype=bool)

            n_gt_in_bucket = bucket_gt_mask.sum()
            n_gt_bucket += n_gt_in_bucket

            if n_pred == 0:
                continue

            if n_gt_in_bucket == 0:
                # 所有预测都是 FP（对于这个桶）
                # 但这些预测可能匹配了其他桶的 GT，不应算作 FP
                # 标准做法：只有没匹配到任何 GT 的预测才是 FP
                # 简化处理：跳过该图的预测（不影响这个桶的 AP）
                continue

            # 用该桶的 GT 子集做匹配
            bucket_gt_indices = np.where(bucket_gt_mask)[0]
            sub_iou = iou_matrix[bucket_gt_indices, :]  # (n_gt_bucket, n_pred)
            sub_gt_cls = gt_cls[bucket_gt_indices]

            # 单阈值匹配
            th = IOU_THRESHOLDS[iou_th_idx]
            cls_match = sub_gt_cls[:, None] == pred_cls[None, :]
            iou_masked = sub_iou * cls_match

            tp_pred = np.zeros(n_pred, dtype=bool)
            matches = np.argwhere(iou_masked >= th)
            if len(matches) > 0:
                match_ious = iou_masked[matches[:, 0], matches[:, 1]]
                order = match_ious.argsort()[::-1]
                matches = matches[order]
                matched_gt_set = set()
                matched_pred_set = set()
                for g, p in matches:
                    if g in matched_gt_set or p in matched_pred_set:
                        continue
                    tp_pred[p] = True
                    matched_gt_set.add(g)
                    matched_pred_set.add(p)

            # 只保留与该桶相关的预测：匹配到桶内 GT 的是 TP，
            # 没匹配到任何 GT 的是 FP，匹配到桶外 GT 的忽略
            # 为简单起见：保留所有预测，TP/FP 按上面标记
            all_tp_bucket.append(tp_pred)
            all_conf_bucket.append(pred_conf)

        if n_gt_bucket == 0:
            return 0.0, 0, 0.0, 0.0

        if not all_tp_bucket:
            return 0.0, n_gt_bucket, 0.0, 0.0

        tp_arr = np.concatenate(all_tp_bucket)
        conf_arr = np.concatenate(all_conf_bucket)
        ap, p, r = compute_ap_from_tp(tp_arr, conf_arr, n_gt_bucket)
        return ap, n_gt_bucket, p, r

    # 按长度分桶，在 AP50 和 AP75 下计算
    print("\n  [按长度分桶]")
    length_results = {}
    for lo, hi, name in LENGTH_BUCKETS:
        filter_fn = lambda props, lo=lo, hi=hi: lo <= props["length"] < hi
        row = {"name": name, "n_gt": 0}
        for th_label, th_idx in [("AP50", 0), ("AP75", 5)]:
            ap, n_gt, p, r = compute_bucket_ap(filter_fn, th_idx)
            row[th_label] = ap
            row[f"{th_label}_P"] = p
            row[f"{th_label}_R"] = r
            row["n_gt"] = n_gt
        length_results[name] = row
        print(f"    {name:>8s}: n_gt={row['n_gt']:5d}  AP50={row['AP50']:.4f}  AP75={row['AP75']:.4f}")

    # 按长宽比分桶
    print("\n  [按长宽比分桶]")
    ar_results = {}
    for lo, hi, name in AR_BUCKETS:
        filter_fn = lambda props, lo=lo, hi=hi: lo <= props["ar"] < hi
        row = {"name": name, "n_gt": 0}
        for th_label, th_idx in [("AP50", 0), ("AP75", 5)]:
            ap, n_gt, p, r = compute_bucket_ap(filter_fn, th_idx)
            row[th_label] = ap
            row[f"{th_label}_P"] = p
            row[f"{th_label}_R"] = r
            row["n_gt"] = n_gt
        ar_results[name] = row
        print(f"    AR {name:>5s}: n_gt={row['n_gt']:5d}  AP50={row['AP50']:.4f}  AP75={row['AP75']:.4f}")

    # 交叉分桶：长度 × AP 梯度
    print("\n  [按长度分桶 × AP 梯度]")
    cross_results = {}
    for lo, hi, name in LENGTH_BUCKETS:
        filter_fn = lambda props, lo=lo, hi=hi: lo <= props["length"] < hi
        row = {"name": name}
        for th_label, th_idx in zip(
            ["AP50", "AP60", "AP70", "AP75", "AP80", "AP90", "AP95"],
            range(len(IOU_THRESHOLDS))
        ):
            ap, n_gt, _, _ = compute_bucket_ap(filter_fn, th_idx)
            row[th_label] = ap
            row["n_gt"] = n_gt
        cross_results[name] = row
        vals = " | ".join(f"{row.get(f'AP{int(t*100)}', 0):.3f}" for t in IOU_THRESHOLDS)
        print(f"    {name:>8s}: {vals}")

    return length_results, ar_results, cross_results


def diag3_loc_errors(all_data):
    """诊断 3：定位误差拆分。"""
    print("\n--- 诊断 3：定位误差拆分 ---")

    all_errors = []

    for d in all_data:
        pred_conf = d["pred_conf"]
        matched_gt_idx = d["matched_gt_idx"]
        gt_xywhr = d["gt_xywhr"]
        pred_xywhr = d["pred_xywhr"]

        for pi in range(len(pred_conf)):
            gi = matched_gt_idx[pi]
            if gi < 0:
                continue
            errors = compute_loc_errors(gt_xywhr[gi], pred_xywhr[pi])
            errors["conf"] = float(pred_conf[pi])
            all_errors.append(errors)

    if not all_errors:
        print("  没有匹配对，无法计算误差。")
        return {}, {}

    print(f"  匹配对数: {len(all_errors)}")

    # 汇总统计
    keys = ["perp_offset", "para_offset", "angle_error",
            "width_error", "length_error", "width_error_rel", "length_error_rel"]
    labels = {
        "perp_offset": "横向偏移 (px)",
        "para_offset": "纵向偏移 (px)",
        "angle_error": "角度误差 (°)",
        "width_error": "宽度误差 (px)",
        "length_error": "长度误差 (px)",
        "width_error_rel": "宽度相对误差 (%)",
        "length_error_rel": "长度相对误差 (%)",
    }

    overall_stats = {}
    for k in keys:
        vals = np.array([e[k] for e in all_errors])
        stats = {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "std": float(np.std(vals)),
            "P90": float(np.percentile(vals, 90)),
            "P95": float(np.percentile(vals, 95)),
        }
        overall_stats[k] = stats
        print(f"  {labels[k]:>20s}: "
              f"mean={stats['mean']:.2f}  median={stats['median']:.2f}  "
              f"P90={stats['P90']:.2f}  P95={stats['P95']:.2f}")

    # 按 GT 长度分桶统计误差
    print("\n  [按 GT 长度分桶的误差统计]")
    bucketed_errors = defaultdict(lambda: defaultdict(list))
    for e in all_errors:
        bname = bucket_name(e["gt_length"], LENGTH_BUCKETS)
        for k in keys:
            bucketed_errors[bname][k].append(e[k])

    bucket_stats = {}
    for bname in [b[2] for b in LENGTH_BUCKETS]:
        if bname not in bucketed_errors:
            continue
        n = len(bucketed_errors[bname]["perp_offset"])
        bucket_stats[bname] = {"n": n}
        for k in keys:
            vals = np.array(bucketed_errors[bname][k])
            bucket_stats[bname][k] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "P95": float(np.percentile(vals, 95)),
            }
        print(f"    {bname:>8s} (n={n:5d}): "
              f"横向={bucket_stats[bname]['perp_offset']['mean']:.1f}px  "
              f"角度={bucket_stats[bname]['angle_error']['mean']:.2f}°  "
              f"宽度={bucket_stats[bname]['width_error']['mean']:.1f}px  "
              f"长度={bucket_stats[bname]['length_error']['mean']:.1f}px")

    # 按长宽比分桶
    print("\n  [按长宽比分桶的误差统计]")
    bucketed_errors_ar = defaultdict(lambda: defaultdict(list))
    for e in all_errors:
        bname = bucket_name(e["gt_ar"], AR_BUCKETS)
        for k in keys:
            bucketed_errors_ar[bname][k].append(e[k])

    bucket_stats_ar = {}
    for bname in [b[2] for b in AR_BUCKETS]:
        if bname not in bucketed_errors_ar:
            continue
        n = len(bucketed_errors_ar[bname]["perp_offset"])
        bucket_stats_ar[bname] = {"n": n}
        for k in keys:
            vals = np.array(bucketed_errors_ar[bname][k])
            bucket_stats_ar[bname][k] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "P95": float(np.percentile(vals, 95)),
            }
        print(f"    AR {bname:>5s} (n={n:5d}): "
              f"横向={bucket_stats_ar[bname]['perp_offset']['mean']:.1f}px  "
              f"角度={bucket_stats_ar[bname]['angle_error']['mean']:.2f}°  "
              f"宽度={bucket_stats_ar[bname]['width_error']['mean']:.1f}px  "
              f"长度={bucket_stats_ar[bname]['length_error']['mean']:.1f}px")

    return overall_stats, bucket_stats, bucket_stats_ar


def diag4_dfl_saturation(all_data, model_info):
    """诊断 4：DFL 回归距离饱和分析。

    对每个 GT 框，计算其在各 stride 层上的最优 anchor（GT 中心）处的
    LTRB 目标回归距离（grid 单位），检查是否超出 reg_max - 1。
    """
    print("\n--- 诊断 4：DFL 回归距离饱和分析 ---")

    reg_max = model_info["reg_max"]
    strides = model_info["strides"]
    d_max = reg_max - 1
    print(f"  reg_max = {reg_max}, D_max = {d_max}")

    # 收集所有 GT 的距离数据
    # 对每个 GT，在其"最佳分配层"（面积最匹配的 stride）计算距离
    # 同时也统计在所有层上的分布
    all_long_dists = {s: [] for s in strides}   # 长边半距（grid）
    all_short_dists = {s: [] for s in strides}  # 短边半距（grid）
    all_max_dists = {s: [] for s in strides}    # max(长,短)
    gt_assigned_dists = []  # 每个 GT 在"最可能被分配层"的距离

    for d in all_data:
        gt_xywhr = d["gt_xywhr"]
        for i in range(len(d["gt_cls"])):
            w_px, h_px = gt_xywhr[i, 2], gt_xywhr[i, 3]  # w >= h (xyxyxyxy2xywhr 保证)
            gt_area = w_px * h_px

            for s in strides:
                d_long = w_px / (2 * s)    # 长边半距 (grid)
                d_short = h_px / (2 * s)   # 短边半距 (grid)
                d_max_val = max(d_long, d_short)
                all_long_dists[s].append(d_long)
                all_short_dists[s].append(d_short)
                all_max_dists[s].append(d_max_val)

            # 启发式"最可能分配层"：取使最大距离最接近 d_max/2 的层
            # 更简单：取 sqrt(area) 对应的 stride
            best_stride = strides[0]
            box_scale = np.sqrt(gt_area)
            for s in strides:
                if box_scale >= s * 4:  # 粗略：大目标分配到大 stride
                    best_stride = s
            d_long_best = w_px / (2 * best_stride)
            d_short_best = h_px / (2 * best_stride)
            gt_assigned_dists.append({
                "stride": best_stride,
                "d_long": d_long_best,
                "d_short": d_short_best,
                "d_max": max(d_long_best, d_short_best),
                "w_px": w_px,
                "h_px": h_px,
                "ar": w_px / max(h_px, 1e-6),
            })

    total_gt = len(gt_assigned_dists)
    print(f"  总 GT 数: {total_gt}")

    # 1. 各 stride 层的距离分布统计
    stride_stats = {}
    for s in strides:
        long_arr = np.array(all_long_dists[s])
        short_arr = np.array(all_short_dists[s])
        max_arr = np.array(all_max_dists[s])

        sat_long = (long_arr > d_max).sum()
        sat_short = (short_arr > d_max).sum()
        sat_any = (max_arr > d_max).sum()

        stride_stats[s] = {
            "long_mean": float(np.mean(long_arr)),
            "long_median": float(np.median(long_arr)),
            "long_P95": float(np.percentile(long_arr, 95)),
            "long_max": float(np.max(long_arr)),
            "short_mean": float(np.mean(short_arr)),
            "short_median": float(np.median(short_arr)),
            "short_P95": float(np.percentile(short_arr, 95)),
            "short_max": float(np.max(short_arr)),
            "sat_long": int(sat_long),
            "sat_short": int(sat_short),
            "sat_any": int(sat_any),
            "sat_long_pct": float(sat_long / max(total_gt, 1) * 100),
            "sat_short_pct": float(sat_short / max(total_gt, 1) * 100),
            "sat_any_pct": float(sat_any / max(total_gt, 1) * 100),
        }
        print(f"  stride={s:2d}: 长边 mean={stride_stats[s]['long_mean']:.1f} P95={stride_stats[s]['long_P95']:.1f} "
              f"max={stride_stats[s]['long_max']:.1f} | "
              f"短边 mean={stride_stats[s]['short_mean']:.1f} P95={stride_stats[s]['short_P95']:.1f} | "
              f"饱和(d>{d_max}): 长边={sat_long}({stride_stats[s]['sat_long_pct']:.1f}%) "
              f"短边={sat_short}({stride_stats[s]['sat_short_pct']:.1f}%)")

    # 2. DFL bin 命中频率直方图（在最可能分配层）
    print(f"\n  [DFL bin 命中分布（最可能分配层）]")
    assigned_long = np.array([e["d_long"] for e in gt_assigned_dists])
    assigned_short = np.array([e["d_short"] for e in gt_assigned_dists])

    # 将距离 clip 到 [0, d_max] 然后统计 bin
    bins = np.arange(0, d_max + 2)  # [0, 1, ..., d_max, d_max+1]
    long_clipped = np.clip(assigned_long, 0, d_max)
    short_clipped = np.clip(assigned_short, 0, d_max)

    long_hist, _ = np.histogram(long_clipped, bins=bins)
    short_hist, _ = np.histogram(short_clipped, bins=bins)

    # 末端 bin 堆积分析
    end_bin_long = (assigned_long >= d_max - 0.5).sum()
    end_bin_short = (assigned_short >= d_max - 0.5).sum()
    overflow_long = (assigned_long > d_max).sum()
    overflow_short = (assigned_short > d_max).sum()

    bin_stats = {
        "long_hist": long_hist.tolist(),
        "short_hist": short_hist.tolist(),
        "end_bin_long": int(end_bin_long),
        "end_bin_short": int(end_bin_short),
        "end_bin_long_pct": float(end_bin_long / max(total_gt, 1) * 100),
        "end_bin_short_pct": float(end_bin_short / max(total_gt, 1) * 100),
        "overflow_long": int(overflow_long),
        "overflow_short": int(overflow_short),
        "overflow_long_pct": float(overflow_long / max(total_gt, 1) * 100),
        "overflow_short_pct": float(overflow_short / max(total_gt, 1) * 100),
    }

    print(f"  末端 bin (>={d_max-0.5}): 长边={end_bin_long}({bin_stats['end_bin_long_pct']:.1f}%)  "
          f"短边={end_bin_short}({bin_stats['end_bin_short_pct']:.1f}%)")
    print(f"  溢出 (>{d_max}): 长边={overflow_long}({bin_stats['overflow_long_pct']:.1f}%)  "
          f"短边={overflow_short}({bin_stats['overflow_short_pct']:.1f}%)")

    # 3. 按长宽比分桶的饱和率
    print(f"\n  [按长宽比分桶的饱和情况]")
    ar_sat_stats = {}
    for lo, hi, name in AR_BUCKETS:
        subset = [e for e in gt_assigned_dists if lo <= e["ar"] < hi]
        n = len(subset)
        if n == 0:
            ar_sat_stats[name] = {"n": 0, "sat_long_pct": 0, "sat_short_pct": 0}
            continue
        sat_l = sum(1 for e in subset if e["d_long"] > d_max)
        sat_s = sum(1 for e in subset if e["d_short"] > d_max)
        ar_sat_stats[name] = {
            "n": n,
            "sat_long": sat_l,
            "sat_short": sat_s,
            "sat_long_pct": float(sat_l / n * 100),
            "sat_short_pct": float(sat_s / n * 100),
            "d_long_mean": float(np.mean([e["d_long"] for e in subset])),
            "d_short_mean": float(np.mean([e["d_short"] for e in subset])),
        }
        print(f"    AR {name:>5s} (n={n:5d}): "
              f"长边饱和={sat_l}({ar_sat_stats[name]['sat_long_pct']:.1f}%)  "
              f"短边饱和={sat_s}({ar_sat_stats[name]['sat_short_pct']:.1f}%)")

    return stride_stats, bin_stats, ar_sat_stats, d_max


def diag5_iou_sensitivity(all_data):
    """诊断 5：GT 扰动 IoU 敏感性曲线。

    对每个 GT 框施加微小扰动，计算 ProbIoU 下降量，
    建立"指标敏感性曲线"，评估任务的先天难度和标注噪声天花板。
    """
    print("\n--- 诊断 5：GT 扰动 IoU 敏感性曲线 ---")

    # 收集所有 GT 框
    all_gt = []
    all_gt_props = []
    for d in all_data:
        for i in range(len(d["gt_cls"])):
            all_gt.append(d["gt_xywhr"][i])
            all_gt_props.append(d["gt_props"][i])

    n_gt = len(all_gt)
    if n_gt == 0:
        print("  没有 GT 框。")
        return {}, {}
    print(f"  GT 总数: {n_gt}")

    gt_arr = np.array(all_gt, dtype=np.float32)  # (N, 5)
    gt_tensor = torch.from_numpy(gt_arr).float()

    def compute_mean_iou_with_perturbation(perturbed_arr):
        """计算原始 GT 与扰动 GT 之间的逐对 ProbIoU 均值。"""
        pert_tensor = torch.from_numpy(perturbed_arr).float()
        # batch_probiou 计算 (N, M) 矩阵，我们只需对角线（逐对）
        # 为效率：分批计算
        batch_size = 500
        ious = []
        for start in range(0, n_gt, batch_size):
            end = min(start + batch_size, n_gt)
            gt_batch = gt_tensor[start:end]
            pt_batch = pert_tensor[start:end]
            iou_mat = batch_probiou(gt_batch, pt_batch)  # (bs, bs)
            diag_iou = torch.diag(iou_mat).clamp(0, 1).numpy()
            ious.append(diag_iou)
        return np.concatenate(ious)

    results = {}

    # --- 中心横向偏移 ---
    print("\n  [中心横向偏移（垂直于长边方向）]")
    center_results = {}
    for px in PERTURB_CENTER_PX:
        perturbed = gt_arr.copy()
        # 垂直于长边方向偏移：方向 = angle + π/2
        perp_dx = -np.sin(gt_arr[:, 4]) * px
        perp_dy = np.cos(gt_arr[:, 4]) * px
        perturbed[:, 0] += perp_dx
        perturbed[:, 1] += perp_dy
        ious = compute_mean_iou_with_perturbation(perturbed)
        mean_iou = float(np.mean(ious))
        median_iou = float(np.median(ious))
        p10_iou = float(np.percentile(ious, 10))
        center_results[px] = {
            "mean_iou": mean_iou,
            "median_iou": median_iou,
            "P10_iou": p10_iou,
            "mean_drop": float(1.0 - mean_iou),
        }
        print(f"    偏移 {px}px: mean_IoU={mean_iou:.4f}  drop={1-mean_iou:.4f}  P10={p10_iou:.4f}")
    results["center"] = center_results

    # --- 角度扰动 ---
    print("\n  [角度扰动]")
    angle_results = {}
    for deg in PERTURB_ANGLE_DEG:
        perturbed = gt_arr.copy()
        perturbed[:, 4] += np.radians(deg)
        ious = compute_mean_iou_with_perturbation(perturbed)
        mean_iou = float(np.mean(ious))
        median_iou = float(np.median(ious))
        p10_iou = float(np.percentile(ious, 10))
        angle_results[deg] = {
            "mean_iou": mean_iou,
            "median_iou": median_iou,
            "P10_iou": p10_iou,
            "mean_drop": float(1.0 - mean_iou),
        }
        print(f"    角度 +{deg}°: mean_IoU={mean_iou:.4f}  drop={1-mean_iou:.4f}  P10={p10_iou:.4f}")
    results["angle"] = angle_results

    # --- 宽度扰动 ---
    print("\n  [宽度扰动（短边 h 增加）]")
    width_results = {}
    for px in PERTURB_WIDTH_PX:
        perturbed = gt_arr.copy()
        perturbed[:, 3] += px  # h 增加（短边）
        ious = compute_mean_iou_with_perturbation(perturbed)
        mean_iou = float(np.mean(ious))
        median_iou = float(np.median(ious))
        p10_iou = float(np.percentile(ious, 10))
        width_results[px] = {
            "mean_iou": mean_iou,
            "median_iou": median_iou,
            "P10_iou": p10_iou,
            "mean_drop": float(1.0 - mean_iou),
        }
        print(f"    宽度 +{px}px: mean_IoU={mean_iou:.4f}  drop={1-mean_iou:.4f}  P10={p10_iou:.4f}")
    results["width"] = width_results

    # --- 按长宽比分桶的敏感性 ---
    print("\n  [按长宽比分桶的敏感性（中心偏移 2px）]")
    ar_sensitivity = {}
    for lo, hi, name in AR_BUCKETS:
        mask = np.array([lo <= p["ar"] < hi for p in all_gt_props])
        n = mask.sum()
        if n == 0:
            ar_sensitivity[name] = {"n": 0}
            continue
        subset_gt = gt_arr[mask]
        subset_tensor = torch.from_numpy(subset_gt).float()

        # 中心偏移 2px
        perturbed = subset_gt.copy()
        perp_dx = -np.sin(subset_gt[:, 4]) * 2
        perp_dy = np.cos(subset_gt[:, 4]) * 2
        perturbed[:, 0] += perp_dx
        perturbed[:, 1] += perp_dy
        pert_tensor = torch.from_numpy(perturbed).float()
        iou_mat = batch_probiou(subset_tensor, pert_tensor)
        diag_iou = torch.diag(iou_mat).clamp(0, 1).numpy()

        # 角度偏移 1°
        perturbed_a = subset_gt.copy()
        perturbed_a[:, 4] += np.radians(1.0)
        pert_a_tensor = torch.from_numpy(perturbed_a).float()
        iou_mat_a = batch_probiou(subset_tensor, pert_a_tensor)
        diag_iou_a = torch.diag(iou_mat_a).clamp(0, 1).numpy()

        # 宽度偏移 1px
        perturbed_w = subset_gt.copy()
        perturbed_w[:, 3] += 1
        pert_w_tensor = torch.from_numpy(perturbed_w).float()
        iou_mat_w = batch_probiou(subset_tensor, pert_w_tensor)
        diag_iou_w = torch.diag(iou_mat_w).clamp(0, 1).numpy()

        ar_sensitivity[name] = {
            "n": int(n),
            "center_2px_iou": float(np.mean(diag_iou)),
            "angle_1deg_iou": float(np.mean(diag_iou_a)),
            "width_1px_iou": float(np.mean(diag_iou_w)),
        }
        print(f"    AR {name:>5s} (n={n:5d}): "
              f"中心2px→IoU={np.mean(diag_iou):.4f}  "
              f"角度1°→IoU={np.mean(diag_iou_a):.4f}  "
              f"宽度1px→IoU={np.mean(diag_iou_w):.4f}")

    return results, ar_sensitivity


def generate_report(config, diag1, diag2, diag3, diag4, diag5, model_info, output_dir):
    """生成诊断 Markdown 报告。"""
    length_results, ar_results, cross_results = diag2
    overall_stats, bucket_stats, bucket_stats_ar = diag3
    stride_stats, bin_stats, ar_sat_stats, d_max_val = diag4
    sensitivity_results, ar_sensitivity = diag5
    reg_max = model_info["reg_max"]
    strides = model_info["strides"]

    lines = []
    lines.append(f"# OBB 模型诊断报告")
    lines.append(f"")
    lines.append(f"**模型**: `{config['model']}`  ")
    lines.append(f"**标签**: {config['model_label']}  ")
    lines.append(f"**数据集**: `{config['data']}`  ")
    lines.append(f"**推理参数**: conf={config['conf']}, NMS IoU={config['iou_nms']}, imgsz={config['imgsz']}  ")
    lines.append(f"")

    # ---- 诊断 1 ----
    lines.append(f"## 诊断 1：按 IoU 阈值拆分 AP")
    lines.append(f"")
    lines.append(f"| IoU 阈值 | AP | Precision | Recall |")
    lines.append(f"|:--------:|:---:|:---------:|:------:|")
    for th in IOU_THRESHOLDS:
        r = diag1.get(th, {})
        ap = r.get("ap", 0)
        p = r.get("precision", 0)
        rec = r.get("recall", 0)
        marker = " **" if th in (0.75, 0.90) else ""
        marker_end = "**" if marker else ""
        lines.append(f"| {marker}AP{int(th*100)}{marker_end} | {ap:.4f} | {p:.3f} | {rec:.3f} |")

    # mAP50-95
    aps = [diag1.get(th, {}).get("ap", 0) for th in IOU_THRESHOLDS]
    map_all = np.mean(aps) if aps else 0
    lines.append(f"| **mAP50-95** | **{map_all:.4f}** | - | - |")
    lines.append(f"")

    lines.append(f"**解读**：如果 AP50 尚可但 AP75+ 断崖下降，说明瓶颈在\"高 IoU 几何拟合失败\"而非\"检不出来\"。")
    lines.append(f"")

    # ---- 诊断 2 ----
    lines.append(f"## 诊断 2：按目标属性分桶 AP")
    lines.append(f"")

    # 按长度
    lines.append(f"### 2.1 按长边长度分桶")
    lines.append(f"")
    lines.append(f"| 长度桶 | GT 数量 | AP50 | AP75 | AP50-AP75 落差 |")
    lines.append(f"|:------:|-------:|:----:|:----:|:-------------:|")
    for name in [b[2] for b in LENGTH_BUCKETS]:
        r = length_results.get(name, {})
        n = r.get("n_gt", 0)
        a50 = r.get("AP50", 0)
        a75 = r.get("AP75", 0)
        drop = a50 - a75
        lines.append(f"| {name} | {n} | {a50:.4f} | {a75:.4f} | {drop:+.4f} |")
    lines.append(f"")

    # 按长宽比
    lines.append(f"### 2.2 按长宽比分桶")
    lines.append(f"")
    lines.append(f"| 长宽比桶 | GT 数量 | AP50 | AP75 | AP50-AP75 落差 |")
    lines.append(f"|:--------:|-------:|:----:|:----:|:-------------:|")
    for name in [b[2] for b in AR_BUCKETS]:
        r = ar_results.get(name, {})
        n = r.get("n_gt", 0)
        a50 = r.get("AP50", 0)
        a75 = r.get("AP75", 0)
        drop = a50 - a75
        lines.append(f"| {name} | {n} | {a50:.4f} | {a75:.4f} | {drop:+.4f} |")
    lines.append(f"")

    # 交叉分桶
    lines.append(f"### 2.3 长度 × IoU 阈值交叉表")
    lines.append(f"")
    header = "| 长度桶 | n_gt | " + " | ".join(f"AP{int(t*100)}" for t in IOU_THRESHOLDS) + " |"
    sep = "|:------:|-----:|" + "|".join(":----:" for _ in IOU_THRESHOLDS) + "|"
    lines.append(header)
    lines.append(sep)
    for name in [b[2] for b in LENGTH_BUCKETS]:
        r = cross_results.get(name, {})
        n = r.get("n_gt", 0)
        vals = " | ".join(f"{r.get(f'AP{int(t*100)}', 0):.4f}" for t in IOU_THRESHOLDS)
        lines.append(f"| {name} | {n} | {vals} |")
    lines.append(f"")

    lines.append(f"**解读**：如果超长/超细目标的 AP75+ 崩得最厉害，证明问题来自\"尺度-形态不对称性\"。")
    lines.append(f"")

    # ---- 诊断 3 ----
    lines.append(f"## 诊断 3：定位误差分解")
    lines.append(f"")
    lines.append(f"匹配阈值: IoU ≥ {ERROR_MATCH_IOU}  ")
    n_matched = sum(len(bucket_stats.get(b[2], {}).get("n", 0).__class__.__name__) and 0
                     for b in LENGTH_BUCKETS)
    total_matched = sum(v.get("n", 0) for v in bucket_stats.values())
    lines.append(f"总匹配对数: {total_matched}")
    lines.append(f"")

    # 总体误差
    lines.append(f"### 3.1 总体误差统计")
    lines.append(f"")
    labels_map = {
        "perp_offset": "横向偏移 (px)",
        "para_offset": "纵向偏移 (px)",
        "angle_error": "角度误差 (°)",
        "width_error": "宽度误差 (px)",
        "length_error": "长度误差 (px)",
        "width_error_rel": "宽度相对误差 (%)",
        "length_error_rel": "长度相对误差 (%)",
    }

    lines.append(f"| 误差类型 | Mean | Median | Std | P90 | P95 |")
    lines.append(f"|:--------:|:----:|:------:|:---:|:---:|:---:|")
    for k in ["perp_offset", "para_offset", "angle_error",
              "width_error", "length_error", "width_error_rel", "length_error_rel"]:
        s = overall_stats.get(k, {})
        label = labels_map.get(k, k)
        lines.append(
            f"| {label} | {s.get('mean',0):.2f} | {s.get('median',0):.2f} | "
            f"{s.get('std',0):.2f} | {s.get('P90',0):.2f} | {s.get('P95',0):.2f} |"
        )
    lines.append(f"")

    lines.append(f"**重点关注**：")
    lines.append(f"- **横向偏移**（垂直于线方向）比纵向偏移更致命")
    lines.append(f"- **角度误差**：哪怕 0.5°–1.5°，对高 IoU 都可能是致命的")
    lines.append(f"- **宽度误差**：极细目标的宽度错 1–2px，IoU 非常难看")
    lines.append(f"- **长度误差**：若 Coverage-Aware TAL 已修好，此项应不再是主矛盾")
    lines.append(f"")

    # 按长度分桶的误差
    lines.append(f"### 3.2 按 GT 长度分桶的误差")
    lines.append(f"")
    lines.append(f"| 长度桶 | 匹配数 | 横向偏移(px) | 角度误差(°) | 宽度误差(px) | 长度误差(px) | 宽度相对(%) |")
    lines.append(f"|:------:|------:|:----------:|:----------:|:----------:|:----------:|:----------:|")
    for bname in [b[2] for b in LENGTH_BUCKETS]:
        bs = bucket_stats.get(bname, {})
        n = bs.get("n", 0)
        if n == 0:
            lines.append(f"| {bname} | 0 | - | - | - | - | - |")
            continue
        lines.append(
            f"| {bname} | {n} | "
            f"{bs['perp_offset']['mean']:.1f} | "
            f"{bs['angle_error']['mean']:.2f} | "
            f"{bs['width_error']['mean']:.1f} | "
            f"{bs['length_error']['mean']:.1f} | "
            f"{bs['width_error_rel']['mean']:.1f} |"
        )
    lines.append(f"")

    # 按长宽比分桶的误差
    lines.append(f"### 3.3 按长宽比分桶的误差")
    lines.append(f"")
    lines.append(f"| 长宽比桶 | 匹配数 | 横向偏移(px) | 角度误差(°) | 宽度误差(px) | 长度误差(px) | 宽度相对(%) |")
    lines.append(f"|:--------:|------:|:----------:|:----------:|:----------:|:----------:|:----------:|")
    for bname in [b[2] for b in AR_BUCKETS]:
        bs = bucket_stats_ar.get(bname, {})
        n = bs.get("n", 0)
        if n == 0:
            lines.append(f"| {bname} | 0 | - | - | - | - | - |")
            continue
        lines.append(
            f"| {bname} | {n} | "
            f"{bs['perp_offset']['mean']:.1f} | "
            f"{bs['angle_error']['mean']:.2f} | "
            f"{bs['width_error']['mean']:.1f} | "
            f"{bs['length_error']['mean']:.1f} | "
            f"{bs['width_error_rel']['mean']:.1f} |"
        )
    lines.append(f"")

    # ---- 诊断 4 ----
    lines.append(f"## 诊断 4：DFL 回归距离饱和分析")
    lines.append(f"")
    lines.append(f"**reg_max** = {reg_max}, **D_max** = {d_max_val} (grid 单位)  ")
    lines.append(f"")

    lines.append(f"### 4.1 各 stride 层距离分布")
    lines.append(f"")
    lines.append(f"| Stride | 长边 Mean | 长边 P95 | 长边 Max | 短边 Mean | 短边 P95 | 长边饱和% | 短边饱和% |")
    lines.append(f"|:------:|:--------:|:-------:|:-------:|:--------:|:-------:|:--------:|:--------:|")
    for s in strides:
        ss = stride_stats.get(s, {})
        lines.append(
            f"| P{int(np.log2(s))} (s={s}) | "
            f"{ss.get('long_mean',0):.1f} | {ss.get('long_P95',0):.1f} | {ss.get('long_max',0):.1f} | "
            f"{ss.get('short_mean',0):.1f} | {ss.get('short_P95',0):.1f} | "
            f"{ss.get('sat_long_pct',0):.1f}% | {ss.get('sat_short_pct',0):.1f}% |"
        )
    lines.append(f"")

    lines.append(f"### 4.2 末端 bin 堆积分析（最可能分配层）")
    lines.append(f"")
    lines.append(f"| 指标 | 长边 | 短边 |")
    lines.append(f"|:----:|:----:|:----:|")
    lines.append(f"| 末端 bin (≥{d_max_val-0.5}) | "
                 f"{bin_stats.get('end_bin_long',0)} ({bin_stats.get('end_bin_long_pct',0):.1f}%) | "
                 f"{bin_stats.get('end_bin_short',0)} ({bin_stats.get('end_bin_short_pct',0):.1f}%) |")
    lines.append(f"| 溢出 (>{d_max_val}) | "
                 f"{bin_stats.get('overflow_long',0)} ({bin_stats.get('overflow_long_pct',0):.1f}%) | "
                 f"{bin_stats.get('overflow_short',0)} ({bin_stats.get('overflow_short_pct',0):.1f}%) |")
    lines.append(f"")

    # DFL bin 直方图（只显示关键区域：末尾 10 个 bin）
    lines.append(f"### 4.3 DFL bin 命中直方图（末尾区域）")
    lines.append(f"")
    long_hist = bin_stats.get("long_hist", [])
    short_hist = bin_stats.get("short_hist", [])
    show_start = max(0, d_max_val - 9)  # 显示最后 10 个 bin
    if long_hist:
        header_bins = " | ".join(f"bin{i}" for i in range(show_start, min(d_max_val + 1, len(long_hist))))
        lines.append(f"| 边 | " + header_bins + " |")
        lines.append(f"|:--:|" + "|".join(":---:" for _ in range(show_start, min(d_max_val + 1, len(long_hist)))) + "|")
        long_vals = " | ".join(str(long_hist[i]) for i in range(show_start, min(d_max_val + 1, len(long_hist))))
        short_vals = " | ".join(str(short_hist[i]) for i in range(show_start, min(d_max_val + 1, len(short_hist))))
        lines.append(f"| 长边 | {long_vals} |")
        lines.append(f"| 短边 | {short_vals} |")
    lines.append(f"")

    lines.append(f"### 4.4 按长宽比分桶的饱和情况")
    lines.append(f"")
    lines.append(f"| 长宽比桶 | GT 数 | 长边饱和% | 短边饱和% | 长边距离均值 | 短边距离均值 |")
    lines.append(f"|:--------:|------:|:--------:|:--------:|:----------:|:----------:|")
    for name in [b[2] for b in AR_BUCKETS]:
        a = ar_sat_stats.get(name, {})
        n = a.get("n", 0)
        if n == 0:
            lines.append(f"| {name} | 0 | - | - | - | - |")
            continue
        lines.append(
            f"| {name} | {n} | "
            f"{a.get('sat_long_pct',0):.1f}% | {a.get('sat_short_pct',0):.1f}% | "
            f"{a.get('d_long_mean',0):.1f} | {a.get('d_short_mean',0):.1f} |"
        )
    lines.append(f"")

    lines.append(f"**解读**：")
    lines.append(f"- 如果仍有末端 bin 堆积，说明从\"硬截断\"变成了\"软饱和\"，reg_max 可能还需更大")
    lines.append(f"- 长边比短边更容易饱和是正常的（长线缆特征）")
    lines.append(f"- 关注高长宽比桶的饱和率，这些是最关键的目标")
    lines.append(f"")

    # ---- 诊断 5 ----
    lines.append(f"## 诊断 5：GT 扰动 IoU 敏感性曲线")
    lines.append(f"")
    lines.append(f"通过对 GT 框施加已知微扰动并计算 ProbIoU 下降，评估任务先天难度和标注噪声天花板。")
    lines.append(f"")

    lines.append(f"### 5.1 中心横向偏移（垂直于长边方向）")
    lines.append(f"")
    lines.append(f"| 偏移量 (px) | Mean IoU | IoU 下降 | Median IoU | P10 IoU |")
    lines.append(f"|:-----------:|:--------:|:-------:|:----------:|:-------:|")
    center_r = sensitivity_results.get("center", {})
    for px in PERTURB_CENTER_PX:
        r = center_r.get(px, {})
        lines.append(f"| {px} | {r.get('mean_iou',0):.4f} | {r.get('mean_drop',0):.4f} | "
                     f"{r.get('median_iou',0):.4f} | {r.get('P10_iou',0):.4f} |")
    lines.append(f"")

    lines.append(f"### 5.2 角度扰动")
    lines.append(f"")
    lines.append(f"| 角度扰动 (°) | Mean IoU | IoU 下降 | Median IoU | P10 IoU |")
    lines.append(f"|:------------:|:--------:|:-------:|:----------:|:-------:|")
    angle_r = sensitivity_results.get("angle", {})
    for deg in PERTURB_ANGLE_DEG:
        r = angle_r.get(deg, {})
        lines.append(f"| {deg} | {r.get('mean_iou',0):.4f} | {r.get('mean_drop',0):.4f} | "
                     f"{r.get('median_iou',0):.4f} | {r.get('P10_iou',0):.4f} |")
    lines.append(f"")

    lines.append(f"### 5.3 宽度扰动（短边增加）")
    lines.append(f"")
    lines.append(f"| 宽度扰动 (px) | Mean IoU | IoU 下降 | Median IoU | P10 IoU |")
    lines.append(f"|:-------------:|:--------:|:-------:|:----------:|:-------:|")
    width_r = sensitivity_results.get("width", {})
    for px in PERTURB_WIDTH_PX:
        r = width_r.get(px, {})
        lines.append(f"| +{px} | {r.get('mean_iou',0):.4f} | {r.get('mean_drop',0):.4f} | "
                     f"{r.get('median_iou',0):.4f} | {r.get('P10_iou',0):.4f} |")
    lines.append(f"")

    lines.append(f"### 5.4 按长宽比分桶的敏感性")
    lines.append(f"")
    lines.append(f"| 长宽比桶 | GT 数 | 中心 2px → IoU | 角度 1° → IoU | 宽度 1px → IoU |")
    lines.append(f"|:--------:|------:|:--------------:|:-------------:|:--------------:|")
    for name in [b[2] for b in AR_BUCKETS]:
        a = ar_sensitivity.get(name, {})
        n = a.get("n", 0)
        if n == 0:
            lines.append(f"| {name} | 0 | - | - | - |")
            continue
        lines.append(
            f"| {name} | {n} | "
            f"{a.get('center_2px_iou',0):.4f} | "
            f"{a.get('angle_1deg_iou',0):.4f} | "
            f"{a.get('width_1px_iou',0):.4f} |"
        )
    lines.append(f"")

    lines.append(f"**解读**：")
    lines.append(f"- 如果仅 1-2px 横向偏移就导致 IoU 显著下降，说明任务评估先天苛刻")
    lines.append(f"- 如果标注噪声本身就有 1-2px，那么 mAP75+ 的天花板可能是标注精度决定的")
    lines.append(f"- P10 列表示最敏感的 10% 目标（通常是最细的线缆），是最脆弱的群体")
    lines.append(f"- 不同长宽比桶的敏感性差异反映了\"形态不对称性\"的程度")
    lines.append(f"")

    # 结论模板
    lines.append(f"## 初步诊断结论")
    lines.append(f"")
    lines.append(f"_(根据上述数据填写)_")
    lines.append(f"")

    ap50 = diag1.get(0.50, {}).get("ap", 0)
    ap75 = diag1.get(0.75, {}).get("ap", 0)
    ap90 = diag1.get(0.90, {}).get("ap", 0)
    lines.append(f"1. **AP 梯度特征**：AP50={ap50:.4f} → AP75={ap75:.4f} → AP90={ap90:.4f}")
    if ap50 > 0 and ap75 / ap50 < 0.5:
        lines.append(f"   - AP75/AP50 = {ap75/ap50:.2f}，存在显著的高 IoU 几何拟合问题")
    if ap90 < 0.05:
        lines.append(f"   - AP90 接近零，极高 IoU 下模型完全无法精确拟合")
    lines.append(f"")

    lines.append(f"2. **形态不对称性**：对比不同长度/长宽比桶的 AP75 落差")
    lines.append(f"")

    perp_mean = overall_stats.get("perp_offset", {}).get("mean", 0)
    angle_mean = overall_stats.get("angle_error", {}).get("mean", 0)
    width_mean = overall_stats.get("width_error", {}).get("mean", 0)
    lines.append(f"3. **误差主矛盾**：")
    lines.append(f"   - 横向偏移均值 = {perp_mean:.1f}px")
    lines.append(f"   - 角度误差均值 = {angle_mean:.2f}°")
    lines.append(f"   - 宽度误差均值 = {width_mean:.1f}px")
    lines.append(f"")

    report_path = os.path.join(output_dir, "diag_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n报告已保存: {report_path}")
    return report_path


def main():
    config = CONFIG.copy()
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 收集数据
    all_data, class_names, model_info = collect_data(config)

    if not all_data:
        print("[!] 没有收集到数据，退出。")
        return

    # 诊断 1
    diag1 = diag1_ap_by_threshold(all_data)

    # 诊断 2
    diag2 = diag2_ap_by_bucket(all_data)

    # 诊断 3
    diag3 = diag3_loc_errors(all_data)

    # 诊断 4
    diag4 = diag4_dfl_saturation(all_data, model_info)

    # 诊断 5
    diag5 = diag5_iou_sensitivity(all_data)

    # 生成报告
    generate_report(config, diag1, diag2, diag3, diag4, diag5, model_info, output_dir)

    print("\n" + "=" * 60)
    print("  诊断完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
