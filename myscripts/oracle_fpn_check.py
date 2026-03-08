"""
OBB 模型几何排查脚本
功能：
  排查 1 — 几何替换 Oracle 实验（逐项替换预测框的宽度/角度/中心/长度为 GT，重算 AP75/AP90）
  排查 2 — 按 FPN 层分桶分析误差（每层的 AP / 宽度误差 / 角度误差 / 横向偏移 / 饱和率）
  排查 3 — Coverage-Aware TAL 正样本层分布验证（按长度/长宽比桶统计 GT 可覆盖层分布）
输出：排查 Markdown 报告

使用方法:
    修改下方 CONFIG 后运行: python myscripts/oracle_fpn_check.py
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

# 添加项目根目录和脚本目录
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
SCRIPT_DIR = str(Path(__file__).resolve().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ultralytics import YOLO
from ultralytics.utils.metrics import batch_probiou
from diag_val import (
    load_dataset_paths, load_gt_for_image, extract_predictions,
    gt_properties, bucket_name, compute_ap_from_tp, compute_loc_errors,
    match_at_thresholds, LENGTH_BUCKETS, AR_BUCKETS,
)

# ========================== 配置 ==========================
CONFIG = {
    # ---------- 模型 ----------
    "model": "/root/autodl-tmp/work-dirs/yolo11_obb_sp_ca/weights/best.pt",
    "model_label": "CA (reg_max=32)",

    # ---------- 数据集 ----------
    "data": "/root/autodl-tmp/dataset/TTPLA-1024/dataset.yaml",

    # ---------- 推理参数 ----------
    "conf": 0.001,
    "iou_nms": 0.7,
    "imgsz": 1024,
    "device": 0,
    "max_det": 1000,

    # ---------- 输出 ----------
    "output_dir": "/root/autodl-tmp/work-dirs/oracle_fpn_check_sp_ca",
}

# 排查用 IoU 阈值
CHECK_THRESHOLDS = [0.50, 0.75, 0.90]


# ========================== 数据收集 ==========================

def collect_data(config):
    """收集所有图片的预测和 GT 数据，提取模型信息。"""
    print("=" * 60)
    print("  OBB 几何排查脚本")
    print("=" * 60)

    image_files, label_dir, class_names = load_dataset_paths(config["data"])
    print(f"  数据集: {config['data']}")
    print(f"  验证图片数: {len(image_files)}")
    print(f"  标注目录: {label_dir}")

    print(f"\n  加载模型: {config['model']}")
    model = YOLO(config["model"])

    det_head = model.model.model[-1]
    reg_max = int(det_head.reg_max)
    strides = [int(s) for s in det_head.stride.tolist()]
    model_info = {"reg_max": reg_max, "strides": strides}
    print(f"  reg_max = {reg_max}, strides = {strides}")

    all_data = []

    for idx, img_path in enumerate(image_files):
        img_name = os.path.basename(img_path)
        stem = os.path.splitext(img_name)[0]

        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(image_files)}] 处理: {img_name}")

        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        label_path = os.path.join(label_dir, stem + ".txt")
        gt_cls, gt_xywhr = load_gt_for_image(label_path, img_w, img_h)

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

        n_gt = len(gt_cls)
        n_pred = len(pred_cls)
        if n_gt > 0 and n_pred > 0:
            gt_t = torch.from_numpy(gt_xywhr).float()
            pd_t = torch.from_numpy(pred_xywhr).float()
            iou_matrix = batch_probiou(gt_t, pd_t).numpy()
            iou_matrix = np.clip(iou_matrix, 0, 1)
        else:
            iou_matrix = np.zeros((n_gt, n_pred), dtype=np.float32)

        # 在 IoU50 下匹配，用于 Oracle 实验和误差分析
        tp, matched_gt_idx = match_at_thresholds(
            iou_matrix, gt_cls, pred_cls, CHECK_THRESHOLDS
        )

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


# ========================== 排查 1：Oracle 几何替换实验 ==========================

def oracle_replace_width(pred, gt):
    """替换宽度（短边 h）为 GT。"""
    r = pred.copy()
    r[3] = gt[3]
    return r

def oracle_replace_length(pred, gt):
    """替换长度（长边 w）为 GT。"""
    r = pred.copy()
    r[2] = gt[2]
    return r

def oracle_replace_angle(pred, gt):
    """替换角度为 GT。"""
    r = pred.copy()
    r[4] = gt[4]
    return r

def oracle_replace_center(pred, gt):
    """替换中心坐标为 GT。"""
    r = pred.copy()
    r[0] = gt[0]
    r[1] = gt[1]
    return r

def oracle_replace_width_angle(pred, gt):
    """同时替换宽度和角度为 GT。"""
    r = pred.copy()
    r[3] = gt[3]
    r[4] = gt[4]
    return r

def oracle_replace_all(pred, gt):
    """替换所有几何量为 GT（理论上限）。"""
    return gt.copy()


def compute_oracle_ap(all_data, modify_fn, iou_threshold):
    """用 Oracle 修改后的预测计算 AP。

    对每张图片：
      1. 复制预测框
      2. 对 IoU50 匹配成功的预测框，用 modify_fn 修改
      3. 重新计算 IoU 矩阵
      4. 在 iou_threshold 下重新匹配
      5. 汇总 TP/FP 并计算 AP
    """
    all_tp = []
    all_conf = []
    total_gt = 0

    for d in all_data:
        gt_cls = d["gt_cls"]
        gt_xywhr = d["gt_xywhr"]
        pred_cls = d["pred_cls"]
        pred_conf = d["pred_conf"]
        pred_xywhr = d["pred_xywhr"].copy()
        matched_gt_idx = d["matched_gt_idx"]

        n_gt = len(gt_cls)
        n_pred = len(pred_cls)
        total_gt += n_gt

        if n_pred == 0:
            continue

        # 修改匹配成功的预测框
        for pi in range(n_pred):
            gi = matched_gt_idx[pi]
            if gi >= 0:
                pred_xywhr[pi] = modify_fn(pred_xywhr[pi], gt_xywhr[gi])

        # 重新计算 IoU
        if n_gt > 0 and n_pred > 0:
            gt_t = torch.from_numpy(gt_xywhr).float()
            pd_t = torch.from_numpy(pred_xywhr).float()
            new_iou = batch_probiou(gt_t, pd_t).numpy()
            new_iou = np.clip(new_iou, 0, 1)
        else:
            new_iou = np.zeros((n_gt, n_pred), dtype=np.float32)

        # 在目标阈值下重新匹配
        tp, _ = match_at_thresholds(new_iou, gt_cls, pred_cls, [iou_threshold])
        all_tp.append(tp[:, 0])
        all_conf.append(pred_conf)

    if not all_tp:
        return 0.0

    all_tp = np.concatenate(all_tp)
    all_conf = np.concatenate(all_conf)
    ap, _, _ = compute_ap_from_tp(all_tp, all_conf, total_gt)
    return ap


def check1_oracle_experiment(all_data):
    """排查 1：几何替换 Oracle 实验。"""
    print("\n" + "=" * 60)
    print("  排查 1：几何替换 Oracle 实验")
    print("=" * 60)

    # 基线 AP
    print("\n  计算基线 AP ...")
    baseline = {}
    for th in [0.75, 0.90]:
        all_tp = []
        all_conf = []
        total_gt = 0
        th_idx = CHECK_THRESHOLDS.index(th)
        for d in all_data:
            all_tp.append(d["tp"][:, th_idx])
            all_conf.append(d["pred_conf"])
            total_gt += len(d["gt_cls"])
        if all_tp:
            tp_arr = np.concatenate(all_tp)
            conf_arr = np.concatenate(all_conf)
            ap, _, _ = compute_ap_from_tp(tp_arr, conf_arr, total_gt)
        else:
            ap = 0.0
        baseline[th] = ap
        print(f"    基线 AP{int(th*100)} = {ap:.4f}")

    # Oracle 变体
    oracle_variants = [
        ("replace_width",       "仅替换宽度(短边h)→GT",     oracle_replace_width),
        ("replace_angle",       "仅替换角度→GT",            oracle_replace_angle),
        ("replace_center",      "仅替换中心→GT",            oracle_replace_center),
        ("replace_length",      "仅替换长度(长边w)→GT",     oracle_replace_length),
        ("replace_width_angle", "同时替换宽度+角度→GT",     oracle_replace_width_angle),
        ("replace_all",         "替换全部→GT（理论上限）",  oracle_replace_all),
    ]

    results = {}
    for key, label, modify_fn in oracle_variants:
        print(f"\n  [{label}]")
        row = {"label": label}
        for th in [0.75, 0.90]:
            ap = compute_oracle_ap(all_data, modify_fn, th)
            gain = ap - baseline[th]
            row[f"ap{int(th*100)}"] = ap
            row[f"gain{int(th*100)}"] = gain
            print(f"    AP{int(th*100)} = {ap:.4f}  (提升 {gain:+.4f})")
        results[key] = row

    return baseline, results


# ========================== 排查 2：按 FPN 层分桶分析 ==========================

def estimate_gt_layer(gt_xywhr, strides, d_max):
    """估算 GT 的"自然分配层"：最小的能覆盖 GT 的 stride。

    覆盖条件：max(w, h) / (2 * stride) ≤ d_max

    Args:
        gt_xywhr: (5,) xywhr
        strides: [8, 16, 32]
        d_max: reg_max - 1

    Returns:
        stride: int, 分配的 stride
    """
    w, h = gt_xywhr[2], gt_xywhr[3]
    for s in strides:
        d_long = w / (2 * s)
        d_short = h / (2 * s)
        if d_long <= d_max and d_short <= d_max:
            return s
    return strides[-1]  # fallback 到最大 stride


def check2_per_layer_analysis(all_data, model_info):
    """排查 2：按 FPN 层分桶分析误差。"""
    print("\n" + "=" * 60)
    print("  排查 2：按 FPN 层分桶分析误差")
    print("=" * 60)

    reg_max = model_info["reg_max"]
    strides = model_info["strides"]
    d_max = reg_max - 1

    # 1. 给每个 GT 分配"自然层"
    # 2. 按层分桶，计算 AP50/AP75 和误差统计
    layer_gt_data = {s: {"tp_50": [], "tp_75": [], "conf": [], "n_gt": 0, "errors": []}
                     for s in strides}

    for d in all_data:
        gt_cls = d["gt_cls"]
        gt_xywhr = d["gt_xywhr"]
        pred_cls = d["pred_cls"]
        pred_conf = d["pred_conf"]
        pred_xywhr = d["pred_xywhr"]
        iou_matrix = d["iou_matrix"]
        matched_gt_idx = d["matched_gt_idx"]

        n_gt = len(gt_cls)
        n_pred = len(pred_cls)

        # 给每个 GT 分配自然层
        gt_layers = np.array([estimate_gt_layer(gt_xywhr[i], strides, d_max)
                              for i in range(n_gt)], dtype=np.int32) if n_gt > 0 else np.array([], dtype=np.int32)

        for s in strides:
            # 筛选该层的 GT
            layer_mask = (gt_layers == s) if n_gt > 0 else np.array([], dtype=bool)
            n_gt_layer = layer_mask.sum() if n_gt > 0 else 0
            layer_gt_data[s]["n_gt"] += n_gt_layer

            if n_pred == 0 or n_gt_layer == 0:
                continue

            # 用该层 GT 子集做匹配
            layer_gt_indices = np.where(layer_mask)[0]
            sub_iou = iou_matrix[layer_gt_indices, :]
            sub_gt_cls = gt_cls[layer_gt_indices]

            for th_key, th_val in [("tp_50", 0.50), ("tp_75", 0.75)]:
                cls_match = sub_gt_cls[:, None] == pred_cls[None, :]
                iou_masked = sub_iou * cls_match
                tp_pred = np.zeros(n_pred, dtype=bool)
                matches = np.argwhere(iou_masked >= th_val)
                if len(matches) > 0:
                    match_ious = iou_masked[matches[:, 0], matches[:, 1]]
                    order = match_ious.argsort()[::-1]
                    matches = matches[order]
                    matched_g, matched_p = set(), set()
                    for g, p in matches:
                        if g in matched_g or p in matched_p:
                            continue
                        tp_pred[p] = True
                        matched_g.add(g)
                        matched_p.add(p)
                layer_gt_data[s][th_key].append(tp_pred)
                if th_key == "tp_50":
                    layer_gt_data[s]["conf"].append(pred_conf)

            # 误差分析：对 IoU50 匹配到该层 GT 的预测
            for pi in range(n_pred):
                gi = matched_gt_idx[pi]
                if gi < 0:
                    continue
                if gi < n_gt and gt_layers[gi] == s:
                    errors = compute_loc_errors(gt_xywhr[gi], pred_xywhr[pi])
                    # 添加饱和信息
                    w_gt = gt_xywhr[gi, 2]
                    d_long_s = w_gt / (2 * s)
                    errors["d_long"] = float(d_long_s)
                    errors["saturated"] = d_long_s > d_max
                    layer_gt_data[s]["errors"].append(errors)

    # 汇总统计
    layer_results = {}
    for s in strides:
        ld = layer_gt_data[s]
        n_gt = ld["n_gt"]

        # AP
        ap50, ap75 = 0.0, 0.0
        if n_gt > 0 and ld["tp_50"]:
            tp50 = np.concatenate(ld["tp_50"])
            conf = np.concatenate(ld["conf"])
            ap50, _, _ = compute_ap_from_tp(tp50, conf, n_gt)
        if n_gt > 0 and ld["tp_75"]:
            tp75 = np.concatenate(ld["tp_75"])
            conf = np.concatenate(ld["conf"])
            ap75, _, _ = compute_ap_from_tp(tp75, conf, n_gt)

        # 误差统计
        errors = ld["errors"]
        n_matched = len(errors)
        error_stats = {}
        if n_matched > 0:
            for k in ["perp_offset", "angle_error", "width_error", "length_error",
                       "width_error_rel"]:
                vals = np.array([e[k] for e in errors])
                error_stats[k] = {
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "P95": float(np.percentile(vals, 95)),
                }
            sat_count = sum(1 for e in errors if e["saturated"])
            error_stats["sat_pct"] = float(sat_count / n_matched * 100)
        else:
            for k in ["perp_offset", "angle_error", "width_error", "length_error",
                       "width_error_rel"]:
                error_stats[k] = {"mean": 0, "median": 0, "P95": 0}
            error_stats["sat_pct"] = 0

        layer_name = f"P{int(np.log2(s))}"
        layer_results[s] = {
            "name": layer_name,
            "stride": s,
            "n_gt": n_gt,
            "n_matched": n_matched,
            "ap50": ap50,
            "ap75": ap75,
            "error_stats": error_stats,
        }
        print(f"  {layer_name} (stride={s}): n_gt={n_gt:5d}  AP50={ap50:.4f}  AP75={ap75:.4f}  "
              f"matched={n_matched}  横向={error_stats['perp_offset']['mean']:.1f}px  "
              f"角度={error_stats['angle_error']['mean']:.2f}°  "
              f"宽度误差={error_stats['width_error']['mean']:.1f}px  "
              f"饱和={error_stats['sat_pct']:.1f}%")

    return layer_results


# ========================== 排查 3：Coverage-Aware TAL 正样本层分布验证 ==========================

def check3_ca_verification(all_data, model_info):
    """排查 3：验证 Coverage-Aware TAL 是否把极端样本送到了正确层。"""
    print("\n" + "=" * 60)
    print("  排查 3：Coverage-Aware TAL 正样本层分布验证")
    print("=" * 60)

    reg_max = model_info["reg_max"]
    strides = model_info["strides"]
    d_max = reg_max - 1

    # 对每个 GT，计算：
    # 1. 无 CA 过滤时的可覆盖层（所有 stride 都可选）
    # 2. 有 CA 过滤时的可覆盖层（D_req ≤ D_max 的 stride）
    # 3. 分配到哪层（最小可覆盖 stride）

    gt_records = []
    for d in all_data:
        gt_xywhr = d["gt_xywhr"]
        gt_props = d["gt_props"]
        for i in range(len(d["gt_cls"])):
            w, h = gt_xywhr[i, 2], gt_xywhr[i, 3]
            props = gt_props[i]

            # 各 stride 的回归距离
            stride_info = {}
            eligible_no_ca = []  # 无 CA：所有 stride 都可选
            eligible_ca = []     # 有 CA：D_req ≤ D_max

            for s in strides:
                d_long = w / (2 * s)
                d_short = h / (2 * s)
                can_cover = (d_long <= d_max) and (d_short <= d_max)
                stride_info[s] = {
                    "d_long": d_long,
                    "d_short": d_short,
                    "can_cover": can_cover,
                }
                eligible_no_ca.append(s)  # 无 CA 时所有层都可选
                if can_cover:
                    eligible_ca.append(s)

            # 分配层
            assigned_no_ca = strides[0]  # 无 CA 时默认最小 stride（最高分辨率）
            assigned_ca = eligible_ca[0] if eligible_ca else strides[-1]  # 有 CA 时最小可覆盖 stride

            gt_records.append({
                "length": props["length"],
                "width": props["width"],
                "ar": props["ar"],
                "w_px": float(w),
                "h_px": float(h),
                "stride_info": stride_info,
                "eligible_no_ca": eligible_no_ca,
                "eligible_ca": eligible_ca,
                "assigned_no_ca": assigned_no_ca,
                "assigned_ca": assigned_ca,
                "no_layer_fits": len(eligible_ca) == 0,
            })

    total = len(gt_records)
    print(f"  总 GT 数: {total}")

    # 1. 总体层分布对比
    print("\n  [总体层分布：无 CA vs 有 CA]")
    dist_no_ca = defaultdict(int)
    dist_ca = defaultdict(int)
    no_fit_count = 0
    for r in gt_records:
        dist_no_ca[r["assigned_no_ca"]] += 1
        dist_ca[r["assigned_ca"]] += 1
        if r["no_layer_fits"]:
            no_fit_count += 1

    overall_dist = {}
    for s in strides:
        name = f"P{int(np.log2(s))}"
        n_no = dist_no_ca.get(s, 0)
        n_ca = dist_ca.get(s, 0)
        overall_dist[s] = {
            "name": name,
            "no_ca": n_no,
            "no_ca_pct": n_no / max(total, 1) * 100,
            "ca": n_ca,
            "ca_pct": n_ca / max(total, 1) * 100,
        }
        print(f"    {name}: 无CA={n_no}({overall_dist[s]['no_ca_pct']:.1f}%)  "
              f"有CA={n_ca}({overall_dist[s]['ca_pct']:.1f}%)")
    print(f"    无层可覆盖（fallback）: {no_fit_count}({no_fit_count/max(total,1)*100:.1f}%)")

    # 2. 按长度桶的层分布
    print("\n  [按长度桶的层分布（有 CA）]")
    bucket_layer_dist = {}
    for lo, hi, bname in LENGTH_BUCKETS:
        subset = [r for r in gt_records if lo <= r["length"] < hi]
        n = len(subset)
        if n == 0:
            bucket_layer_dist[bname] = {"n": 0}
            continue
        dist = {s: 0 for s in strides}
        no_fit = 0
        for r in subset:
            dist[r["assigned_ca"]] += 1
            if r["no_layer_fits"]:
                no_fit += 1
        bucket_layer_dist[bname] = {
            "n": n,
            "dist": {s: {"count": dist[s], "pct": dist[s]/n*100} for s in strides},
            "no_fit": no_fit,
            "no_fit_pct": no_fit / n * 100,
        }
        dist_str = "  ".join(
            f"P{int(np.log2(s))}={dist[s]}({dist[s]/n*100:.0f}%)" for s in strides
        )
        print(f"    {bname:>8s} (n={n:5d}): {dist_str}  无层={no_fit}({no_fit/n*100:.1f}%)")

    # 3. 按长宽比桶的层分布
    print("\n  [按长宽比桶的层分布（有 CA）]")
    ar_layer_dist = {}
    for lo, hi, bname in AR_BUCKETS:
        subset = [r for r in gt_records if lo <= r["ar"] < hi]
        n = len(subset)
        if n == 0:
            ar_layer_dist[bname] = {"n": 0}
            continue
        dist = {s: 0 for s in strides}
        no_fit = 0
        for r in subset:
            dist[r["assigned_ca"]] += 1
            if r["no_layer_fits"]:
                no_fit += 1
        ar_layer_dist[bname] = {
            "n": n,
            "dist": {s: {"count": dist[s], "pct": dist[s]/n*100} for s in strides},
            "no_fit": no_fit,
            "no_fit_pct": no_fit / n * 100,
        }
        dist_str = "  ".join(
            f"P{int(np.log2(s))}={dist[s]}({dist[s]/n*100:.0f}%)" for s in strides
        )
        print(f"    AR {bname:>5s} (n={n:5d}): {dist_str}  无层={no_fit}({no_fit/n*100:.1f}%)")

    # 4. 关键统计：>512 长度 和 >60 长宽比 的详细分析
    print("\n  [极端目标详细分析]")
    extreme_records = [r for r in gt_records if r["length"] >= 512 or r["ar"] >= 60]
    n_extreme = len(extreme_records)
    if n_extreme > 0:
        stuck_p3 = sum(1 for r in extreme_records if r["assigned_ca"] == strides[0])
        no_fit_ext = sum(1 for r in extreme_records if r["no_layer_fits"])
        print(f"    极端目标 (长度≥512 或 AR≥60): {n_extreme}")
        print(f"    仍停留在 P{int(np.log2(strides[0]))} (stride={strides[0]}): "
              f"{stuck_p3} ({stuck_p3/n_extreme*100:.1f}%)")
        print(f"    无层可覆盖: {no_fit_ext} ({no_fit_ext/n_extreme*100:.1f}%)")

        # 按分配层统计
        ext_dist = {s: 0 for s in strides}
        for r in extreme_records:
            ext_dist[r["assigned_ca"]] += 1
        for s in strides:
            name = f"P{int(np.log2(s))}"
            print(f"      {name}: {ext_dist[s]} ({ext_dist[s]/n_extreme*100:.1f}%)")

    extreme_stats = {
        "n": n_extreme,
        "stuck_p3": stuck_p3 if n_extreme > 0 else 0,
        "no_fit": no_fit_ext if n_extreme > 0 else 0,
    }

    return overall_dist, bucket_layer_dist, ar_layer_dist, extreme_stats, no_fit_count


# ========================== 报告生成 ==========================

def generate_report(config, check1_result, check2_result, check3_result, model_info, output_dir):
    """生成排查 Markdown 报告。"""
    baseline, oracle_results = check1_result
    layer_results = check2_result
    overall_dist, bucket_layer_dist, ar_layer_dist, extreme_stats, no_fit_count = check3_result

    reg_max = model_info["reg_max"]
    strides = model_info["strides"]
    d_max = reg_max - 1

    lines = []
    lines.append("# OBB 模型几何排查报告")
    lines.append("")
    lines.append(f"**模型**: `{config['model']}`  ")
    lines.append(f"**标签**: {config['model_label']}  ")
    lines.append(f"**数据集**: `{config['data']}`  ")
    lines.append(f"**reg_max** = {reg_max}, **D_max** = {d_max}, **strides** = {strides}  ")
    lines.append("")

    # ---- 排查 1 ----
    lines.append("## 排查 1：几何替换 Oracle 实验")
    lines.append("")
    lines.append("对 IoU50 匹配成功的预测框，逐项替换为 GT 值后重新计算 AP。")
    lines.append("")
    lines.append("| Oracle 变体 | AP75 | AP75 提升 | AP90 | AP90 提升 |")
    lines.append("|:----------:|:----:|:--------:|:----:|:--------:|")
    lines.append(f"| **基线（不替换）** | {baseline[0.75]:.4f} | - | {baseline[0.90]:.4f} | - |")

    # 按 AP75 提升排序
    sorted_keys = sorted(oracle_results.keys(),
                          key=lambda k: oracle_results[k].get("gain75", 0), reverse=True)
    for key in sorted_keys:
        r = oracle_results[key]
        lines.append(
            f"| {r['label']} | {r['ap75']:.4f} | **{r['gain75']:+.4f}** | "
            f"{r['ap90']:.4f} | **{r['gain90']:+.4f}** |"
        )
    lines.append("")

    # 解读
    lines.append("**解读**：")
    lines.append("")
    # 找出提升最大的
    if oracle_results:
        top_key = sorted_keys[0]
        top_r = oracle_results[top_key]
        lines.append(f"- AP75 提升最大的替换：**{top_r['label']}** (提升 {top_r['gain75']:+.4f})")
        if len(sorted_keys) > 1:
            second_key = sorted_keys[1]
            second_r = oracle_results[second_key]
            lines.append(f"- AP75 提升第二的替换：**{second_r['label']}** (提升 {second_r['gain75']:+.4f})")
    lines.append("")

    # 瓶颈判断
    lines.append("**瓶颈判定**：")
    rw = oracle_results.get("replace_width", {})
    ra = oracle_results.get("replace_angle", {})
    rc = oracle_results.get("replace_center", {})
    rl = oracle_results.get("replace_length", {})
    rwa = oracle_results.get("replace_width_angle", {})
    rall = oracle_results.get("replace_all", {})
    lines.append(f"- 宽度提升 ({rw.get('gain75',0):+.4f}) + 角度提升 ({ra.get('gain75',0):+.4f}) "
                 f"= {rw.get('gain75',0)+ra.get('gain75',0):+.4f}，"
                 f"而联合替换提升 = {rwa.get('gain75',0):+.4f}")
    if rwa.get('gain75', 0) > rw.get('gain75', 0) + ra.get('gain75', 0):
        lines.append("  - 联合效果 > 单项之和 → 宽度和角度误差之间存在**协同放大**效应")
    else:
        lines.append("  - 联合效果 ≈ 单项之和 → 两类误差较为独立")
    lines.append(f"- 替换全部的理论上限 AP75 = {rall.get('ap75',0):.4f}，"
                 f"距离当前提升空间 = {rall.get('gain75',0):+.4f}")
    if rl.get('gain75', 0) < rw.get('gain75', 0) * 0.5:
        lines.append("- 长度替换提升远小于宽度 → **主瓶颈不是\"画不长\"，而是\"画不细、画不正、画不准\"**")
    lines.append("")

    # ---- 排查 2 ----
    lines.append("## 排查 2：按 FPN 层分桶分析误差")
    lines.append("")
    lines.append("GT 按\"自然分配层\"分组（最小可覆盖 stride），统计各层 AP 和定位误差。")
    lines.append("")
    lines.append("| 层 | Stride | GT 数 | 匹配数 | AP50 | AP75 | 横向偏移(px) | 角度误差(°) | "
                 "宽度误差(px) | 长度误差(px) | 长边饱和% |")
    lines.append("|:--:|:------:|------:|------:|:----:|:----:|:----------:|:----------:|"
                 ":----------:|:----------:|:--------:|")
    for s in strides:
        lr = layer_results.get(s, {})
        name = lr.get("name", f"stride={s}")
        n_gt = lr.get("n_gt", 0)
        n_m = lr.get("n_matched", 0)
        a50 = lr.get("ap50", 0)
        a75 = lr.get("ap75", 0)
        es = lr.get("error_stats", {})
        lines.append(
            f"| {name} | {s} | {n_gt} | {n_m} | {a50:.4f} | {a75:.4f} | "
            f"{es.get('perp_offset',{}).get('mean',0):.1f} | "
            f"{es.get('angle_error',{}).get('mean',0):.2f} | "
            f"{es.get('width_error',{}).get('mean',0):.1f} | "
            f"{es.get('length_error',{}).get('mean',0):.1f} | "
            f"{es.get('sat_pct',0):.1f}% |"
        )
    lines.append("")

    lines.append("**解读**：")
    lines.append("- 如果 P4/P5 长边不太饱和但短边/角度精修差 → 深层特征分辨率不足")
    lines.append("- 如果 P3 短边较好但长边容易截断 → 浅层覆盖不足")
    lines.append("- 如果两端都有问题 → 需要**双尺度解耦**，而非单头硬修")
    lines.append("")

    # ---- 排查 3 ----
    lines.append("## 排查 3：Coverage-Aware TAL 正样本层分布验证")
    lines.append("")
    lines.append(f"**reg_max** = {reg_max}, **D_max** = {d_max}  ")
    lines.append("")

    # 总体层分布
    lines.append("### 3.1 总体层分布（无 CA vs 有 CA）")
    lines.append("")
    lines.append("| 层 | 无 CA 分配 | 无 CA % | 有 CA 分配 | 有 CA % | 变化 |")
    lines.append("|:--:|:---------:|:------:|:---------:|:------:|:----:|")
    for s in strides:
        od = overall_dist.get(s, {})
        name = od.get("name", f"s={s}")
        n_no = od.get("no_ca", 0)
        p_no = od.get("no_ca_pct", 0)
        n_ca = od.get("ca", 0)
        p_ca = od.get("ca_pct", 0)
        delta = n_ca - n_no
        lines.append(f"| {name} | {n_no} | {p_no:.1f}% | {n_ca} | {p_ca:.1f}% | {delta:+d} |")
    lines.append(f"| 无层可覆盖 | 0 | 0% | {no_fit_count} | "
                 f"{no_fit_count/max(sum(od.get('ca',0) for od in overall_dist.values())+no_fit_count,1)*100:.1f}% | - |")
    lines.append("")

    # 按长度桶
    lines.append("### 3.2 按长度桶的层分布（有 CA）")
    lines.append("")
    layer_headers = " | ".join(f"P{int(np.log2(s))}" for s in strides)
    lines.append(f"| 长度桶 | GT 数 | {layer_headers} | 无层可覆盖 |")
    sep = "|:------:|------:|" + "|".join(":---:" for _ in strides) + "|:----------:|"
    lines.append(sep)
    for bname in [b[2] for b in LENGTH_BUCKETS]:
        bd = bucket_layer_dist.get(bname, {})
        n = bd.get("n", 0)
        if n == 0:
            lines.append(f"| {bname} | 0 | " + " | ".join("-" for _ in strides) + " | - |")
            continue
        dist = bd.get("dist", {})
        vals = " | ".join(
            f"{dist.get(s,{}).get('count',0)} ({dist.get(s,{}).get('pct',0):.0f}%)"
            for s in strides
        )
        nf = bd.get("no_fit", 0)
        nf_pct = bd.get("no_fit_pct", 0)
        lines.append(f"| {bname} | {n} | {vals} | {nf} ({nf_pct:.1f}%) |")
    lines.append("")

    # 按长宽比桶
    lines.append("### 3.3 按长宽比桶的层分布（有 CA）")
    lines.append("")
    lines.append(f"| 长宽比桶 | GT 数 | {layer_headers} | 无层可覆盖 |")
    lines.append(sep.replace("长度桶", "长宽比桶"))
    for bname in [b[2] for b in AR_BUCKETS]:
        ad = ar_layer_dist.get(bname, {})
        n = ad.get("n", 0)
        if n == 0:
            lines.append(f"| {bname} | 0 | " + " | ".join("-" for _ in strides) + " | - |")
            continue
        dist = ad.get("dist", {})
        vals = " | ".join(
            f"{dist.get(s,{}).get('count',0)} ({dist.get(s,{}).get('pct',0):.0f}%)"
            for s in strides
        )
        nf = ad.get("no_fit", 0)
        nf_pct = ad.get("no_fit_pct", 0)
        lines.append(f"| {bname} | {n} | {vals} | {nf} ({nf_pct:.1f}%) |")
    lines.append("")

    # 极端目标
    lines.append("### 3.4 极端目标分析（长度≥512 或 AR≥60）")
    lines.append("")
    n_ext = extreme_stats.get("n", 0)
    if n_ext > 0:
        lines.append(f"- 极端目标总数: **{n_ext}**")
        lines.append(f"- 仍停留在 P{int(np.log2(strides[0]))}: "
                     f"**{extreme_stats.get('stuck_p3',0)}** "
                     f"({extreme_stats.get('stuck_p3',0)/n_ext*100:.1f}%)")
        lines.append(f"- 无层可覆盖（fallback 到 P{int(np.log2(strides[-1]))}）: "
                     f"**{extreme_stats.get('no_fit',0)}** "
                     f"({extreme_stats.get('no_fit',0)/n_ext*100:.1f}%)")
    else:
        lines.append("- 无极端目标")
    lines.append("")

    lines.append("**解读**：")
    lines.append("- 如果 >512 / >60 桶仍有大量 GT 停留在 P3 → Coverage-Aware TAL 约束未到位")
    lines.append("- 如果\"无层可覆盖\"比例高 → 即使最大 stride (P5) 也不够，需要更大 reg_max 或更深的 FPN 层")
    lines.append("- 理想情况：极端目标 100% 转移到 P5，且 P5 的饱和率为 0")
    lines.append("")

    # ---- 综合结论 ----
    lines.append("## 综合排查结论")
    lines.append("")
    lines.append("_(根据上述数据填写)_")
    lines.append("")

    # 自动生成一些判断
    top_gain_key = sorted(oracle_results.keys(),
                           key=lambda k: oracle_results[k].get("gain75", 0), reverse=True)[0] if oracle_results else ""
    top_r = oracle_results.get(top_gain_key, {})
    lines.append(f"1. **主瓶颈判定**：AP75 提升最大的 Oracle 替换 = \"{top_r.get('label', 'N/A')}\" "
                 f"(+{top_r.get('gain75',0):.4f})")
    if rl.get('gain75', 0) < rw.get('gain75', 0) * 0.5:
        lines.append("   → 当前主瓶颈是**\"画不细、画不正\"**，而非\"画不长\"")
    lines.append("")

    # 层间差异
    if layer_results:
        ap75_by_layer = {s: layer_results[s]["ap75"] for s in strides if s in layer_results}
        if ap75_by_layer:
            best_s = max(ap75_by_layer, key=ap75_by_layer.get)
            worst_s = min(ap75_by_layer, key=ap75_by_layer.get)
            lines.append(f"2. **层间差异**：AP75 最好层 = P{int(np.log2(best_s))} "
                         f"({ap75_by_layer[best_s]:.4f})，"
                         f"最差层 = P{int(np.log2(worst_s))} "
                         f"({ap75_by_layer[worst_s]:.4f})")
            if ap75_by_layer[best_s] - ap75_by_layer[worst_s] > 0.1:
                lines.append("   → 层间差异显著，暗示需要**双尺度解耦**策略")
    lines.append("")

    lines.append(f"3. **CA TAL 有效性**：")
    if n_ext > 0:
        stuck_pct = extreme_stats.get('stuck_p3', 0) / n_ext * 100
        if stuck_pct > 10:
            lines.append(f"   - 仍有 {stuck_pct:.0f}% 极端目标留在浅层 → CA 约束还需加强")
        else:
            lines.append(f"   - 极端目标已基本转移到深层 → CA 约束有效")
    no_fit_pct = extreme_stats.get('no_fit', 0) / max(n_ext, 1) * 100
    if no_fit_pct > 20:
        lines.append(f"   - {no_fit_pct:.0f}% 极端目标无层可覆盖 → reg_max 仍不足或需要 P6 层")
    lines.append("")

    report_path = os.path.join(output_dir, "oracle_fpn_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n报告已保存: {report_path}")
    return report_path


# ========================== 主函数 ==========================

def main():
    config = CONFIG.copy()
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 收集数据
    all_data, class_names, model_info = collect_data(config)

    if not all_data:
        print("[!] 没有收集到数据，退出。")
        return

    # 排查 1：Oracle 几何替换实验
    check1_result = check1_oracle_experiment(all_data)

    # 排查 2：按 FPN 层分桶分析
    check2_result = check2_per_layer_analysis(all_data, model_info)

    # 排查 3：Coverage-Aware TAL 验证
    check3_result = check3_ca_verification(all_data, model_info)

    # 生成报告
    generate_report(config, check1_result, check2_result, check3_result, model_info, output_dir)

    print("\n" + "=" * 60)
    print("  排查完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
