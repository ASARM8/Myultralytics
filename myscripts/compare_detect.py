"""
双模型 OBB 检测对比脚本
功能：用两个模型检测同一张图片，输出 2×2 对比图（原图 / GT / 模型A / 模型B）

使用方法:
    python myscripts/compare_detect.py

配置说明:
    修改下方 CONFIG 字典即可：
    - model_a / model_b: 两个模型权重路径（.pt）
    - label_a / label_b: 对应的模型标签名（用于标题显示）
    - image_dir: 待检测图片目录（或单张图片路径）
    - gt_label_dir: GT 标注目录（YOLO OBB 格式：class x1 y1 x2 y2 x3 y3 x4 y4，归一化坐标）
    - class_names: 类别名称列表
    - max_images: 最多处理图片数（0 表示全部）
    - output_dir: 对比图输出目录
"""

import os
import sys
import glob
import numpy as np
import cv2
from pathlib import Path

# 添加项目根目录到 sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ultralytics import YOLO

# ========================== 配置 ==========================
CONFIG = {
    # ---------- 模型配置 ----------
    # 模型 A（例如原版 reg_max=16）
    "model_a": "/root/autodl-tmp/work-dirs/yolo11_obb-v2.1full/weights/best.pt",
    "label_a": "Baseline (reg_max=16)",

    # 模型 B（例如 CA 版 reg_max=32 + Coverage-Aware）
    "model_b": "/root/autodl-tmp/work-dirs/yolo11_obb-ca/weights/best.pt",
    "label_b": "CA (reg_max=32)",

    # ---------- 数据配置 ----------
    # 图片目录或单张图片路径
    "image_dir": "/root/autodl-tmp/dataset/TTPLA-1024/images/val",
    # GT 标注目录（与图片同名 .txt，YOLO OBB 格式）
    # 留空则不绘制 GT
    "gt_label_dir": "/root/autodl-tmp/dataset/TTPLA-1024/labels/val",
    # 类别名称
    "class_names": ["cable"],

    # ---------- 推理参数 ----------
    "conf": 0.25,               # 置信度阈值
    "iou": 0.45,                # NMS IoU 阈值
    "imgsz": 1024,              # 推理图片尺寸
    "device": 0,                # GPU 设备

    # ---------- 输出配置 ----------
    "output_dir": "/root/autodl-tmp/work-dirs/compare_detect_output",
    "max_images": 20,           # 最多处理图片数（0=全部）
    "line_width": 2,            # 检测框线宽
    "font_scale": 0.6,          # 字体大小
    "title_height": 40,         # 标题栏高度（像素）
}

# ========================== 颜色配置 ==========================
# GT 用绿色，模型 A 用蓝色，模型 B 用红色
COLOR_GT = (0, 255, 0)       # 绿色 (BGR)
COLOR_A = (255, 165, 0)      # 橙色 (BGR)
COLOR_B = (0, 100, 255)      # 红色偏橙 (BGR)
COLOR_TITLE_BG = (40, 40, 40)  # 标题背景深灰
COLOR_TITLE_FG = (255, 255, 255)  # 标题白色文字


def load_gt_labels(label_path, img_w, img_h):
    """加载 YOLO OBB 格式的 GT 标注。

    YOLO OBB 格式：class x1 y1 x2 y2 x3 y3 x4 y4（归一化坐标）

    Args:
        label_path: 标注文件路径
        img_w: 图片宽度
        img_h: 图片高度

    Returns:
        list of (class_id, points)，points 为 (4, 2) 的像素坐标数组
    """
    labels = []
    if not os.path.exists(label_path):
        return labels

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            cls_id = int(parts[0])
            coords = list(map(float, parts[1:9]))
            # 归一化 → 像素坐标
            points = np.array(coords).reshape(4, 2)
            points[:, 0] *= img_w
            points[:, 1] *= img_h
            labels.append((cls_id, points.astype(np.int32)))
    return labels


def draw_obb_boxes(img, boxes, color, line_width=2, font_scale=0.5, label_prefix=""):
    """在图片上绘制旋转框。

    Args:
        img: BGR 图片（会被就地修改）
        boxes: list of (class_id, points, conf) 或 (class_id, points)
        color: BGR 颜色元组
        line_width: 线宽
        font_scale: 字体缩放
        label_prefix: 标签前缀
    """
    for box in boxes:
        if len(box) == 3:
            cls_id, pts, conf = box
            label = f"{label_prefix}{conf:.2f}"
        else:
            cls_id, pts = box
            conf = None
            label = label_prefix if label_prefix else ""

        pts = pts.astype(np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=line_width)

        if label:
            # 在第一个顶点上方绘制标签
            x, y = pts[0]
            y_text = max(y - 5, 15)
            cv2.putText(img, label, (x, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
    return img


def add_title_bar(img, title, height=40, bg_color=COLOR_TITLE_BG, fg_color=COLOR_TITLE_FG):
    """在图片顶部添加标题栏。"""
    h, w = img.shape[:2]
    bar = np.full((height, w, 3), bg_color, dtype=np.uint8)
    # 计算文字位置（居中）
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (tw, th), _ = cv2.getTextSize(title, font, font_scale, thickness)
    x = (w - tw) // 2
    y = (height + th) // 2
    cv2.putText(bar, title, (x, y), font, font_scale, fg_color, thickness, cv2.LINE_AA)
    return np.vstack([bar, img])


def extract_obb_results(results):
    """从 Ultralytics 推理结果中提取 OBB 信息。

    Returns:
        list of (class_id, points_array(4,2), confidence)
    """
    boxes = []
    if results and results[0].obb is not None:
        obb = results[0].obb
        for i in range(len(obb)):
            cls_id = int(obb.cls[i].item())
            conf = float(obb.conf[i].item())
            # xyxyxyxy: (N, 4, 2) 像素坐标
            pts = obb.xyxyxyxy[i].cpu().numpy().astype(np.int32)
            boxes.append((cls_id, pts, conf))
    return boxes


def process_single_image(img_path, model_a, model_b, gt_label_dir, config):
    """处理单张图片，返回 2×2 对比图。"""
    # 读取原图
    img = cv2.imread(img_path)
    if img is None:
        print(f"  [!] 无法读取图片: {img_path}")
        return None
    img_h, img_w = img.shape[:2]
    img_name = os.path.basename(img_path)

    lw = config["line_width"]
    fs = config["font_scale"]
    th = config["title_height"]

    # --- 面板 1：原图 ---
    panel_orig = img.copy()
    panel_orig = add_title_bar(panel_orig, f"Original: {img_name}", th)

    # --- 面板 2：GT ---
    panel_gt = img.copy()
    gt_boxes = []
    if gt_label_dir:
        stem = os.path.splitext(img_name)[0]
        label_path = os.path.join(gt_label_dir, stem + ".txt")
        gt_labels = load_gt_labels(label_path, img_w, img_h)
        for cls_id, pts in gt_labels:
            gt_boxes.append((cls_id, pts))
        draw_obb_boxes(panel_gt, gt_boxes, COLOR_GT, lw, fs, label_prefix="GT ")
    panel_gt = add_title_bar(panel_gt, f"Ground Truth ({len(gt_boxes)} objs)", th)

    # --- 面板 3：模型 A 检测 ---
    panel_a = img.copy()
    results_a = model_a.predict(
        img_path, conf=config["conf"], iou=config["iou"],
        imgsz=config["imgsz"], device=config["device"], verbose=False
    )
    boxes_a = extract_obb_results(results_a)
    draw_obb_boxes(panel_a, boxes_a, COLOR_A, lw, fs)
    panel_a = add_title_bar(panel_a, f"{config['label_a']} ({len(boxes_a)} dets)", th)

    # --- 面板 4：模型 B 检测 ---
    panel_b = img.copy()
    results_b = model_b.predict(
        img_path, conf=config["conf"], iou=config["iou"],
        imgsz=config["imgsz"], device=config["device"], verbose=False
    )
    boxes_b = extract_obb_results(results_b)
    draw_obb_boxes(panel_b, boxes_b, COLOR_B, lw, fs)
    panel_b = add_title_bar(panel_b, f"{config['label_b']} ({len(boxes_b)} dets)", th)

    # --- 拼接 2×2 ---
    top_row = np.hstack([panel_orig, panel_gt])
    bot_row = np.hstack([panel_a, panel_b])
    combined = np.vstack([top_row, bot_row])

    return combined


def main():
    config = CONFIG.copy()
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 收集图片列表
    image_dir = config["image_dir"]
    if os.path.isfile(image_dir):
        image_paths = [image_dir]
    else:
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        image_paths = []
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        image_paths.sort()

    if config["max_images"] > 0:
        image_paths = image_paths[:config["max_images"]]

    print(f"找到 {len(image_paths)} 张图片")
    if not image_paths:
        print("[!] 没有找到图片，请检查 image_dir 配置")
        return

    # 加载模型
    print(f"加载模型 A: {config['model_a']}")
    model_a = YOLO(config["model_a"])
    print(f"加载模型 B: {config['model_b']}")
    model_b = YOLO(config["model_b"])

    # 逐张处理
    for i, img_path in enumerate(image_paths):
        print(f"[{i+1}/{len(image_paths)}] 处理: {os.path.basename(img_path)}")
        combined = process_single_image(img_path, model_a, model_b, config["gt_label_dir"], config)
        if combined is None:
            continue

        # 保存
        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(output_dir, f"compare_{stem}.jpg")
        cv2.imwrite(out_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"\n完成！对比图已保存到: {output_dir}")
    print(f"共处理 {len(image_paths)} 张图片")


if __name__ == "__main__":
    main()
