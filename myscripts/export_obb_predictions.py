"""
导出 OBB 预测结果与 GT 标注脚本

功能：
    1. 加载已训练好的 YOLO11-OBB 模型权重
    2. 在验证集上运行推理，收集每张图的预测结果
    3. 导出预测结果和 GT 标注到 txt 文件

输出格式：
    预测结果 (predictions/): 每行 x y w h theta conf class_id
    GT 标注 (ground_truths/): 每行 x y w h theta class_id

使用方法：
    python export_obb_predictions.py \
        --weights /path/to/best.pt \
        --data /path/to/dataset.yaml \
        --output-dir ./tnms_data \
        --imgsz 1024 \
        --conf-thres 0.01 \
        --device 0
"""

import argparse
import os
os.environ["OMP_NUM_THREADS"] = "8"  # 修复 libgomp 警告
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# 将项目根目录加入 sys.path，确保可以导入 ultralytics
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="导出 OBB 预测结果与 GT 标注")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="训练好的模型权重路径 (.pt 文件)",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="数据集配置文件路径 (.yaml 文件)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./tnms_data",
        help="输出目录路径 (默认: ./tnms_data)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="推理输入图片尺寸 (默认: 1024)",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.01,
        help="置信度过滤阈值，设得很低以保留更多预测框 (默认: 0.01)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU 设备编号 (默认: 0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="推理批次大小 (默认: 16)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test", "train"],
        help="使用哪个数据划分进行推理 (默认: val)",
    )
    return parser.parse_args()


def export_predictions_and_gt(args):
    """
    导出预测结果和 GT 标注。

    使用 YOLO 的 val() 接口跑验证集，然后从 save_txt 结果中收集数据。
    同时从数据集标签目录中收集 GT 标注。
    """
    output_dir = Path(args.output_dir)
    pred_dir = output_dir / "predictions"
    gt_dir = output_dir / "ground_truths"
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  模型权重: {args.weights}")
    print(f"  数据集: {args.data}")
    print(f"  输出目录: {output_dir}")
    print(f"  图片尺寸: {args.imgsz}")
    print(f"  置信度阈值: {args.conf_thres}")
    print(f"  数据划分: {args.split}")
    print("=" * 60)

    # 加载模型
    print("\n[1/3] 加载模型...")
    model = YOLO(args.weights)

    # 运行验证，开启 save_txt 以保存预测结果
    print("\n[2/3] 运行推理并导出预测结果...")
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch_size,
        conf=args.conf_thres,
        device=args.device,
        split=args.split,
        save_txt=True,      # 保存预测结果到 txt
        save_conf=True,      # txt 中包含置信度
        plots=False,         # 不需要画图
        verbose=True,
    )

    # 获取验证结果保存目录
    save_dir = Path(results.save_dir)
    labels_dir = save_dir / "labels"

    if labels_dir.exists():
        print(f"\n  预测结果 txt 已由 YOLO 保存到: {labels_dir}")
        # 将预测结果复制到我们的输出目录
        txt_files = list(labels_dir.glob("*.txt"))
        for txt_file in tqdm(txt_files, desc="  复制预测结果"):
            # 读取 YOLO 保存的 OBB txt 格式并转换
            # YOLO OBB txt 格式: class_id x1 y1 x2 y2 x3 y3 x4 y4 [conf]
            # 我们需要转换为: x y w h theta conf class_id
            convert_yolo_obb_txt_to_xywht(txt_file, pred_dir / txt_file.name)
        print(f"  已导出 {len(txt_files)} 个预测结果文件到: {pred_dir}")
    else:
        print(f"  警告: 预测结果目录不存在: {labels_dir}")
        print("  尝试使用备用方案直接推理...")
        export_predictions_direct(model, args, pred_dir)

    # 导出 GT 标注
    print("\n[3/3] 导出 GT 标注...")
    export_ground_truths(model, args, gt_dir)

    print("\n" + "=" * 60)
    print("  导出完成！")
    print(f"  预测结果目录: {pred_dir}")
    print(f"  GT 标注目录: {gt_dir}")
    print("=" * 60)

    return pred_dir, gt_dir


def convert_yolo_obb_txt_to_xywht(src_file, dst_file):
    """
    将 YOLO OBB 的多边形 txt 格式转换为 xywht 格式。

    YOLO OBB 保存格式: class_id x1 y1 x2 y2 x3 y3 x4 y4 [conf]
    目标格式: x y w h theta conf class_id
    """
    lines = []
    with open(src_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue

            class_id = int(parts[0])
            coords = [float(p) for p in parts[1:9]]
            conf = float(parts[9]) if len(parts) > 9 else 1.0

            # 将4个顶点坐标转换为 (x, y, w, h, theta)
            points = np.array(coords).reshape(4, 2)
            x, y, w, h, theta = points_to_xywht(points)

            lines.append(f"{x:.4f} {y:.4f} {w:.4f} {h:.4f} {theta:.6f} {conf:.6f} {class_id}\n")

    with open(dst_file, "w") as f:
        f.writelines(lines)


def points_to_xywht(points):
    """
    将4个顶点坐标转换为 (x_center, y_center, width, height, theta) 格式。

    Args:
        points: shape (4, 2) 的数组，表示4个顶点坐标

    Returns:
        (x, y, w, h, theta): 中心点坐标、宽、高、旋转角度（弧度）
    """
    # 计算中心点
    x = np.mean(points[:, 0])
    y = np.mean(points[:, 1])

    # 计算相邻两边的长度
    edge1 = np.linalg.norm(points[1] - points[0])
    edge2 = np.linalg.norm(points[2] - points[1])

    # 短边为 w（宽度），长边为 h（高度/长度）
    if edge1 >= edge2:
        h = edge1
        w = edge2
        # 角度由长边方向决定
        dx = points[1][0] - points[0][0]
        dy = points[1][1] - points[0][1]
    else:
        h = edge2
        w = edge1
        dx = points[2][0] - points[1][0]
        dy = points[2][1] - points[1][1]

    theta = np.arctan2(dy, dx)  # 与 X 轴的夹角（弧度）

    return x, y, w, h, theta


def export_predictions_direct(model, args, pred_dir):
    """
    备用方案：直接使用 predict 接口逐图推理并保存结果。

    当 val() 的 save_txt 结果不可用时使用此方案。
    """
    import yaml

    # 读取数据集配置
    with open(args.data, "r") as f:
        data_cfg = yaml.safe_load(f)

    # 获取验证集图片路径
    data_root = Path(args.data).parent
    val_path = data_cfg.get(args.split, data_cfg.get("val", ""))
    if not Path(val_path).is_absolute():
        val_path = data_root / val_path

    # 收集所有图片
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if Path(val_path).is_dir():
        images_dir = Path(val_path) / "images" if (Path(val_path) / "images").exists() else Path(val_path)
        img_files = [f for f in images_dir.iterdir() if f.suffix.lower() in img_extensions]
    else:
        img_files = [Path(val_path)]

    print(f"  找到 {len(img_files)} 张图片，开始逐图推理...")

    for img_file in tqdm(img_files, desc="  推理中"):
        results = model.predict(
            source=str(img_file),
            imgsz=args.imgsz,
            conf=args.conf_thres,
            device=args.device,
            verbose=False,
        )

        # 提取 OBB 结果
        for result in results:
            if result.obb is not None and len(result.obb) > 0:
                obb_data = result.obb.data.cpu().numpy()  # [x, y, w, h, theta, conf, cls]
                stem = Path(result.path).stem
                out_file = pred_dir / f"{stem}.txt"

                lines = []
                for det in obb_data:
                    x, y, w_det, h_det, theta = det[0], det[1], det[2], det[3], det[4]
                    conf, cls_id = det[5], int(det[6])
                    # 确保 h >= w（长边为 h）
                    if w_det > h_det:
                        w_det, h_det = h_det, w_det
                        theta = theta + np.pi / 2
                    lines.append(f"{x:.4f} {y:.4f} {w_det:.4f} {h_det:.4f} {theta:.6f} {conf:.6f} {cls_id}\n")

                with open(out_file, "w") as f:
                    f.writelines(lines)

    print(f"  已导出预测结果到: {pred_dir}")


def export_ground_truths(model, args, gt_dir):
    """
    导出验证集的 GT 标注。

    从数据集标签目录中读取 YOLO OBB 格式的标注文件并转换。
    """
    import yaml

    # 读取数据集配置
    with open(args.data, "r") as f:
        data_cfg = yaml.safe_load(f)

    # 获取标签路径
    data_root = Path(args.data).parent
    val_path = data_cfg.get(args.split, data_cfg.get("val", ""))
    if not Path(val_path).is_absolute():
        val_path = data_root / val_path

    # 推断标签目录
    val_path = Path(val_path)
    val_str = str(val_path)
    
    # YOLO 常用目录结构替换规则：把最后出现的 'images' 替换为 'labels'
    if 'images' in val_str:
        labels_dir_str = 'labels'.join(val_str.rsplit('images', 1))
        labels_dir = Path(labels_dir_str)
    elif (val_path / "images").exists() and (val_path / "labels").exists():
        labels_dir = val_path / "labels"
    else:
        labels_dir = val_path.parent / "labels"

    if not labels_dir.exists():
        print(f"  警告: 标签目录不存在: {labels_dir}")
        print("  尝试在图片目录同级寻找 labels 目录...")
        # 尝试其他常见路径
        for candidate in [val_path / "labels", val_path.parent / "labels"]:
            if candidate.exists():
                labels_dir = candidate
                break
        else:
            print(f"  错误: 无法找到标签目录！尝试过的路径: {labels_dir.parent} 下的 labels")
            return

    print(f"  GT 标签目录: {labels_dir}")

    label_files = list(labels_dir.glob("*.txt"))
    print(f"  找到 {len(label_files)} 个标签文件")

    for label_file in tqdm(label_files, desc="  导出 GT"):
        lines_out = []
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue

                class_id = int(parts[0])
                coords = [float(p) for p in parts[1:9]]

                # 将4个顶点坐标转换为 (x, y, w, h, theta)
                points = np.array(coords).reshape(4, 2)
                x, y, w, h, theta = points_to_xywht(points)

                lines_out.append(f"{x:.4f} {y:.4f} {w:.4f} {h:.4f} {theta:.6f} {class_id}\n")

        if lines_out:
            with open(gt_dir / label_file.name, "w") as f:
                f.writelines(lines_out)

    exported_count = len(list(gt_dir.glob("*.txt")))
    print(f"  已导出 {exported_count} 个 GT 标注文件到: {gt_dir}")


if __name__ == "__main__":
    args = parse_args()
    export_predictions_and_gt(args)
