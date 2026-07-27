from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.utils import ops
from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.nms import TorchNMS


@dataclass
class TNMSConfig:
    """T-NMS 参数配置。"""

    angle_thr_deg: float = 8.0
    perp_thr: float = 8.0
    gap_thr: float = 20.0
    conf_thr: float = 0.001
    pre_nms_iou: float = 0.0
    class_aware: bool = True

    @property
    def angle_thr_rad(self) -> float:
        """角度阈值（弧度）。"""
        return math.radians(self.angle_thr_deg)


def normalize_theta_pi(theta: float) -> float:
    """将角度归一化到 [-pi/2, pi/2)。"""
    return (theta + math.pi / 2) % math.pi - math.pi / 2


def min_periodic_angle_diff(theta1: float, theta2: float) -> float:
    """计算以 pi 为周期的最小夹角差。"""
    diff = abs(theta1 - theta2) % math.pi
    return min(diff, math.pi - diff)


def long_axis_repr(box_xywhr: np.ndarray) -> tuple[float, float, float]:
    """将 OBB 变换为长轴表达：长轴长度、短轴长度、长轴方向角。"""
    _, _, w, h, theta = [float(v) for v in box_xywhr]
    if w >= h:
        long_len, short_len, theta_long = w, h, theta
    else:
        long_len, short_len, theta_long = h, w, theta + math.pi / 2
    return long_len, short_len, normalize_theta_pi(theta_long)


def segment_endpoints(box_xywhr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """根据 OBB 的长轴表达提取两端端点。"""
    cx, cy = float(box_xywhr[0]), float(box_xywhr[1])
    long_len, _, theta = long_axis_repr(box_xywhr)
    half = 0.5 * long_len
    dx = half * math.cos(theta)
    dy = half * math.sin(theta)
    p1 = np.array([cx - dx, cy - dy], dtype=np.float64)
    p2 = np.array([cx + dx, cy + dy], dtype=np.float64)
    return p1, p2


def can_merge(seed: np.ndarray, target: np.ndarray, cfg: TNMSConfig) -> bool:
    """判断两个 OBB 是否满足 T-NMS 三大拓扑融合条件。"""
    if cfg.class_aware and int(round(seed[6])) != int(round(target[6])):
        return False

    long1, _, theta1 = long_axis_repr(seed[:5])
    long2, _, theta2 = long_axis_repr(target[:5])

    delta_theta = min_periodic_angle_diff(theta1, theta2)
    if delta_theta > cfg.angle_thr_rad:
        return False

    dx = float(target[0] - seed[0])
    dy = float(target[1] - seed[1])
    cos_t = math.cos(theta1)
    sin_t = math.sin(theta1)

    d_perp = abs(dy * cos_t - dx * sin_t)
    if d_perp > cfg.perp_thr:
        return False

    d_para = abs(dx * cos_t + dy * sin_t)
    return d_para <= 0.5 * (long1 + long2) + cfg.gap_thr


def merge_two_boxes(seed: np.ndarray, target: np.ndarray) -> np.ndarray:
    """将两个满足条件的 OBB 融合为一个更长的 OBB。"""
    long1, short1, theta1 = long_axis_repr(seed[:5])
    _, short2, _ = long_axis_repr(target[:5])

    p1a, p1b = segment_endpoints(seed[:5])
    p2a, p2b = segment_endpoints(target[:5])
    points = np.stack([p1a, p1b, p2a, p2b], axis=0)

    diff = points[:, None, :] - points[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    i, j = np.unravel_index(int(np.argmax(dist2)), dist2.shape)
    p_start, p_end = points[i], points[j]

    center = 0.5 * (p_start + p_end)
    vec = p_end - p_start
    long_new = float(np.linalg.norm(vec))
    if long_new < 1e-6:
        long_new = max(long1, float(long_axis_repr(target[:5])[0]))
        theta_new = theta1
    else:
        theta_new = normalize_theta_pi(float(math.atan2(vec[1], vec[0])))

    short_new = max(short1, short2)
    conf_new = float(max(seed[5], target[5]))
    cls_new = float(seed[6] if seed[5] >= target[5] else target[6])

    return np.array(
        [
            float(center[0]),
            float(center[1]),
            long_new,
            short_new,
            theta_new,
            conf_new,
            cls_new,
        ],
        dtype=np.float32,
    )


def regularize_rboxes_np(boxes: np.ndarray) -> np.ndarray:
    """调用 Ultralytics 官方 regularize_rboxes 规整角度与边长。"""
    if boxes.size == 0:
        return boxes.astype(np.float32, copy=False)
    out = boxes.astype(np.float32, copy=True)
    reg = ops.regularize_rboxes(torch.as_tensor(out[:, :5], dtype=torch.float32)).cpu().numpy()
    out[:, :5] = reg
    return out


def apply_official_rotated_nms(boxes: np.ndarray, iou_thres: float, class_aware: bool) -> np.ndarray:
    """调用 Ultralytics 官方 TorchNMS.fast_nms + batch_probiou 做预去重。"""
    if boxes.size == 0 or iou_thres <= 0:
        return boxes.astype(np.float32, copy=False)

    nms_boxes = boxes[:, :5].astype(np.float32, copy=True)
    scores = boxes[:, 5].astype(np.float32, copy=False)

    if class_aware:
        max_wh = float(np.max(np.abs(nms_boxes[:, :2]))) * 2.0 + 1.0
        nms_boxes[:, :2] += boxes[:, 6:7].astype(np.float32) * max_wh

    keep = TorchNMS.fast_nms(
        boxes=torch.as_tensor(nms_boxes, dtype=torch.float32),
        scores=torch.as_tensor(scores, dtype=torch.float32),
        iou_threshold=iou_thres,
        iou_func=batch_probiou,
    )
    return boxes[keep.cpu().numpy()]


def topology_nms_single_image(boxes: np.ndarray, cfg: TNMSConfig) -> np.ndarray:
    """对单张图像的预测框执行 T-NMS。"""
    if boxes.size == 0:
        return np.zeros((0, 7), dtype=np.float32)

    boxes = boxes.astype(np.float32, copy=True)
    boxes = boxes[boxes[:, 5] >= cfg.conf_thr]
    if boxes.size == 0:
        return np.zeros((0, 7), dtype=np.float32)

    boxes = regularize_rboxes_np(boxes)
    if cfg.pre_nms_iou > 0:
        boxes = apply_official_rotated_nms(boxes, cfg.pre_nms_iou, cfg.class_aware)
        if boxes.size == 0:
            return np.zeros((0, 7), dtype=np.float32)

    order = np.argsort(-boxes[:, 5])
    pool = [boxes[i].copy() for i in order.tolist()]
    keep: list[np.ndarray] = []

    while pool:
        seed = pool.pop(0)
        changed = True
        while changed and pool:
            changed = False
            idx = 0
            while idx < len(pool):
                target = pool[idx]
                if can_merge(seed, target, cfg):
                    seed = merge_two_boxes(seed, pool.pop(idx))
                    changed = True
                    idx = 0
                else:
                    idx += 1
        keep.append(seed)

    merged = np.stack(keep, axis=0).astype(np.float32)
    merged = regularize_rboxes_np(merged)
    return merged[np.argsort(-merged[:, 5])]


def _validate_boxes(arr: np.ndarray, source: str) -> np.ndarray:
    """校验并标准化输入框数组，形状应为 (N, 7+)。"""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 7), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 7:
        raise ValueError(f"输入格式错误：{source} 中的数据应为 (N, 7+)。实际形状: {arr.shape}")
    return arr[:, :7]


def load_predictions_json(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """读取 Ultralytics 风格 predictions.json（含 rbox/score/category_id）。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON 输入应为列表结构。")

    grouped: dict[str, list[list[float]]] = defaultdict(list)
    image_to_file: dict[str, str] = {}

    for i, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"JSON 第 {i} 项不是对象。")
        if "rbox" not in item:
            continue

        image_id = str(item.get("image_id", "image_0"))
        rbox = item["rbox"]
        if not isinstance(rbox, (list, tuple)) or len(rbox) != 5:
            raise ValueError(f"JSON 第 {i} 项 rbox 格式错误，应为长度 5。")

        score = float(item.get("score", item.get("conf", 0.0)))
        cls = float(item.get("category_id", item.get("cls", item.get("class", 0))))
        grouped[image_id].append([float(rbox[0]), float(rbox[1]), float(rbox[2]), float(rbox[3]), float(rbox[4]), score, cls])

        if "file_name" in item:
            image_to_file[image_id] = str(item["file_name"])

    return {k: _validate_boxes(v, f"JSON:{k}") for k, v in grouped.items()}, image_to_file


def load_predictions_txt(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """读取 txt/csv 文本预测。

    支持两种格式：
    1) x y w h theta conf cls
    2) image_id x y w h theta conf cls
    """
    grouped: dict[str, list[list[float]]] = defaultdict(list)

    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.replace(",", " ").split()
        if len(parts) == 7:
            image_id = "image_0"
            nums = parts
        elif len(parts) >= 8:
            image_id = str(parts[0])
            nums = parts[1:8]
        else:
            raise ValueError(f"第 {lineno} 行格式错误，至少需要 7 列数值。")

        try:
            values = [float(x) for x in nums]
        except ValueError as exc:
            raise ValueError(f"第 {lineno} 行存在无法解析的数字。") from exc

        grouped[image_id].append(values)

    return {k: _validate_boxes(v, f"TXT:{k}") for k, v in grouped.items()}, {}


def load_predictions_npy(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """读取 npy 预测，支持 (N, 7) 或 dict[str, (N, 7)]。"""
    raw = np.load(path, allow_pickle=True)

    if isinstance(raw, np.ndarray) and raw.dtype != object:
        return {"image_0": _validate_boxes(raw, "NPY:image_0")}, {}

    if isinstance(raw, np.ndarray) and raw.shape == ():
        obj: Any = raw.item()
        if isinstance(obj, dict):
            out: dict[str, np.ndarray] = {}
            for k, v in obj.items():
                out[str(k)] = _validate_boxes(v, f"NPY:{k}")
            return out, {}

    raise ValueError("NPY 输入格式不支持，请使用 (N,7) 数组或 dict[str, (N,7)]。")


def load_predictions(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """按文件后缀自动读取预测数据。"""
    suffix = path.suffix.lower()
    if suffix == ".json":
        return load_predictions_json(path)
    if suffix in {".txt", ".csv"}:
        return load_predictions_txt(path)
    if suffix == ".npy":
        return load_predictions_npy(path)
    raise ValueError(f"不支持的输入格式: {suffix}，请使用 json/txt/csv/npy。")


def save_predictions_json(
    path: Path,
    merged: dict[str, np.ndarray],
    image_to_file: dict[str, str],
    save_poly: bool,
) -> None:
    """保存为 Ultralytics 风格 JSON，支持可选 poly 输出。"""
    records: list[dict[str, Any]] = []

    for image_id in sorted(merged.keys()):
        boxes = merged[image_id]
        for row in boxes:
            rec: dict[str, Any] = {
                "image_id": image_id,
                "score": round(float(row[5]), 5),
                "category_id": int(round(float(row[6]))),
                "rbox": [round(float(x), 3) for x in row[:5]],
            }
            if image_id in image_to_file:
                rec["file_name"] = image_to_file[image_id]
            if save_poly:
                poly = ops.xywhr2xyxyxyxy(torch.as_tensor(row[:5], dtype=torch.float32).view(1, 5)).view(-1).tolist()
                rec["poly"] = [round(float(x), 3) for x in poly]
            records.append(rec)

    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def save_predictions_txt(path: Path, merged: dict[str, np.ndarray]) -> None:
    """保存为文本格式：image_id x y w h theta conf cls。"""
    lines: list[str] = []
    for image_id in sorted(merged.keys()):
        boxes = merged[image_id]
        for row in boxes:
            cls = int(round(float(row[6])))
            line = (
                f"{image_id} "
                f"{float(row[0]):.6f} {float(row[1]):.6f} {float(row[2]):.6f} {float(row[3]):.6f} "
                f"{float(row[4]):.6f} {float(row[5]):.6f} {cls}"
            )
            lines.append(line)
    content = "\n".join(lines)
    path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def save_predictions_npy(path: Path, merged: dict[str, np.ndarray]) -> None:
    """保存为 npy（dict[str, np.ndarray]）。"""
    np.save(path, {k: v.astype(np.float32) for k, v in merged.items()}, allow_pickle=True)


def save_predictions(path: Path, merged: dict[str, np.ndarray], image_to_file: dict[str, str], save_poly: bool) -> None:
    """按后缀自动保存结果。"""
    suffix = path.suffix.lower()
    if suffix == ".json":
        save_predictions_json(path, merged, image_to_file, save_poly=save_poly)
        return
    if suffix in {".txt", ".csv"}:
        save_predictions_txt(path, merged)
        return
    if suffix == ".npy":
        save_predictions_npy(path, merged)
        return
    raise ValueError(f"不支持的输出格式: {suffix}，请使用 json/txt/csv/npy。")


def run_topology_nms(preds: dict[str, np.ndarray], cfg: TNMSConfig) -> dict[str, np.ndarray]:
    """在数据集维度执行 T-NMS。"""
    merged: dict[str, np.ndarray] = {}
    for image_id in sorted(preds.keys()):
        merged[image_id] = topology_nms_single_image(preds[image_id], cfg)
    return merged


def count_total_boxes(data: dict[str, np.ndarray]) -> int:
    """统计总框数。"""
    return int(sum(v.shape[0] for v in data.values()))


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description=(
            "离线拓扑非极大值抑制（T-NMS）脚本。"
            "支持 json/txt/csv/npy 输入，输出同样支持 json/txt/csv/npy。"
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="输入预测文件路径。")
    parser.add_argument("--output", type=Path, required=True, help="输出文件路径。")
    parser.add_argument("--angle-thr-deg", type=float, default=8.0, help="角度阈值（度）。")
    parser.add_argument("--perp-thr", type=float, default=8.0, help="法向共线距离阈值（像素）。")
    parser.add_argument("--gap-thr", type=float, default=20.0, help="轴向间隙阈值（像素）。")
    parser.add_argument("--conf-thr", type=float, default=0.001, help="置信度筛选阈值。")
    parser.add_argument(
        "--pre-nms-iou",
        type=float,
        default=0.0,
        help="若 > 0，则先调用官方 TorchNMS.fast_nms（batch_probiou）进行预去重。",
    )
    parser.add_argument("--class-agnostic", action="store_true", help="启用类别无关融合（默认类别感知）。")
    parser.add_argument("--save-poly", action="store_true", help="仅 JSON 输出时有效，额外保存 poly 顶点。")
    return parser


def main() -> None:
    """命令行入口。"""
    args = build_parser().parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"输入文件不存在: {args.input}")

    cfg = TNMSConfig(
        angle_thr_deg=float(args.angle_thr_deg),
        perp_thr=float(args.perp_thr),
        gap_thr=float(args.gap_thr),
        conf_thr=float(args.conf_thr),
        pre_nms_iou=float(args.pre_nms_iou),
        class_aware=not bool(args.class_agnostic),
    )

    preds, image_to_file = load_predictions(args.input)
    before = count_total_boxes(preds)

    merged = run_topology_nms(preds, cfg)
    after = count_total_boxes(merged)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_predictions(args.output, merged, image_to_file, save_poly=bool(args.save_poly))

    ratio = (after / before) if before > 0 else 0.0
    print(f"处理完成：图像数={len(preds)}，输入框数={before}，输出框数={after}，保留比例={ratio:.4f}")
    print(f"输出文件：{args.output}")


if __name__ == "__main__":
    main()
