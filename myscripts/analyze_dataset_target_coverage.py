import argparse
import csv
import datetime
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
try:
    import yaml
except ModuleNotFoundError:
    yaml = None

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_DATASETS = [
    ("TTPLA-1024", "E:/WorkPlace/my-autodl-mmdet/dataset/TTPLA/sliced_claude/dataset.yaml"),
    ("TTPLA-640", "E:/WorkPlace/my-autodl-mmdet/dataset/TTPLA/sliced_640/dataset.yaml"),
]
DEFAULT_OUTPUT_DIR = "E:/WorkPlace/my-autodl-mmdet/dataset/TTPLA/640-1024-compare_results"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
LONG_EDGE_BINS = [0, 32, 64, 128, 256, 512, float("inf")]
SHORT_EDGE_BINS = [0, 4, 8, 12, 16, 32, 64, float("inf")]
AR_BINS = [0, 5, 10, 30, 60, 100, float("inf")]
LENGTH_RATIO_BINS = [0, 0.125, 0.25, 0.5, 0.75, 1.0, float("inf")]


def parse_dataset_arg(text):
    if "=" not in text:
        path = text
        name = Path(path).parent.name or Path(path).stem
        return name, path
    name, path = text.split("=", 1)
    return name.strip(), path.strip()


def resolve_path(path_text, base_dir):
    path = Path(os.path.expanduser(str(path_text)))
    if path.is_absolute():
        return path
    return (Path(base_dir) / path).resolve()


def parse_scalar_value(value):
    value = value.strip()
    if not value:
        return ""
    if "#" in value:
        value = value.split("#", 1)[0].strip()
    if len(value) >= 2 and value[0] in ("'", '"') and value[-1] == value[0]:
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1].strip()
        if not body:
            return []
        return [parse_scalar_value(x.strip()) for x in body.split(",")]
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def simple_yaml_load(text):
    data = {}
    current_key = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not raw_line.startswith((" ", "\t")):
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value:
                data[key] = parse_scalar_value(value)
                current_key = None
            else:
                data[key] = {}
                current_key = key
            continue
        if current_key is None:
            continue
        child = stripped
        if child.startswith("- "):
            if not isinstance(data[current_key], list):
                data[current_key] = []
            data[current_key].append(parse_scalar_value(child[2:].strip()))
        elif ":" in child:
            if not isinstance(data[current_key], dict):
                data[current_key] = {}
            key, value = child.split(":", 1)
            data[current_key][parse_scalar_value(key.strip())] = parse_scalar_value(value.strip())
    return data


def load_yaml(data_yaml):
    data_yaml = Path(data_yaml)
    with open(data_yaml, "r", encoding="utf-8") as f:
        if yaml is not None:
            data = yaml.safe_load(f)
        else:
            data = simple_yaml_load(f.read())
    if data is None:
        data = {}
    root = data.get("path", data_yaml.parent)
    root = resolve_path(root, data_yaml.parent)
    return data, root


def collect_images_from_entry(entry, data_root):
    paths = []
    if isinstance(entry, (list, tuple)):
        for item in entry:
            paths.extend(collect_images_from_entry(item, data_root))
        return paths

    entry_path = resolve_path(entry, data_root)
    if entry_path.is_dir():
        for ext in IMAGE_EXTS:
            paths.extend(entry_path.rglob(f"*{ext}"))
            paths.extend(entry_path.rglob(f"*{ext.upper()}"))
        return sorted(set(paths))

    if entry_path.is_file() and entry_path.suffix.lower() == ".txt":
        with open(entry_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                image_path = resolve_path(line, data_root)
                paths.append(image_path)
        return sorted(set(paths))

    if entry_path.is_file() and entry_path.suffix.lower() in IMAGE_EXTS:
        return [entry_path]

    matches = []
    for item in glob.glob(str(entry_path)):
        item_path = Path(item)
        if item_path.is_dir():
            matches.extend(collect_images_from_entry(str(item_path), data_root))
        elif item_path.suffix.lower() in IMAGE_EXTS:
            matches.append(item_path)
    return sorted(set(matches))


def get_split_images(data, data_root, splits):
    split_to_images = {}
    for split in splits:
        if split not in data or data[split] is None:
            continue
        split_to_images[split] = collect_images_from_entry(data[split], data_root)
    return split_to_images


def image_to_label_path(image_path):
    image_path = Path(image_path)
    parts = list(image_path.parts)
    lower_parts = [p.lower() for p in parts]
    if "images" in lower_parts:
        idx = len(lower_parts) - 1 - lower_parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    if len(image_path.parents) >= 2:
        return image_path.parent.parent / "labels" / image_path.parent.name / f"{image_path.stem}.txt"
    return image_path.with_suffix(".txt")


def read_image_size(image_path):
    try:
        from PIL import Image
        with Image.open(image_path) as im:
            return im.size
    except Exception:
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        h, w = img.shape[:2]
        return w, h


def parse_obb_label(label_path, img_w, img_h):
    records = []
    if not Path(label_path).exists():
        return records
    with open(label_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            try:
                cls = int(float(parts[0]))
                coords = np.array([float(x) for x in parts[1:9]], dtype=np.float32).reshape(4, 2)
            except ValueError:
                continue
            coords[:, 0] *= img_w
            coords[:, 1] *= img_h
            geom = corners_to_geometry(coords)
            if geom is None:
                continue
            geom.update({"cls": cls, "line_no": line_no})
            records.append(geom)
    return records


def corners_to_geometry(points):
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    if not np.isfinite(points).all():
        return None
    rect = cv2.minAreaRect(points)
    (cx, cy), (w, h), angle_deg = rect
    if w <= 0 or h <= 0:
        edge_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        if edge_lengths.max(initial=0) <= 0:
            return None
        long_edge = float(edge_lengths.max())
        short_edge = float(np.partition(edge_lengths, 1)[1]) if len(edge_lengths) >= 2 else 0.0
        cx, cy = points.mean(axis=0).tolist()
        angle_deg = 0.0
        w, h = long_edge, short_edge
    long_edge = float(max(w, h))
    short_edge = float(min(w, h))
    area = float(max(w, 0.0) * max(h, 0.0))
    ar = float(long_edge / max(short_edge, 1e-6))
    theta = math.radians(float(angle_deg))
    return {
        "cx": float(cx),
        "cy": float(cy),
        "w": float(w),
        "h": float(h),
        "theta": theta,
        "long_edge": long_edge,
        "short_edge": short_edge,
        "area": area,
        "aspect_ratio": ar,
        "points": points,
    }


def calc_nearest_anchor_dreq_px(cx, cy, w, h, theta, stride, img_w, img_h):
    ax = (math.floor(cx / stride) + 0.5) * stride
    ay = (math.floor(cy / stride) + 0.5) * stride
    ax = min(max(ax, 0.5 * stride), max(img_w - 0.5 * stride, 0.5 * stride))
    ay = min(max(ay, 0.5 * stride), max(img_h - 0.5 * stride, 0.5 * stride))
    offset_x = cx - ax
    offset_y = cy - ay
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    xf = offset_x * cos_t + offset_y * sin_t
    yf = -offset_x * sin_t + offset_y * cos_t
    left = w / 2.0 - xf
    top = h / 2.0 - yf
    right = w / 2.0 + xf
    bottom = h / 2.0 + yf
    dists = [left, top, right, bottom]
    inside = min(dists) >= -1e-6
    return max(dists), inside, ax, ay


def target_touch_border(points, img_w, img_h, eps_px):
    x = points[:, 0]
    y = points[:, 1]
    return bool((x <= eps_px).any() or (x >= img_w - eps_px).any() or (y <= eps_px).any() or (y >= img_h - eps_px).any())


def bin_label(value, bins):
    for left, right in zip(bins[:-1], bins[1:]):
        if value < right:
            if math.isinf(right):
                return f">={left:g}"
            return f"{left:g}-{right:g}"
    return f">={bins[-2]:g}"


def all_bin_labels(bins):
    labels = []
    for left, right in zip(bins[:-1], bins[1:]):
        if math.isinf(right):
            labels.append(f">={left:g}")
        else:
            labels.append(f"{left:g}-{right:g}")
    return labels


def safe_percentile(values, q):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_values(values):
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def init_coverage_counter(strides, reg_max_values):
    counter = {}
    for reg_max in reg_max_values:
        counter[str(reg_max)] = {
            "any_center": 0,
            "any_nearest": 0,
            "none_center": 0,
            "none_nearest": 0,
            "min_stride_center": Counter(),
            "min_stride_nearest": Counter(),
            "stride_center": {str(s): 0 for s in strides},
            "stride_nearest": {str(s): 0 for s in strides},
        }
    return counter


def analyze_dataset(name, data_yaml, splits, strides, reg_max_values, border_eps_px, read_each_image_size=False):
    data_yaml = Path(data_yaml)
    data, data_root = load_yaml(data_yaml)
    split_to_images = get_split_images(data, data_root, splits)
    dataset_result = {
        "name": name,
        "yaml": str(data_yaml),
        "root": str(data_root),
        "names": data.get("names", {}),
        "splits": {},
        "objects": [],
        "warnings": [],
    }

    fixed_size = None
    if not read_each_image_size:
        for image_paths in split_to_images.values():
            for image_path in image_paths:
                fixed_size = read_image_size(image_path)
                if fixed_size is not None:
                    break
            if fixed_size is not None:
                break
        if fixed_size is None:
            dataset_result["warnings"].append("未能读取任意图片尺寸，后续将逐张尝试读取。")

    for split, image_paths in split_to_images.items():
        split_stats = {
            "images": len(image_paths),
            "readable_images": 0,
            "positive_images": 0,
            "negative_images": 0,
            "missing_labels": 0,
            "objects": 0,
        }
        for idx, image_path in enumerate(image_paths):
            size = read_image_size(image_path) if read_each_image_size or fixed_size is None else fixed_size
            if size is None:
                dataset_result["warnings"].append(f"无法读取图片尺寸: {image_path}")
                continue
            if idx == 0 or (idx + 1) % 2000 == 0:
                print(f"  [{name}/{split}] 已处理图片 {idx + 1}/{len(image_paths)}", flush=True)
            img_w, img_h = size
            split_stats["readable_images"] += 1
            label_path = image_to_label_path(image_path)
            if not label_path.exists():
                split_stats["missing_labels"] += 1
            labels = parse_obb_label(label_path, img_w, img_h)
            if labels:
                split_stats["positive_images"] += 1
            else:
                split_stats["negative_images"] += 1
            split_stats["objects"] += len(labels)

            for item_idx, geom in enumerate(labels):
                rec = {
                    "dataset": name,
                    "split": split,
                    "image": str(image_path),
                    "label": str(label_path),
                    "object_index": item_idx,
                    "cls": geom["cls"],
                    "img_w": img_w,
                    "img_h": img_h,
                    "cx": geom["cx"],
                    "cy": geom["cy"],
                    "w": geom["w"],
                    "h": geom["h"],
                    "long_edge": geom["long_edge"],
                    "short_edge": geom["short_edge"],
                    "aspect_ratio": geom["aspect_ratio"],
                    "area": geom["area"],
                    "area_ratio": geom["area"] / max(float(img_w * img_h), 1.0),
                    "long_ratio": geom["long_edge"] / max(float(max(img_w, img_h)), 1.0),
                    "touch_border": target_touch_border(geom["points"], img_w, img_h, border_eps_px),
                    "long_bin": bin_label(geom["long_edge"], LONG_EDGE_BINS),
                    "short_bin": bin_label(geom["short_edge"], SHORT_EDGE_BINS),
                    "ar_bin": bin_label(geom["aspect_ratio"], AR_BINS),
                    "long_ratio_bin": bin_label(geom["long_edge"] / max(float(max(img_w, img_h)), 1.0), LENGTH_RATIO_BINS),
                }

                for reg_max in reg_max_values:
                    min_stride_center = None
                    min_stride_nearest = None
                    for stride in strides:
                        dmax_px = float((reg_max - 1) * stride)
                        center_dreq_px = float(max(geom["w"], geom["h"]) / 2.0)
                        nearest_dreq_px, anchor_inside, ax, ay = calc_nearest_anchor_dreq_px(
                            geom["cx"], geom["cy"], geom["w"], geom["h"], geom["theta"], stride, img_w, img_h
                        )
                        center_cover = center_dreq_px <= dmax_px
                        nearest_cover = bool(anchor_inside and nearest_dreq_px <= dmax_px)
                        prefix = f"rm{reg_max}_s{stride}"
                        rec[f"{prefix}_dmax_px"] = dmax_px
                        rec[f"{prefix}_center_dreq_px"] = center_dreq_px
                        rec[f"{prefix}_nearest_dreq_px"] = float(nearest_dreq_px)
                        rec[f"{prefix}_nearest_anchor_inside"] = bool(anchor_inside)
                        rec[f"{prefix}_center_cover"] = bool(center_cover)
                        rec[f"{prefix}_nearest_cover"] = bool(nearest_cover)
                        rec[f"{prefix}_nearest_anchor_x"] = float(ax)
                        rec[f"{prefix}_nearest_anchor_y"] = float(ay)
                        if center_cover and min_stride_center is None:
                            min_stride_center = stride
                        if nearest_cover and min_stride_nearest is None:
                            min_stride_nearest = stride
                    rec[f"rm{reg_max}_min_stride_center"] = min_stride_center if min_stride_center is not None else "none"
                    rec[f"rm{reg_max}_min_stride_nearest"] = min_stride_nearest if min_stride_nearest is not None else "none"

                dataset_result["objects"].append(rec)
        dataset_result["splits"][split] = split_stats
    return dataset_result


def build_summary(dataset_result, strides, reg_max_values):
    objects = dataset_result["objects"]
    summary = {
        "name": dataset_result["name"],
        "yaml": dataset_result["yaml"],
        "splits": dataset_result["splits"],
        "object_count": len(objects),
        "class_counts": Counter(),
        "long_edge": summarize_values([o["long_edge"] for o in objects]),
        "short_edge": summarize_values([o["short_edge"] for o in objects]),
        "aspect_ratio": summarize_values([o["aspect_ratio"] for o in objects]),
        "area_ratio": summarize_values([o["area_ratio"] for o in objects]),
        "long_ratio": summarize_values([o["long_ratio"] for o in objects]),
        "touch_border_count": 0,
        "bins": {
            "long_edge": Counter(),
            "short_edge": Counter(),
            "aspect_ratio": Counter(),
            "long_ratio": Counter(),
        },
        "coverage": init_coverage_counter(strides, reg_max_values),
        "coverage_by_long_bin": {},
    }

    for obj in objects:
        summary["class_counts"][str(obj["cls"])] += 1
        summary["touch_border_count"] += int(bool(obj["touch_border"]))
        summary["bins"]["long_edge"][obj["long_bin"]] += 1
        summary["bins"]["short_edge"][obj["short_bin"]] += 1
        summary["bins"]["aspect_ratio"][obj["ar_bin"]] += 1
        summary["bins"]["long_ratio"][obj["long_ratio_bin"]] += 1

    for reg_max in reg_max_values:
        key = str(reg_max)
        cov = summary["coverage"][key]
        for obj in objects:
            center_any = False
            nearest_any = False
            for stride in strides:
                if obj[f"rm{reg_max}_s{stride}_center_cover"]:
                    center_any = True
                    cov["stride_center"][str(stride)] += 1
                if obj[f"rm{reg_max}_s{stride}_nearest_cover"]:
                    nearest_any = True
                    cov["stride_nearest"][str(stride)] += 1
            cov["any_center"] += int(center_any)
            cov["any_nearest"] += int(nearest_any)
            cov["none_center"] += int(not center_any)
            cov["none_nearest"] += int(not nearest_any)
            cov["min_stride_center"][str(obj[f"rm{reg_max}_min_stride_center"])] += 1
            cov["min_stride_nearest"][str(obj[f"rm{reg_max}_min_stride_nearest"])] += 1

        by_bin = {}
        for label in all_bin_labels(LONG_EDGE_BINS):
            bucket_objs = [o for o in objects if o["long_bin"] == label]
            if not bucket_objs:
                by_bin[label] = {"total": 0, "nearest_any": 0, "nearest_ratio": 0.0}
                continue
            covered = sum(
                any(o[f"rm{reg_max}_s{stride}_nearest_cover"] for stride in strides)
                for o in bucket_objs
            )
            by_bin[label] = {"total": len(bucket_objs), "nearest_any": int(covered), "nearest_ratio": covered / len(bucket_objs)}
        summary["coverage_by_long_bin"][key] = by_bin

    summary["class_counts"] = dict(summary["class_counts"])
    for group in summary["bins"].values():
        group_keys = list(group.keys())
        for k in group_keys:
            group[k] = int(group[k])
    for reg_max in reg_max_values:
        cov = summary["coverage"][str(reg_max)]
        cov["min_stride_center"] = dict(cov["min_stride_center"])
        cov["min_stride_nearest"] = dict(cov["min_stride_nearest"])
    return summary


def fmt_num(value, digits=2):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.{digits}f}"


def fmt_pct(numer, denom):
    if denom <= 0:
        return "0.00%"
    return f"{numer / denom * 100:.2f}%"


def markdown_table(headers, rows):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join([":---" for _ in headers]) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return lines


def write_outputs(results, summaries, output_dir, strides, reg_max_values):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    objects = []
    for result in results:
        objects.extend(result["objects"])

    csv_path = output_dir / "target_objects.csv"
    if objects:
        fieldnames = list(objects[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(objects)

    json_path = output_dir / "target_coverage_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summaries": summaries}, f, indent=2, ensure_ascii=False)

    md = []
    md.append("# 双切片数据集目标尺度与覆盖性分析报告")
    md.append("")
    md.append(f"- **生成时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"- **分析 stride**: {', '.join(str(s) for s in strides)}")
    md.append(f"- **分析 reg_max**: {', '.join(str(r) for r in reg_max_values)}")
    md.append("- **覆盖判据**: `D_max = stride × (reg_max - 1)`；`center` 为目标中心理想锚点，`nearest` 为离目标中心最近的实际网格锚点。")
    md.append("- **注意**: 本报告的覆盖性主要指 DFL/旋转框回归距离是否可覆盖，不等同于严格的 CNN 有效感受野。")
    md.append("")

    rows = []
    for s in summaries:
        total_images = sum(v["images"] for v in s["splits"].values())
        pos_images = sum(v["positive_images"] for v in s["splits"].values())
        neg_images = sum(v["negative_images"] for v in s["splits"].values())
        n_obj = s["object_count"]
        rows.append([
            s["name"],
            total_images,
            pos_images,
            neg_images,
            n_obj,
            fmt_pct(s["touch_border_count"], max(n_obj, 1)),
            fmt_num(s["long_edge"]["mean"]),
            fmt_num(s["long_edge"]["p95"]),
            fmt_num(s["long_edge"]["max"]),
            fmt_num(s["short_edge"]["p50"]),
            fmt_num(s["aspect_ratio"]["p95"]),
        ])
    md.append("## 1. 总览")
    md.extend(markdown_table(
        ["数据集", "图片数", "正样本图", "负样本图", "目标数", "触边目标比例", "长边均值", "长边P95", "长边最大", "短边P50", "AR P95"],
        rows,
    ))
    md.append("")

    for s in summaries:
        md.append(f"## 2. {s['name']} 尺度分布")
        stats_rows = []
        for key, label in [("long_edge", "长边(px)"), ("short_edge", "短边(px)"), ("aspect_ratio", "长宽比"), ("long_ratio", "长边/切片尺寸")]:
            st = s[key]
            stats_rows.append([label, st["count"], fmt_num(st["mean"]), fmt_num(st["p50"]), fmt_num(st["p75"]), fmt_num(st["p90"]), fmt_num(st["p95"]), fmt_num(st["p99"]), fmt_num(st["max"])] )
        md.extend(markdown_table(["指标", "数量", "均值", "P50", "P75", "P90", "P95", "P99", "最大"], stats_rows))
        md.append("")

        for bin_key, title in [("long_edge", "长边分桶"), ("short_edge", "短边分桶"), ("aspect_ratio", "长宽比分桶"), ("long_ratio", "长边/切片尺寸分桶")]:
            bin_rows = []
            total = max(s["object_count"], 1)
            for label, count in sorted(s["bins"][bin_key].items(), key=lambda kv: kv[0]):
                bin_rows.append([label, count, fmt_pct(count, total)])
            md.append(f"### {title}")
            md.extend(markdown_table(["分桶", "目标数", "占比"], bin_rows))
            md.append("")

    md.append("## 3. DFL 回归覆盖性")
    for s in summaries:
        n_obj = max(s["object_count"], 1)
        md.append(f"### {s['name']}")
        for reg_max in reg_max_values:
            cov = s["coverage"][str(reg_max)]
            rows = []
            rows.append(["任意层可覆盖(center)", cov["any_center"], fmt_pct(cov["any_center"], n_obj)])
            rows.append(["任意层可覆盖(nearest)", cov["any_nearest"], fmt_pct(cov["any_nearest"], n_obj)])
            rows.append(["任意层不可覆盖(center)", cov["none_center"], fmt_pct(cov["none_center"], n_obj)])
            rows.append(["任意层不可覆盖(nearest)", cov["none_nearest"], fmt_pct(cov["none_nearest"], n_obj)])
            md.append(f"#### reg_max={reg_max}")
            md.extend(markdown_table(["项目", "目标数", "比例"], rows))
            md.append("")

            stride_rows = []
            for stride in strides:
                dmax = stride * (reg_max - 1)
                centered_long = 2 * dmax
                stride_rows.append([
                    f"stride={stride}",
                    dmax,
                    centered_long,
                    cov["stride_center"][str(stride)],
                    fmt_pct(cov["stride_center"][str(stride)], n_obj),
                    cov["stride_nearest"][str(stride)],
                    fmt_pct(cov["stride_nearest"][str(stride)], n_obj),
                ])
            md.extend(markdown_table(["层级", "单边D_max(px)", "居中最大长边(px)", "center覆盖数", "center覆盖率", "nearest覆盖数", "nearest覆盖率"], stride_rows))
            md.append("")

            min_rows = []
            keys = sorted(set(cov["min_stride_nearest"].keys()) | set(cov["min_stride_center"].keys()), key=lambda x: 9999 if x == "none" else int(x))
            for key in keys:
                min_rows.append([
                    key,
                    cov["min_stride_center"].get(key, 0),
                    fmt_pct(cov["min_stride_center"].get(key, 0), n_obj),
                    cov["min_stride_nearest"].get(key, 0),
                    fmt_pct(cov["min_stride_nearest"].get(key, 0), n_obj),
                ])
            md.append("最小可覆盖层统计：")
            md.extend(markdown_table(["最小stride", "center数量", "center比例", "nearest数量", "nearest比例"], min_rows))
            md.append("")

            bin_rows = []
            for label, item in s["coverage_by_long_bin"][str(reg_max)].items():
                bin_rows.append([label, item["total"], item["nearest_any"], f"{item['nearest_ratio'] * 100:.2f}%"])
            md.append("按长边分桶的 nearest 覆盖率：")
            md.extend(markdown_table(["长边分桶", "目标数", "可覆盖数", "可覆盖率"], bin_rows))
            md.append("")

    if len(summaries) >= 2:
        md.append("## 4. 两套数据集直接对比")
        base = summaries[0]
        for other in summaries[1:]:
            md.append(f"### {base['name']} vs {other['name']}")
            compare_rows = []
            for key, label in [("object_count", "目标数"), ("long_edge", "长边P95"), ("long_edge", "长边最大"), ("short_edge", "短边P50"), ("aspect_ratio", "AR P95")]:
                if key == "object_count":
                    a = base[key]
                    b = other[key]
                elif label == "长边最大":
                    a = base[key]["max"]
                    b = other[key]["max"]
                elif label == "长边P95":
                    a = base[key]["p95"]
                    b = other[key]["p95"]
                elif label == "短边P50":
                    a = base[key]["p50"]
                    b = other[key]["p50"]
                else:
                    a = base[key]["p95"]
                    b = other[key]["p95"]
                delta = float(b) - float(a)
                compare_rows.append([label, fmt_num(a), fmt_num(b), f"{delta:+.2f}"])
            md.extend(markdown_table(["指标", base["name"], other["name"], "差值(后者-前者)"], compare_rows))
            md.append("")

    md.append("## 5. 解读建议")
    md.append("- **如果 640 数据集长边 P95/最大值明显低于 1024**，说明切片变小后长目标被切得更短，原始 `reg_max=16` 更容易覆盖。")
    md.append("- **如果 `reg_max=16` 在 stride=8 或 nearest 任意层的不可覆盖比例仍高**，则原始 YOLO 的 DFL 回归范围仍可能是瓶颈。")
    md.append("- **如果触边目标比例很高**，说明大量目标被切片截断，指标变化可能同时来自目标长度变短与标注形态改变。")
    md.append("- **如果 640 的短边更小但长边也显著缩短**，AP75 变化需要结合宽度误差与长度覆盖两个因素解释。")
    md.append("")

    md_path = output_dir / "target_coverage_report.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    return csv_path, json_path, md_path


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", default=None, help="格式：名称=/path/to/dataset.yaml；可重复传入")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--strides", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--reg-max", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--border-eps-px", type=float, default=1.0)
    parser.add_argument("--read-each-image-size", action="store_true", help="逐张读取图片尺寸；固定切片数据集默认不需要开启")
    args = parser.parse_args()

    dataset_items = [parse_dataset_arg(x) for x in args.dataset] if args.dataset else DEFAULT_DATASETS

    print("=" * 80)
    print("双切片数据集目标尺度与覆盖性分析")
    print("=" * 80)
    print(f"输出目录: {args.output_dir}")
    print(f"splits: {args.splits}")
    print(f"strides: {args.strides}")
    print(f"reg_max: {args.reg_max}")

    results = []
    summaries = []
    for name, data_yaml in dataset_items:
        print("\n" + "-" * 80)
        print(f"分析数据集: {name}")
        print(f"dataset.yaml: {data_yaml}")
        if not Path(data_yaml).exists():
            print(f"[警告] dataset.yaml 不存在，跳过: {data_yaml}")
            continue
        result = analyze_dataset(
            name,
            data_yaml,
            args.splits,
            args.strides,
            args.reg_max,
            args.border_eps_px,
            read_each_image_size=args.read_each_image_size,
        )
        summary = build_summary(result, args.strides, args.reg_max)
        results.append(result)
        summaries.append(summary)
        print(f"图片统计: {summary['splits']}")
        print(f"目标数: {summary['object_count']}")
        print(f"长边P95/最大: {summary['long_edge']['p95']:.2f} / {summary['long_edge']['max']:.2f}")
        print(f"短边P50: {summary['short_edge']['p50']:.2f}")
        print(f"触边目标比例: {fmt_pct(summary['touch_border_count'], max(summary['object_count'], 1))}")
        for reg_max in args.reg_max:
            cov = summary["coverage"][str(reg_max)]
            print(
                f"reg_max={reg_max}: nearest 任意层覆盖率 "
                f"{fmt_pct(cov['any_nearest'], max(summary['object_count'], 1))}, "
                f"不可覆盖 {cov['none_nearest']} 个"
            )

    if not summaries:
        raise FileNotFoundError("没有成功分析任何数据集，请检查 --dataset 路径。")

    csv_path, json_path, md_path = write_outputs(results, summaries, args.output_dir, args.strides, args.reg_max)
    print("\n" + "=" * 80)
    print("分析完成")
    print(f"目标明细 CSV: {csv_path}")
    print(f"汇总 JSON: {json_path}")
    print(f"Markdown 报告: {md_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
