from __future__ import annotations

import argparse
import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

try:
    from shapely.geometry import Polygon, box
except ImportError as exc:  # pragma: no cover
    raise ImportError("shapely is required. Install it with: pip install shapely") from exc


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass
class SplitStats:
    images_total: int = 0
    images_missing: int = 0
    tiles_total: int = 0
    tiles_positive: int = 0
    tiles_negative: int = 0
    obb_total: int = 0


def parse_csv_set(raw: str) -> Set[str]:
    items = [item.strip().lower() for item in raw.split(",") if item.strip()]
    return set(items)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    ttpla_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description=(
            "Slice TTPLA original images into 1024x1024 tiles, keep wire-related "
            "annotations only, and export YOLOv11-OBB labels."
        )
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=ttpla_root / "data_original_size_v1" / "splitting_jsons",
        help="Root folder containing train_jsons/val_jsons/test_jsons",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=ttpla_root / "data_original_size_v1" / "data_original_size",
        help="Folder containing original images",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=script_dir,
        help="Output root folder (will create train/val/test subfolders)",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "val", "test"],
        help="Splits to process, e.g. --splits train val test",
    )
    parser.add_argument("--tile-size", type=int, default=1024, help="Tile size")
    parser.add_argument(
        "--stride",
        type=int,
        default=824,
        help="Sliding stride in pixels (200 means heavy overlap)",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=16.0,
        help="Minimum clipped polygon area to keep",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for output images",
    )
    parser.add_argument(
        "--pad-value",
        type=int,
        default=0,
        help="Padding value for right/bottom padding",
    )
    parser.add_argument(
        "--keep-exact-labels",
        type=str,
        default="cable",
        help="Comma-separated exact labels to keep",
    )
    parser.add_argument(
        "--keep-label-keywords",
        type=str,
        default="cable,wire,conductor",
        help="Comma-separated keywords for wire-related labels",
    )
    parser.add_argument(
        "--exclude-label-keywords",
        type=str,
        default="tower,insulator,void,pole",
        help="Comma-separated keywords to exclude",
    )
    parser.add_argument(
        "--class-name",
        type=str,
        default="cable",
        help="Single YOLO class name",
    )
    parser.add_argument(
        "--no-save-tile-json",
        action="store_true",
        help="Disable saving per-tile JSON annotations (enabled by default)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max JSON files per split (debug)",
    )
    return parser.parse_args()


def read_image(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    raw = np.fromfile(str(path), dtype=np.uint8)
    if raw.size == 0:
        return None
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def decode_image_data(image_data: str) -> Optional[np.ndarray]:
    try:
        raw = base64.b64decode(image_data)
    except Exception:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def write_jpg(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    buf.tofile(str(path))


def compute_padded_length(length: int, tile_size: int, stride: int) -> int:
    if length <= tile_size:
        return tile_size
    remainder = (length - tile_size) % stride
    if remainder == 0:
        return length
    return length + (stride - remainder)


def build_positions(length: int, tile_size: int, stride: int) -> List[int]:
    max_start = max(0, length - tile_size)
    return list(range(0, max_start + 1, stride))


def pad_image(image: np.ndarray, padded_h: int, padded_w: int, pad_value: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == padded_h and w == padded_w:
        return image
    return cv2.copyMakeBorder(
        image,
        0,
        max(0, padded_h - h),
        0,
        max(0, padded_w - w),
        cv2.BORDER_CONSTANT,
        value=(pad_value, pad_value, pad_value),
    )


def safe_polygon(points: Sequence[Sequence[float]]) -> Optional[Polygon]:
    if len(points) < 3:
        return None
    try:
        poly = Polygon(points)
    except Exception:
        return None
    if not poly.is_valid:
        try:
            poly = poly.buffer(0)
        except Exception:
            return None
    if poly.is_empty:
        return None
    return poly


def extract_polygons(geom) -> List[Polygon]:
    if geom.is_empty:
        return []
    geom_type = geom.geom_type
    if geom_type == "Polygon":
        return [geom]
    if geom_type == "MultiPolygon":
        return list(geom.geoms)
    if geom_type == "GeometryCollection":
        result: List[Polygon] = []
        for item in geom.geoms:
            result.extend(extract_polygons(item))
        return result
    return []


def polygon_to_local_points(poly: Polygon, offset_x: int, offset_y: int, tile_size: int) -> List[List[float]]:
    coords = list(poly.exterior.coords)
    if len(coords) > 1:
        coords = coords[:-1]
    local: List[List[float]] = []
    for x, y in coords:
        lx = min(float(tile_size), max(0.0, float(x - offset_x)))
        ly = min(float(tile_size), max(0.0, float(y - offset_y)))
        local.append([round(lx, 2), round(ly, 2)])
    return local


def order_quad_clockwise(points: np.ndarray) -> np.ndarray:
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    order = np.argsort(angles)[::-1]
    ordered = points[order]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    return np.roll(ordered, -start, axis=0)


def _shrink_rect_into_tile(rect, ts: float, orig_pts: np.ndarray = None) -> np.ndarray:
    """保持 minAreaRect 的角度，通过沿电线方向平移和/或缩短长边使四角落入 [0, ts]。

    策略（按优先级）：
      1. 沿电线方向（长边方向）做最小量平移
         —— 用解析法求有效平移区间，取 |s| 最小解
         —— 平移后验证：矩形框对原始标注沿电线方向的覆盖率 ≥ 85%
      2. 仅缩短长边 —— 保持短边（cable 宽度）不变
      3. 保底等比缩放 —— 极端情况兜底
    """
    (cx, cy), (w, h), angle = rect
    cx, cy, w, h = float(cx), float(cy), float(w), float(h)
    EPS = 0.01

    def _in_tile(pts: np.ndarray) -> bool:
        return bool((pts >= -EPS).all() and (pts <= ts + EPS).all())

    # ---------- 已在 tile 内，直接返回 ----------
    box_pts = cv2.boxPoints(((cx, cy), (w, h), angle))
    if _in_tile(box_pts):
        return box_pts

    # ---------- 从 box_pts 的边计算电线方向（长边方向）----------
    edge1 = box_pts[1] - box_pts[0]
    edge2 = box_pts[2] - box_pts[1]
    len1 = float(np.linalg.norm(edge1))
    len2 = float(np.linalg.norm(edge2))
    if len1 >= len2:
        cable_dir = edge1 / max(len1, 1e-9)
        long_len = len1
    else:
        cable_dir = edge2 / max(len2, 1e-9)
        long_len = len2
    cdx, cdy = float(cable_dir[0]), float(cable_dir[1])

    # ---------- 第一步：沿电线方向做最小量平移 ----------
    # 对每个角点 (px, py)，平移量 s 需满足：
    #   -EPS <= px + s*cdx <= ts + EPS
    #   -EPS <= py + s*cdy <= ts + EPS
    # 对 4 个角点取交集，得到有效平移区间 [s_lo, s_hi]
    s_lo, s_hi = -1e18, 1e18
    shift_feasible = True
    for px, py in box_pts:
        px, py = float(px), float(py)
        for val, d in [(px, cdx), (py, cdy)]:
            if abs(d) > 1e-9:
                bound_a = (-EPS - val) / d
                bound_b = (ts + EPS - val) / d
                if d > 0:
                    s_lo = max(s_lo, bound_a)
                    s_hi = min(s_hi, bound_b)
                else:
                    s_lo = max(s_lo, bound_b)
                    s_hi = min(s_hi, bound_a)
            else:
                if val < -EPS or val > ts + EPS:
                    shift_feasible = False
                    break
        if not shift_feasible:
            break

    if shift_feasible and s_lo <= s_hi:
        # 取 |s| 最小的平移量
        if s_lo <= 0 <= s_hi:
            best_s = 0.0
        elif s_lo > 0:
            best_s = s_lo
        else:
            best_s = s_hi  # s_hi < 0，绝对值最小

        new_cx = cx + cdx * best_s
        new_cy = cy + cdy * best_s
        shifted_pts = cv2.boxPoints(((new_cx, new_cy), (w, h), angle))

        if _in_tile(shifted_pts):
            # 覆盖率验证：沿电线方向投影，检查平移后矩形是否仍覆盖原始标注的大部分
            coverage_ok = True
            if orig_pts is not None and len(orig_pts) >= 2:
                proj = orig_pts @ cable_dir  # 原始标注点在电线方向的投影
                p_min, p_max = float(proj.min()), float(proj.max())
                p_span = p_max - p_min
                if p_span > 1e-3:
                    # 平移后矩形中心在电线方向的投影
                    rc = new_cx * cdx + new_cy * cdy
                    half_len = long_len / 2.0
                    r_min, r_max = rc - half_len, rc + half_len
                    overlap = max(0.0, min(p_max, r_max) - max(p_min, r_min))
                    coverage = overlap / p_span
                    coverage_ok = coverage >= 0.85

            if coverage_ok:
                return shifted_pts

    # ---------- 第二步：两端独立缩短，解析求解最大长度 ----------
    # 矩形四角 = 中心 + t·cable_dir ± half_short·perp_dir
    # 其中 t = u (正端) 或 t = v (负端), cable_dir = (cdx, cdy), perp = (-cdy, cdx)
    # 对于4个角点，tile 约束对 u 和 v 给出相同的有效区间 [R_lo, R_hi]
    # 最大长度 = R_hi - R_lo, 新中心沿 cable 方向位于 (R_hi + R_lo) / 2
    half_short = min(len1, len2) / 2.0
    # 4 组约束的基准值
    A1 = cx - half_short * cdy   # 角点 perp 正侧的 x 基准
    A2 = cx + half_short * cdy   # 角点 perp 负侧的 x 基准
    B1 = cy + half_short * cdx   # 角点 perp 正侧的 y 基准
    B2 = cy - half_short * cdx   # 角点 perp 负侧的 y 基准

    R_lo, R_hi = -1e18, 1e18
    step2_feasible = True
    for base, d in [(A1, cdx), (A2, cdx), (B1, cdy), (B2, cdy)]:
        if abs(d) > 1e-9:
            ba = (-EPS - base) / d
            bb = (ts + EPS - base) / d
            if d > 0:
                R_lo = max(R_lo, ba)
                R_hi = min(R_hi, bb)
            else:
                R_lo = max(R_lo, bb)
                R_hi = min(R_hi, ba)
        else:
            if base < -EPS or base > ts + EPS:
                step2_feasible = False
                break

    if step2_feasible and R_hi > R_lo:
        # 二次约束：截断到原始多边形在 cable 方向的投影范围，防止矩形超出多边形边界
        if orig_pts is not None and len(orig_pts) >= 2:
            center_proj = cx * cdx + cy * cdy
            proj = orig_pts @ cable_dir
            t_lo = float(proj.min()) - center_proj
            t_hi = float(proj.max()) - center_proj
            R_hi = min(R_hi, t_hi)
            R_lo = max(R_lo, t_lo)

        if R_hi > R_lo:
            new_long = R_hi - R_lo
            new_half_long = new_long / 2.0
            s_center = (R_hi + R_lo) / 2.0
            new_cx = cx + s_center * cdx
            new_cy = cy + s_center * cdy
            cable_v = np.array([cdx, cdy])
            perp_v = np.array([-cdy, cdx])
            c = np.array([new_cx, new_cy])
            box_pts = np.array([
                c + new_half_long * cable_v + half_short * perp_v,
                c + new_half_long * cable_v - half_short * perp_v,
                c - new_half_long * cable_v - half_short * perp_v,
                c - new_half_long * cable_v + half_short * perp_v,
            ], dtype=np.float32)
            if _in_tile(box_pts):
                return box_pts

    # ---------- 第三步：保底等比缩放（正常情况不会走到这里） ----------
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) * 0.5
        pts = cv2.boxPoints(((cx, cy), (w * mid, h * mid), angle))
        if _in_tile(pts):
            lo = mid
        else:
            hi = mid
    return cv2.boxPoints(((cx, cy), (w * lo, h * lo), angle))


def _tighten_to_polygon(box_pts: np.ndarray, orig_pts: np.ndarray) -> np.ndarray:
    """二次紧缩：确保矩形框沿电线方向不超过原始多边形边界。

    保持矩形的角度和宽度不变，只在 cable 方向上将两端截断到
    原始多边形投影范围 [p_min, p_max]。
    """
    if orig_pts is None or len(orig_pts) < 2:
        return box_pts

    # 从矩形的边确定电线方向
    edge1 = box_pts[1] - box_pts[0]
    edge2 = box_pts[2] - box_pts[1]
    len1 = float(np.linalg.norm(edge1))
    len2 = float(np.linalg.norm(edge2))
    if len1 >= len2:
        cable_dir = edge1 / max(len1, 1e-9)
    else:
        cable_dir = edge2 / max(len2, 1e-9)
    cdx, cdy = float(cable_dir[0]), float(cable_dir[1])
    perp_v = np.array([-cdy, cdx])

    # 原始多边形在 cable 方向的投影范围
    proj_poly = orig_pts @ cable_dir
    p_min, p_max = float(proj_poly.min()), float(proj_poly.max())

    # 当前矩形在 cable 方向的投影范围
    proj_rect = box_pts @ cable_dir
    r_min, r_max = float(proj_rect.min()), float(proj_rect.max())

    # 矩形已经在多边形范围内，无需收紧
    if r_min >= p_min - 0.5 and r_max <= p_max + 0.5:
        return box_pts

    # 截断到多边形范围
    new_min = max(r_min, p_min)
    new_max = min(r_max, p_max)
    if new_max - new_min < 1e-3:
        return box_pts  # 避免退化

    # 计算垂直方向的半宽和中心
    proj_perp = box_pts @ perp_v
    perp_center = (float(proj_perp.max()) + float(proj_perp.min())) / 2.0
    half_short = (float(proj_perp.max()) - float(proj_perp.min())) / 2.0

    # 重新构造矩形
    new_half_long = (new_max - new_min) / 2.0
    cable_center = (new_max + new_min) / 2.0
    cable_v = np.array([cdx, cdy])
    c = cable_center * cable_v + perp_center * perp_v
    return np.array([
        c + new_half_long * cable_v + half_short * perp_v,
        c + new_half_long * cable_v - half_short * perp_v,
        c - new_half_long * cable_v - half_short * perp_v,
        c - new_half_long * cable_v + half_short * perp_v,
    ], dtype=np.float32)


def polygon_to_yolo_obb(local_points: Sequence[Sequence[float]], tile_size: int) -> Optional[List[float]]:
    pts = np.asarray(local_points, dtype=np.float32)
    if pts.shape[0] < 3:
        return None
    rect = cv2.minAreaRect(pts)
    w, h = rect[1]
    if w <= 1e-3 or h <= 1e-3:
        return None

    ts = float(tile_size)
    box_pts = _shrink_rect_into_tile(rect, ts, orig_pts=pts)

    # 二次紧缩：确保矩形不超过原始多边形在 cable 方向的边界
    box_pts = _tighten_to_polygon(box_pts, pts)

    box_pts = order_quad_clockwise(box_pts)
    norm = box_pts / ts
    norm = np.clip(norm, 0.0, 1.0)
    return norm.reshape(-1).tolist()


def should_keep_label(
    label: str,
    keep_exact: Set[str],
    keep_keywords: Set[str],
    exclude_keywords: Set[str],
) -> bool:
    name = label.strip().lower()
    if not name:
        return False
    if name in keep_exact:
        return True
    if any(ex_key in name for ex_key in exclude_keywords):
        return False
    return any(key in name for key in keep_keywords)


def build_image_index(image_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for ext in IMAGE_EXTS:
        for image_path in image_root.glob(f"*{ext}"):
            index[image_path.stem.lower()] = image_path
        for image_path in image_root.glob(f"*{ext.upper()}"):
            index[image_path.stem.lower()] = image_path
    return index


def resolve_image_path(
    sample_json: Dict[str, object],
    json_path: Path,
    image_root: Path,
    image_index: Dict[str, Path],
) -> Optional[Path]:
    by_stem = image_index.get(json_path.stem.lower())
    if by_stem is not None:
        return by_stem

    image_name = Path(str(sample_json.get("imagePath", ""))).name
    if image_name:
        direct = image_root / image_name
        if direct.exists():
            return direct
        by_image_stem = image_index.get(Path(image_name).stem.lower())
        if by_image_stem is not None:
            return by_image_stem
    return None


def collect_wire_polygons(
    data: Dict[str, object],
    keep_exact: Set[str],
    keep_keywords: Set[str],
    exclude_keywords: Set[str],
) -> List[Tuple[str, Polygon, Tuple[float, float, float, float]]]:
    polygons: List[Tuple[str, Polygon, Tuple[float, float, float, float]]] = []
    for shape in data.get("shapes", []):
        if not isinstance(shape, dict):
            continue
        shape_type = str(shape.get("shape_type", "polygon")).lower()
        if shape_type != "polygon":
            continue
        label = str(shape.get("label", "")).strip()
        if not should_keep_label(label, keep_exact, keep_keywords, exclude_keywords):
            continue
        points = shape.get("points", [])
        if not isinstance(points, list):
            continue
        poly = safe_polygon(points)
        if poly is None:
            continue
        polygons.append((label, poly, poly.bounds))
    return polygons


def save_tile_json(
    json_path: Path,
    base_meta: Dict[str, object],
    image_name: str,
    tile_size: int,
    tile_shapes: List[Dict[str, object]],
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": base_meta.get("version", "3.16.7"),
        "flags": base_meta.get("flags", {}),
        "shapes": tile_shapes,
        "lineColor": base_meta.get("lineColor"),
        "fillColor": base_meta.get("fillColor"),
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": tile_size,
        "imageWidth": tile_size,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def process_single_json(
    json_path: Path,
    split: str,
    output_root: Path,
    image_root: Path,
    image_index: Dict[str, Path],
    tile_size: int,
    stride: int,
    min_area: float,
    pad_value: int,
    jpeg_quality: int,
    class_name: str,
    keep_exact: Set[str],
    keep_keywords: Set[str],
    exclude_keywords: Set[str],
    save_tile_jsons: bool,
    stats: SplitStats,
) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))

    image_path = resolve_image_path(data, json_path, image_root, image_index)
    if image_path is None:
        print(f"[WARN] Missing image for: {json_path.name}")
        stats.images_missing += 1
        return

    image = read_image(image_path)
    if image is None and isinstance(data.get("imageData"), str) and data.get("imageData"):
        image = decode_image_data(str(data.get("imageData")))
    if image is None:
        print(f"[WARN] Failed to read image: {image_path}")
        stats.images_missing += 1
        return

    stats.images_total += 1

    wire_polygons = collect_wire_polygons(data, keep_exact, keep_keywords, exclude_keywords)
    h, w = image.shape[:2]
    padded_h = compute_padded_length(h, tile_size, stride)
    padded_w = compute_padded_length(w, tile_size, stride)
    padded_image = pad_image(image, padded_h, padded_w, pad_value)

    xs = build_positions(padded_w, tile_size, stride)
    ys = build_positions(padded_h, tile_size, stride)

    images_dir = output_root / "images" / split
    labels_dir = output_root / "labels" / split
    jsons_dir = output_root / "jsons" / split

    base_meta = {
        "version": data.get("version", "3.16.7"),
        "flags": data.get("flags", {}),
        "lineColor": data.get("lineColor"),
        "fillColor": data.get("fillColor"),
    }

    for y in ys:
        for x in xs:
            tile_box = box(x, y, x + tile_size, y + tile_size)
            tile_name = f"{image_path.stem}_x{x}_y{y}"
            image_name = f"{tile_name}.jpg"
            label_name = f"{tile_name}.txt"

            yolo_lines: List[str] = []
            tile_shapes: List[Dict[str, object]] = []

            for original_label, poly, bounds in wire_polygons:
                minx, miny, maxx, maxy = bounds
                if maxx <= x or minx >= x + tile_size:
                    continue
                if maxy <= y or miny >= y + tile_size:
                    continue

                try:
                    clipped = poly.intersection(tile_box)
                except Exception:
                    try:
                        clipped = poly.buffer(0).intersection(tile_box)
                    except Exception:
                        continue

                for piece in extract_polygons(clipped):
                    if piece.area < min_area:
                        continue
                    local_points = polygon_to_local_points(piece, x, y, tile_size)
                    if len(local_points) < 3:
                        continue
                    obb = polygon_to_yolo_obb(local_points, tile_size)
                    if obb is None:
                        continue

                    yolo_lines.append(
                        "0 " + " ".join(f"{value:.6f}" for value in obb)
                    )
                    tile_shapes.append(
                        {
                            "label": original_label,
                            "points": local_points,
                            "shape_type": "polygon",
                            "flags": {},
                        }
                    )

            tile_img = padded_image[y : y + tile_size, x : x + tile_size]
            write_jpg(images_dir / image_name, tile_img, jpeg_quality)
            (labels_dir / label_name).parent.mkdir(parents=True, exist_ok=True)
            (labels_dir / label_name).write_text("\n".join(yolo_lines), encoding="utf-8")

            if save_tile_jsons:
                save_tile_json(
                    json_path=jsons_dir / f"{tile_name}.json",
                    base_meta=base_meta,
                    image_name=image_name,
                    tile_size=tile_size,
                    tile_shapes=tile_shapes,
                )

            stats.tiles_total += 1
            stats.obb_total += len(yolo_lines)
            if yolo_lines:
                stats.tiles_positive += 1
            else:
                stats.tiles_negative += 1


def process_split(
    split: str,
    split_root: Path,
    image_root: Path,
    output_root: Path,
    image_index: Dict[str, Path],
    tile_size: int,
    stride: int,
    min_area: float,
    pad_value: int,
    jpeg_quality: int,
    class_name: str,
    keep_exact: Set[str],
    keep_keywords: Set[str],
    exclude_keywords: Set[str],
    save_tile_jsons: bool,
    limit: int,
) -> SplitStats:
    input_dir = split_root / f"{split}_jsons"
    if not input_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {input_dir}")

    # 创建 images/split、labels/split、jsons/split 目录
    (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
    (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    if save_tile_jsons:
        (output_root / "jsons" / split).mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    if limit > 0:
        json_files = json_files[:limit]

    stats = SplitStats()
    for idx, json_path in enumerate(json_files, start=1):
        if idx % 20 == 0:
            print(f"[{split}] {idx}/{len(json_files)} JSONs processed...")
        process_single_json(
            json_path=json_path,
            split=split,
            output_root=output_root,
            image_root=image_root,
            image_index=image_index,
            tile_size=tile_size,
            stride=stride,
            min_area=min_area,
            pad_value=pad_value,
            jpeg_quality=jpeg_quality,
            class_name=class_name,
            keep_exact=keep_exact,
            keep_keywords=keep_keywords,
            exclude_keywords=exclude_keywords,
            save_tile_jsons=save_tile_jsons,
            stats=stats,
        )

    print(
        f"[{split}] images_ok={stats.images_total}, images_missing={stats.images_missing}, "
        f"tiles={stats.tiles_total}, pos_tiles={stats.tiles_positive}, "
        f"neg_tiles={stats.tiles_negative}, obb={stats.obb_total}"
    )
    return stats


def write_dataset_yaml(output_root: Path, class_name: str) -> Path:
    yaml_path = output_root / "dataset.yaml"
    yaml_text = (
        f'path: "{output_root.as_posix()}"\n'
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        f"  0: {class_name}\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")
    return yaml_path


def main() -> None:
    args = parse_args()

    if args.tile_size <= 0:
        raise ValueError("--tile-size must be > 0")
    if args.stride <= 0:
        raise ValueError("--stride must be > 0")

    split_root = args.split_root.resolve()
    image_root = args.image_root.resolve()
    output_root = args.output_root.resolve()

    keep_exact = parse_csv_set(args.keep_exact_labels)
    keep_keywords = parse_csv_set(args.keep_label_keywords)
    exclude_keywords = parse_csv_set(args.exclude_label_keywords)

    if not split_root.exists():
        raise FileNotFoundError(f"split root not found: {split_root}")
    if not image_root.exists():
        raise FileNotFoundError(f"image root not found: {image_root}")

    print("Building image index...")
    image_index = build_image_index(image_root)
    if not image_index:
        raise RuntimeError(f"No images found under: {image_root}")

    all_stats: Dict[str, SplitStats] = {}
    for split in args.splits:
        split_name = split.strip()
        if not split_name:
            continue
        print(f"\nProcessing split: {split_name}")
        stats = process_split(
            split=split_name,
            split_root=split_root,
            image_root=image_root,
            output_root=output_root,
            image_index=image_index,
            tile_size=args.tile_size,
            stride=args.stride,
            min_area=args.min_area,
            pad_value=args.pad_value,
            jpeg_quality=args.jpeg_quality,
            class_name=args.class_name,
            keep_exact=keep_exact,
            keep_keywords=keep_keywords,
            exclude_keywords=exclude_keywords,
            save_tile_jsons=not args.no_save_tile_json,
            limit=args.limit,
        )
        all_stats[split_name] = stats

    yaml_path = write_dataset_yaml(output_root, args.class_name)

    print("\nDone.")
    print(f"YOLO dataset yaml: {yaml_path}")
    print("Summary:")
    for split_name, stats in all_stats.items():
        print(
            f"  - {split_name}: images_ok={stats.images_total}, images_missing={stats.images_missing}, "
            f"tiles={stats.tiles_total}, pos_tiles={stats.tiles_positive}, "
            f"neg_tiles={stats.tiles_negative}, obb={stats.obb_total}"
        )


if __name__ == "__main__":
    main()
