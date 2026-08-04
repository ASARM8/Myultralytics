"""Audit train/val image overlap for the fixed Refine V3 dataset without opening test.

The audit checks path/stem collisions, exact file SHA256 duplicates, and visual
near-duplicates using a compact difference hash. Near-duplicate pairs are
diagnostic because visually repetitive power-line scenes can legitimately have
small Hamming distances; exact binary cross-split duplicates are a hard error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


class HammingBKTree:
    """Small BK-tree for integer perceptual hashes and Hamming distance."""

    def __init__(self) -> None:
        self.root: dict[str, Any] | None = None

    @staticmethod
    def distance(first: int, second: int) -> int:
        return (int(first) ^ int(second)).bit_count()

    def add(self, value: int, payload: str) -> None:
        if self.root is None:
            self.root = {"value": int(value), "payloads": [payload], "children": {}}
            return
        node = self.root
        while True:
            distance = self.distance(value, node["value"])
            if distance == 0:
                node["payloads"].append(payload)
                return
            child = node["children"].get(distance)
            if child is None:
                node["children"][distance] = {"value": int(value), "payloads": [payload], "children": {}}
                return
            node = child

    def query(self, value: int, maximum_distance: int) -> list[tuple[int, str]]:
        if self.root is None:
            return []
        result: list[tuple[int, str]] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = self.distance(value, node["value"])
            if distance <= maximum_distance:
                result.extend((distance, payload) for payload in node["payloads"])
            lower, upper = distance - maximum_distance, distance + maximum_distance
            stack.extend(child for edge, child in node["children"].items() if lower <= edge <= upper)
        return sorted(result, key=lambda item: (item[0], item[1]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, choices=(640,), default=640)
    parser.add_argument("--near-hamming", type=int, default=4)
    parser.add_argument("--max-near-pairs", type=int, default=5000)
    parser.add_argument("--contact-sheet-pairs", type=int, default=40)
    parser.add_argument("--output-dir", required=True)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not 0 <= args.near_hamming <= 16:
        parser.error("--near-hamming must be in [0, 16]")
    if args.max_near_pairs <= 0 or args.contact_sheet_pairs < 0:
        parser.error("--max-near-pairs must be positive and --contact-sheet-pairs must be non-negative")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(cv2, path: str | Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"unable to read image for perceptual hashing: {path}")
    resized = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    differences = resized[:, 1:] > resized[:, :-1]
    value = 0
    for bit in differences.reshape(-1).tolist():
        value = (value << 1) | int(bit)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_contact_sheets(cv2, output_dir: Path, rows: list[dict[str, Any]], maximum_pairs: int) -> list[str]:
    """Write readable side-by-side pages for manual near-duplicate review."""
    if maximum_pairs <= 0 or not rows:
        return []
    import numpy as np

    selected = sorted(rows, key=lambda row: (int(row["hamming_distance"]), row["val_image"], row["train_image"]))[
        :maximum_pairs
    ]
    pair_width, image_height, label_height = 720, 220, 44
    pairs_per_page = 8
    written = []
    for page_index in range(0, len(selected), pairs_per_page):
        page_rows = selected[page_index : page_index + pairs_per_page]
        canvas = np.full((len(page_rows) * (image_height + label_height), pair_width, 3), 248, dtype=np.uint8)
        for row_index, row in enumerate(page_rows):
            y0 = row_index * (image_height + label_height)
            for side, key in enumerate(("train_image", "val_image")):
                image = cv2.imread(str(row[key]), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                scale = min((pair_width // 2) / image.shape[1], image_height / image.shape[0])
                width = max(1, int(round(image.shape[1] * scale)))
                height = max(1, int(round(image.shape[0] * scale)))
                resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
                x_base = side * (pair_width // 2)
                x = x_base + ((pair_width // 2) - width) // 2
                y = y0 + (image_height - height) // 2
                canvas[y : y + height, x : x + width] = resized
            label = (
                f"dHash={int(row['hamming_distance'])} | train={Path(row['train_image']).name} | "
                f"val={Path(row['val_image']).name}"
            )
            cv2.putText(
                canvas,
                label[:110],
                (8, y0 + image_height + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
        page_number = page_index // pairs_per_page + 1
        path = output_dir / f"near_duplicate_contact_sheet_{page_number:02d}.jpg"
        if not cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"failed to write contact sheet: {path}")
        written.append(path.name)
    return written


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    omp_threads = os.environ.get("OMP_NUM_THREADS", "")
    if not omp_threads.isdigit() or int(omp_threads) <= 0:
        os.environ["OMP_NUM_THREADS"] = "1"

    import cv2

    from .runtime import build_dataset

    output_dir = Path(args.output_dir)
    if (output_dir / "split_audit.json").exists():
        raise FileExistsError(f"split audit output already exists: {output_dir}; use a new directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset, _ = build_dataset(args.data, "train", args.imgsz, batch=1, workers=0, rect=False)
    val_dataset, _ = build_dataset(args.data, "val", args.imgsz, batch=1, workers=0, rect=False)
    train_paths = [str(Path(path).resolve()) for path in train_dataset.im_files]
    val_paths = [str(Path(path).resolve()) for path in val_dataset.im_files]

    train_path_set = {Path(path).as_posix().casefold() for path in train_paths}
    path_overlap = [path for path in val_paths if Path(path).as_posix().casefold() in train_path_set]
    train_stems: dict[str, list[str]] = {}
    for path in train_paths:
        train_stems.setdefault(Path(path).stem.casefold(), []).append(path)
    stem_rows = []
    for val_path in val_paths:
        for train_path in train_stems.get(Path(val_path).stem.casefold(), []):
            stem_rows.append({"train_image": train_path, "val_image": val_path, "stem": Path(val_path).stem})

    print(f"Hashing {len(train_paths)} train and {len(val_paths)} val images...")
    train_sha: dict[str, list[str]] = {}
    train_tree = HammingBKTree()
    train_dhash: dict[str, int] = {}
    for index, path in enumerate(train_paths, 1):
        digest = sha256_file(path)
        train_sha.setdefault(digest, []).append(path)
        perceptual = difference_hash(cv2, path)
        train_dhash[path] = perceptual
        train_tree.add(perceptual, path)
        if index % 1000 == 0:
            print(f"  train {index}/{len(train_paths)}")

    exact_rows = []
    near_rows = []
    near_truncated = False
    for index, val_path in enumerate(val_paths, 1):
        digest = sha256_file(val_path)
        perceptual = difference_hash(cv2, val_path)
        for train_path in train_sha.get(digest, []):
            exact_rows.append(
                {
                    "train_image": train_path,
                    "val_image": val_path,
                    "sha256": digest,
                }
            )
        if len(near_rows) < args.max_near_pairs:
            for distance, train_path in train_tree.query(perceptual, args.near_hamming):
                near_rows.append(
                    {
                        "train_image": train_path,
                        "val_image": val_path,
                        "hamming_distance": distance,
                        "train_dhash": f"{train_dhash[train_path]:016x}",
                        "val_dhash": f"{perceptual:016x}",
                        "exact_binary": int(digest in train_sha and train_path in train_sha[digest]),
                    }
                )
                if len(near_rows) >= args.max_near_pairs:
                    near_truncated = True
                    break
        if index % 500 == 0:
            print(f"  val {index}/{len(val_paths)}")

    audit = {
        "data": args.data,
        "imgsz": args.imgsz,
        "train_images": len(train_paths),
        "val_images": len(val_paths),
        "path_overlap_count": len(path_overlap),
        "stem_overlap_pair_count": len(stem_rows),
        "exact_binary_overlap_pair_count": len(exact_rows),
        "near_duplicate_hamming_threshold": args.near_hamming,
        "near_duplicate_pair_count_saved": len(near_rows),
        "near_duplicate_pairs_truncated": near_truncated,
        "hard_split_integrity_pass": len(path_overlap) == 0 and len(exact_rows) == 0,
        "test_used": False,
    }
    write_csv(output_dir / "path_overlap.csv", [{"path": path} for path in path_overlap])
    write_csv(output_dir / "stem_overlap.csv", stem_rows)
    write_csv(output_dir / "exact_binary_overlap.csv", exact_rows)
    write_csv(output_dir / "near_duplicate_pairs.csv", near_rows)
    contact_sheets = write_contact_sheets(cv2, output_dir, near_rows, args.contact_sheet_pairs)
    audit["near_duplicate_contact_sheets"] = contact_sheets
    write_json(output_dir / "split_audit.json", audit)
    report = [
        "# Refine V3 数据划分真实性审计",
        "",
        f"- train/val：{len(train_paths)} / {len(val_paths)} 张。",
        f"- 完整路径交叉：{len(path_overlap)}。",
        f"- 同名 stem 配对：{len(stem_rows)}。",
        f"- 文件 SHA256 完全重复：{len(exact_rows)}。",
        f"- dHash 距离≤{args.near_hamming} 的候选配对：{len(near_rows)}（截断={near_truncated}）。",
        f"- 人工核对拼图：{len(contact_sheets)} 页。",
        f"- hard split integrity：{audit['hard_split_integrity_pass']}。",
        "- test：未读取。",
        "",
        "dHash 候选不能自动判为泄漏；细长电力线与相似背景可能产生感知哈希碰撞，需要人工查看候选图片。",
    ]
    (output_dir / "split_audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if not audit["hard_split_integrity_pass"]:
        raise RuntimeError(f"train/val exact overlap detected; inspect {output_dir / 'split_audit.json'}")
    print(output_dir / "split_audit_report.md")


if __name__ == "__main__":
    main()
