"""Prepare leakage-aware inputs for the Refine V3.1 pre-freeze audit.

The script consumes the completed V3 split audit, verifies its accounting,
creates an exact-overlap exclusion manifest for a clean validation pass, and
records label consistency plus source/scene overlap diagnostics. It never
reads or modifies the test split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SCENE_REGEX = r"^([^_]+)_"
TILE_SUFFIX = re.compile(r"_x\d+_y\d+$", flags=re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split-audit-dir", required=True)
    parser.add_argument("--imgsz", type=int, choices=(640,), default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--holdout-seed", type=int, default=0)
    parser.add_argument(
        "--holdout-group-regex",
        default="",
        help="Must reproduce the V3 training split; the current seed0 run used an empty value.",
    )
    parser.add_argument(
        "--scene-regex",
        default=DEFAULT_SCENE_REGEX,
        help="Heuristic regex whose first capture group denotes a scene. Scene overlap is diagnostic only.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_frame_id(path: str | Path) -> str:
    """Infer the pre-tiling source-frame identifier from a crop filename."""
    return TILE_SUFFIX.sub("", Path(path).stem)


def scene_id(path: str | Path, pattern: str) -> str:
    match = re.search(pattern, source_frame_id(path))
    if not match or match.lastindex is None:
        raise ValueError(f"scene regex did not match with a capture group: {path}")
    return match.group(1)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _label_hash(path: str | Path) -> str | None:
    label = Path(path)
    return sha256_file(label) if label.is_file() else None


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.holdout_fraction < 1.0:
        raise ValueError("--holdout-fraction must be in (0, 1)")
    if args.batch <= 0 or args.workers < 0:
        raise ValueError("--batch must be positive and --workers must be non-negative")
    audit_dir = Path(args.split_audit_dir)
    output_dir = Path(args.output_dir)
    if (output_dir / "prefreeze_split_manifest.json").exists():
        raise FileExistsError(f"pre-freeze output already exists: {output_dir}; use a new directory")
    exact_csv = audit_dir / "exact_binary_overlap.csv"
    near_csv = audit_dir / "near_duplicate_pairs.csv"
    audit_json = audit_dir / "split_audit.json"
    for required in (exact_csv, near_csv, audit_json):
        if not required.is_file():
            raise FileNotFoundError(f"missing V3 split-audit artifact: {required}")

    split_audit = json.loads(audit_json.read_text(encoding="utf-8"))
    if split_audit.get("test_used") is not False:
        raise RuntimeError("split audit does not prove that test remained unused")
    exact_rows = read_csv(exact_csv)
    near_rows = read_csv(near_csv)
    if len(exact_rows) != int(split_audit.get("exact_binary_overlap_pair_count", -1)):
        raise RuntimeError("exact-overlap CSV count differs from split_audit.json")

    # Heavy imports stay inside main so --help and pure helper tests work in a
    # local document-processing environment without torch/Ultralytics.
    from ultralytics.data.utils import img2label_paths

    from myscripts.V3.runtime import build_dataset, split_dataset_indices

    train_dataset, _ = build_dataset(args.data, "train", args.imgsz, args.batch, args.workers, rect=False)
    val_dataset, _ = build_dataset(args.data, "val", args.imgsz, args.batch, args.workers, rect=True)
    train_paths = [str(Path(path)) for path in train_dataset.im_files]
    val_paths = [str(Path(path)) for path in val_dataset.im_files]
    train_set = {Path(path).as_posix() for path in train_paths}
    val_set = {Path(path).as_posix() for path in val_paths}

    unique_val_overlap = sorted({row["val_image"] for row in exact_rows})
    unique_train_overlap = sorted({row["train_image"] for row in exact_rows})
    missing_train = [path for path in unique_train_overlap if Path(path).as_posix() not in train_set]
    missing_val = [path for path in unique_val_overlap if Path(path).as_posix() not in val_set]
    if missing_train or missing_val:
        raise RuntimeError(
            "split-audit paths do not match the current dataset: "
            f"missing train={len(missing_train)}, val={len(missing_val)}"
        )

    fit_indices, holdout_indices, fit_groups, holdout_groups = split_dataset_indices(
        train_paths,
        args.holdout_fraction,
        args.holdout_seed,
        args.holdout_group_regex,
    )
    fit_set = set(fit_indices)
    hash_to_fit: dict[str, list[str]] = {}
    hash_to_holdout: dict[str, list[str]] = {}
    for index, image_path in enumerate(train_paths):
        digest = sha256_file(image_path)
        destination = hash_to_fit if index in fit_set else hash_to_holdout
        destination.setdefault(digest, []).append(image_path)
        if (index + 1) % 1000 == 0:
            print(f"  hashed train {index + 1}/{len(train_paths)}")
    internal_exact_rows = []
    for digest in sorted(hash_to_fit.keys() & hash_to_holdout.keys()):
        for fit_image in hash_to_fit[digest]:
            for holdout_image in hash_to_holdout[digest]:
                internal_exact_rows.append(
                    {"fit_image": fit_image, "holdout_image": holdout_image, "sha256": digest}
                )
    unique_holdout_overlap = sorted({row["holdout_image"] for row in internal_exact_rows})

    fit_sources: dict[str, list[str]] = {}
    holdout_sources: dict[str, list[str]] = {}
    for index in fit_indices:
        path = train_paths[index]
        fit_sources.setdefault(source_frame_id(path), []).append(path)
    for index in holdout_indices:
        path = train_paths[index]
        holdout_sources.setdefault(source_frame_id(path), []).append(path)
    holdout_source_overlap_rows = [
        {
            "source_frame": key,
            "fit_crops": len(fit_sources[key]),
            "holdout_crops": len(holdout_sources[key]),
            "fit_example": fit_sources[key][0],
            "holdout_example": holdout_sources[key][0],
        }
        for key in sorted(fit_sources.keys() & holdout_sources.keys())
    ]

    train_labels = {Path(image).as_posix(): label for image, label in zip(train_paths, img2label_paths(train_paths))}
    val_labels = {Path(image).as_posix(): label for image, label in zip(val_paths, img2label_paths(val_paths))}
    label_rows = []
    for row in exact_rows:
        train_label = train_labels[Path(row["train_image"]).as_posix()]
        val_label = val_labels[Path(row["val_image"]).as_posix()]
        train_hash = _label_hash(train_label)
        val_hash = _label_hash(val_label)
        label_rows.append(
            {
                **row,
                "train_label": train_label,
                "val_label": val_label,
                "train_label_sha256": train_hash,
                "val_label_sha256": val_hash,
                "label_hash_equal": int(train_hash is not None and train_hash == val_hash),
                "missing_label": int(train_hash is None or val_hash is None),
            }
        )

    train_sources: dict[str, list[str]] = {}
    val_sources: dict[str, list[str]] = {}
    train_scenes: dict[str, list[str]] = {}
    val_scenes: dict[str, list[str]] = {}
    for path in train_paths:
        train_sources.setdefault(source_frame_id(path), []).append(path)
        train_scenes.setdefault(scene_id(path, args.scene_regex), []).append(path)
    for path in val_paths:
        val_sources.setdefault(source_frame_id(path), []).append(path)
        val_scenes.setdefault(scene_id(path, args.scene_regex), []).append(path)

    source_overlap_rows = [
        {
            "source_frame": key,
            "train_crops": len(train_sources[key]),
            "val_crops": len(val_sources[key]),
            "train_example": train_sources[key][0],
            "val_example": val_sources[key][0],
        }
        for key in sorted(train_sources.keys() & val_sources.keys())
    ]
    scene_overlap_rows = [
        {
            "scene": key,
            "train_images": len(train_scenes[key]),
            "val_images": len(val_scenes[key]),
            "train_source_frames": len({source_frame_id(path) for path in train_scenes[key]}),
            "val_source_frames": len({source_frame_id(path) for path in val_scenes[key]}),
            "train_example": train_scenes[key][0],
            "val_example": val_scenes[key][0],
        }
        for key in sorted(train_scenes.keys() & val_scenes.keys())
    ]

    prioritized_near = []
    for row in near_rows:
        if str(row.get("exact_binary", "0")) == "1":
            continue
        train_image = row["train_image"]
        val_image = row["val_image"]
        train_source = source_frame_id(train_image)
        val_source = source_frame_id(val_image)
        train_scene = scene_id(train_image, args.scene_regex)
        val_scene = scene_id(val_image, args.scene_regex)
        prioritized_near.append(
            {
                **row,
                "same_source_frame": int(train_source == val_source),
                "same_scene": int(train_scene == val_scene),
                "train_source_frame": train_source,
                "val_source_frame": val_source,
                "train_scene": train_scene,
                "val_scene": val_scene,
            }
        )
    prioritized_near.sort(
        key=lambda row: (
            -int(row["same_source_frame"]),
            -int(row["same_scene"]),
            int(row["hamming_distance"]),
            row["val_image"],
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "clean_val_exclusions.txt").write_text(
        "\n".join(unique_val_overlap) + "\n",
        encoding="utf-8",
    )
    (output_dir / "clean_holdout_exclusions.txt").write_text(
        ("\n".join(unique_holdout_overlap) + "\n") if unique_holdout_overlap else "# no exact fit/holdout overlap\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "exact_overlap_label_audit.csv", label_rows)
    write_csv(output_dir / "train_fit_holdout_exact_overlap.csv", internal_exact_rows)
    write_csv(output_dir / "train_fit_holdout_source_frame_overlap.csv", holdout_source_overlap_rows)
    write_csv(output_dir / "source_frame_overlap.csv", source_overlap_rows)
    write_csv(output_dir / "scene_overlap_heuristic.csv", scene_overlap_rows)
    write_csv(output_dir / "prioritized_near_duplicate_review.csv", prioritized_near)

    manifest = {
        "stage": "V3.1 pre-freeze split preparation",
        "data": args.data,
        "imgsz": args.imgsz,
        "train_images": len(train_paths),
        "train_fit_images": len(fit_indices),
        "train_holdout_images_original": len(holdout_indices),
        "holdout_fraction": args.holdout_fraction,
        "holdout_seed": args.holdout_seed,
        "holdout_group_regex": args.holdout_group_regex,
        "holdout_group_overlap": len(fit_groups & holdout_groups),
        "fit_holdout_exact_overlap_pairs": len(internal_exact_rows),
        "fit_holdout_unique_holdout_images_excluded": len(unique_holdout_overlap),
        "clean_holdout_images": len(holdout_indices) - len(unique_holdout_overlap),
        "fit_holdout_source_frame_overlap_count": len(holdout_source_overlap_rows),
        "val_images_original": len(val_paths),
        "exact_overlap_pairs": len(exact_rows),
        "exact_overlap_unique_train_images": len(unique_train_overlap),
        "exact_overlap_unique_val_images_excluded": len(unique_val_overlap),
        "clean_val_images": len(val_paths) - len(unique_val_overlap),
        "exact_overlap_unique_hashes": len({row["sha256"] for row in exact_rows}),
        "exact_overlap_pairs_with_equal_labels": sum(row["label_hash_equal"] for row in label_rows),
        "exact_overlap_pairs_with_missing_labels": sum(row["missing_label"] for row in label_rows),
        "source_frame_overlap_count": len(source_overlap_rows),
        "scene_regex": args.scene_regex,
        "scene_overlap_count_heuristic": len(scene_overlap_rows),
        "near_duplicate_pairs_non_exact": len(prioritized_near),
        "near_duplicate_pairs_same_source": sum(row["same_source_frame"] for row in prioritized_near),
        "near_duplicate_pairs_same_scene": sum(row["same_scene"] for row in prioritized_near),
        "clean_val_exclusions": "clean_val_exclusions.txt",
        "clean_holdout_exclusions": "clean_holdout_exclusions.txt",
        "test_used": False,
        "interpretation": {
            "exact_overlap": "excluded from the clean-val sensitivity audit",
            "source_frame_overlap": "hard leakage risk requiring split review",
            "fit_holdout_overlap": (
                "removed only for frozen-checkpoint sensitivity audit; checkpoint selection remains historical"
            ),
            "scene_overlap": "heuristic only; must be interpreted using dataset naming provenance",
            "perceptual_near_duplicate": "manual-review queue, never automatically treated as leakage",
        },
    }
    write_json(output_dir / "prefreeze_split_manifest.json", manifest)
    print(output_dir / "prefreeze_split_manifest.json")


if __name__ == "__main__":
    main()
