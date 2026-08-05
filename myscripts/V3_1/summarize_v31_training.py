"""Compare the equal-budget V3.1 geometry-only and quality-auxiliary runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select the simplest accepted Refine V3.1 training variant.")
    parser.add_argument("--geometry-dir", required=True)
    parser.add_argument("--quality-aux-dir", required=True)
    parser.add_argument("--noninferiority-tolerance", type=float, default=0.002)
    parser.add_argument("--output-dir", required=True)
    return parser


def read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def refined_row(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    row = next((item for item in rows if item.get("variant") == "refined"), None)
    if row is None:
        raise RuntimeError(f"refined row missing from {path}")
    return {key: float(row[key]) for key in ("map50_95", "ap75", "ap90", "ap95")}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.noninferiority_tolerance <= 0:
        parser.error("--noninferiority-tolerance must be positive")
    geometry_dir = Path(args.geometry_dir)
    quality_dir = Path(args.quality_aux_dir)
    geometry_acceptance = read_json(geometry_dir / "acceptance.json")
    quality_acceptance = read_json(quality_dir / "acceptance.json")
    if geometry_acceptance.get("experiment") != "geometry_only":
        raise RuntimeError("--geometry-dir is not a geometry_only run")
    if quality_acceptance.get("experiment") != "quality_aux":
        raise RuntimeError("--quality-aux-dir is not a quality_aux run")
    if geometry_acceptance.get("ca_hash_before") != quality_acceptance.get("ca_hash_before"):
        raise RuntimeError("the two runs do not use the same CA checkpoint hash")

    geometry = refined_row(geometry_dir / "val_metrics.csv")
    quality = refined_row(quality_dir / "val_metrics.csv")
    tolerance = args.noninferiority_tolerance
    geometry_noninferior = all(
        geometry[key] >= quality[key] - tolerance for key in ("map50_95", "ap75", "ap90")
    )
    geometry_pass = bool(geometry_acceptance.get("screening_pass"))
    quality_pass = bool(quality_acceptance.get("screening_pass"))
    if geometry_pass and geometry_noninferior:
        decision, reason = "geometry_only", "通过验收且相对 quality_aux 三项指标均非劣，选择更简单结构"
    elif quality_pass:
        decision, reason = "quality_aux", "只有 quality_aux 通过或 geometry_only 相对其劣化超过预声明容差"
    elif geometry_pass:
        decision, reason = "geometry_only", "geometry_only 通过而 quality_aux 未通过"
    else:
        decision, reason = "NEEDS_OPTIMIZATION", "两个等预算版本均未通过预声明验收"

    payload = {
        "stage": "Refine V3.1 equal-budget selection",
        "decision": decision,
        "reason": reason,
        "noninferiority_tolerance": tolerance,
        "geometry_screening_pass": geometry_pass,
        "quality_aux_screening_pass": quality_pass,
        "geometry_noninferior": geometry_noninferior,
        "geometry_refined": geometry,
        "quality_aux_refined": quality,
        "delta_geometry_minus_quality_aux": {key: geometry[key] - quality[key] for key in geometry},
        "ca_sha256": geometry_acceptance.get("ca_hash_before"),
        "test_used": False,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "v31_training_decision.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Refine V3.1 等预算训练选择",
        "",
        f"- 决定：**{decision}**",
        f"- 原因：{reason}",
        f"- geometry_only mAP50-95：{geometry['map50_95']:.6f}",
        f"- quality_aux mAP50-95：{quality['map50_95']:.6f}",
        f"- geometry_only − quality_aux：{geometry['map50_95'] - quality['map50_95']:+.6f}",
        f"- geometry_only 非劣：{geometry_noninferior}",
        f"- 两项筛选结果：{geometry_pass} / {quality_pass}",
        "- test：未使用。",
        "",
    ]
    (output_dir / "v31_training_decision.md").write_text("\n".join(report), encoding="utf-8")
    print(output_dir / "v31_training_decision.json")


if __name__ == "__main__":
    main()
