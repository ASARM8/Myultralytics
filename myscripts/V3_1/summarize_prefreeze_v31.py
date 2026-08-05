"""Summarize pre-declared Refine V3.1 candidate checks without score fishing."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_VARIANTS = (
    "coarse",
    "roundtrip",
    "gate_off",
    "selected_gate",
    "all_refine",
    "all_refine_no_renms",
    "short_only",
    "short_only_all",
    "short_only_all_no_renms",
    "long_only",
    "long_only_all",
)
DECISION_METRICS = ("map50_95", "ap75", "ap90")
IDENTITY_METRICS = ("precision", "recall", "map50", "map50_95", "ap75", "ap90", "ap95")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-audit-dir", required=True)
    parser.add_argument("--clean-val-audit-dir", required=True)
    parser.add_argument("--split-preparation-dir", required=True)
    parser.add_argument("--noninferiority-tolerance", type=float, default=0.002)
    parser.add_argument("--identity-tolerance", type=float, default=5e-4)
    parser.add_argument("--minimum-refine-gain", type=float, default=0.002)
    parser.add_argument("--output-dir", required=True)
    return parser


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_metrics(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    lookup = {row["variant"]: {key: _number(value) for key, value in row.items()} for row in rows}
    missing = set(REQUIRED_VARIANTS) - lookup.keys()
    if missing:
        raise RuntimeError(f"audit is missing V3.1 pre-freeze variants: {sorted(missing)}")
    return lookup


def _number(value: str) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def metric_delta(metrics: dict[str, dict[str, Any]], candidate: str, reference: str, metric: str) -> float:
    return float(metrics[candidate][metric]) - float(metrics[reference][metric])


def noninferior(
    metrics: dict[str, dict[str, Any]], candidate: str, reference: str, tolerance: float
) -> tuple[bool, dict[str, float]]:
    deltas = {metric: metric_delta(metrics, candidate, reference, metric) for metric in DECISION_METRICS}
    return all(delta >= -tolerance for delta in deltas.values()), deltas


def identity(
    metrics: dict[str, dict[str, Any]], candidate: str, reference: str, tolerance: float
) -> tuple[bool, dict[str, float]]:
    deltas = {metric: metric_delta(metrics, candidate, reference, metric) for metric in IDENTITY_METRICS}
    return max((abs(value) for value in deltas.values()), default=0.0) <= tolerance, deltas


def refine_gain_pass(
    metrics: dict[str, dict[str, Any]], candidate: str, minimum_gain: float
) -> tuple[bool, dict[str, float]]:
    deltas = {
        metric: metric_delta(metrics, candidate, "coarse", metric)
        for metric in ("map50_95", "ap75", "ap90")
    }
    passed = deltas["map50_95"] >= minimum_gain and deltas["ap75"] >= 0.0 and deltas["ap90"] >= -0.002
    return passed, deltas


def evaluate_scope(
    metrics: dict[str, dict[str, Any]],
    *,
    noninferiority_tolerance: float,
    identity_tolerance: float,
    minimum_gain: float,
) -> dict[str, Any]:
    short_pass, short_deltas = noninferior(
        metrics, "short_only_all", "all_refine", noninferiority_tolerance
    )
    all_pass, all_deltas = noninferior(
        metrics, "short_only_all", "short_only", noninferiority_tolerance
    )
    renms_pass, renms_deltas = identity(
        metrics, "short_only_all", "short_only_all_no_renms", identity_tolerance
    )
    gain_pass, gain_deltas = refine_gain_pass(metrics, "short_only_all", minimum_gain)
    return {
        "short_branch_noninferior_to_full_all": short_pass,
        "short_branch_deltas_vs_full_all": short_deltas,
        "all_proposals_noninferior_to_quality_gate": all_pass,
        "all_proposals_deltas_vs_quality_gate": all_deltas,
        "renms_identity_for_short_all": renms_pass,
        "renms_deltas": renms_deltas,
        "short_all_refine_gain_pass": gain_pass,
        "short_all_refine_deltas_vs_coarse": gain_deltas,
    }


def format_deltas(values: dict[str, float]) -> str:
    return ", ".join(f"{key}={value:+.6f}" for key, value in values.items())


def main() -> None:
    args = build_parser().parse_args()
    if min(args.noninferiority_tolerance, args.identity_tolerance, args.minimum_refine_gain) <= 0:
        raise ValueError("all tolerances and gain thresholds must be positive")
    output_dir = Path(args.output_dir)
    if (output_dir / "v31_prefreeze_decision.json").exists():
        raise FileExistsError(f"summary output already exists: {output_dir}; use a new directory")

    holdout_dir = Path(args.holdout_audit_dir)
    clean_val_dir = Path(args.clean_val_audit_dir)
    split_dir = Path(args.split_preparation_dir)
    holdout_audit = read_json(holdout_dir / "truth_audit.json")
    clean_val_audit = read_json(clean_val_dir / "truth_audit.json")
    split_manifest = read_json(split_dir / "prefreeze_split_manifest.json")
    holdout_metrics = read_metrics(holdout_dir / "mechanism_metrics.csv")
    clean_val_metrics = read_metrics(clean_val_dir / "mechanism_metrics.csv")

    if holdout_audit.get("evaluation_scope") != "train-holdout":
        raise RuntimeError("holdout audit was not run with --evaluation-scope=train-holdout")
    if clean_val_audit.get("evaluation_scope") != "val":
        raise RuntimeError("clean-val audit was not run with --evaluation-scope=val")
    expected_exclusions = int(split_manifest["exact_overlap_unique_val_images_excluded"])
    expected_holdout_exclusions = int(split_manifest["fit_holdout_unique_holdout_images_excluded"])
    if int(clean_val_audit.get("excluded_images", -1)) != expected_exclusions:
        raise RuntimeError("clean-val audit did not exclude the exact-overlap validation images")
    if int(holdout_audit.get("excluded_images", -1)) != expected_holdout_exclusions:
        raise RuntimeError("train-holdout audit did not exclude exact duplicates found in train-fit")
    if split_manifest.get("test_used") is not False:
        raise RuntimeError("split preparation does not prove that test remained unused")
    if holdout_audit.get("checkpoint_sha256") != clean_val_audit.get("checkpoint_sha256"):
        raise RuntimeError("holdout and clean-val audits used different V3 checkpoints")
    if holdout_audit.get("ca_sha256") != clean_val_audit.get("ca_sha256"):
        raise RuntimeError("holdout and clean-val audits used different CA checkpoints")

    holdout = evaluate_scope(
        holdout_metrics,
        noninferiority_tolerance=args.noninferiority_tolerance,
        identity_tolerance=args.identity_tolerance,
        minimum_gain=args.minimum_refine_gain,
    )
    clean_val = evaluate_scope(
        clean_val_metrics,
        noninferiority_tolerance=args.noninferiority_tolerance,
        identity_tolerance=args.identity_tolerance,
        minimum_gain=args.minimum_refine_gain,
    )
    scope_results = {"train_holdout": holdout, "clean_val": clean_val}

    retain_short_only = all(item["short_branch_noninferior_to_full_all"] for item in scope_results.values())
    apply_all_proposals = all(item["all_proposals_noninferior_to_quality_gate"] for item in scope_results.values())
    remove_renms = all(item["renms_identity_for_short_all"] for item in scope_results.values())
    refine_signal_pass = all(item["short_all_refine_gain_pass"] for item in scope_results.values())
    integrity_pass = bool(holdout_audit.get("hard_integrity_pass")) and bool(clean_val_audit.get("hard_integrity_pass"))
    source_frame_overlap_count = int(split_manifest.get("source_frame_overlap_count", -1))
    fit_holdout_source_overlap_count = int(split_manifest.get("fit_holdout_source_frame_overlap_count", -1))
    formal_split_ready = source_frame_overlap_count == 0 and fit_holdout_source_overlap_count == 0

    primary = {
        "retain_rotated_roi": True,
        "geometry_output": "short_only" if retain_short_only else "short_and_long",
        "proposal_policy": "all" if apply_all_proposals else "frozen_quality_gate",
        "rerun_nms": not remove_renms,
        "quality_head": "training_ablation_required" if apply_all_proposals else "retain_for_gate",
        "requires_retraining": True,
    }
    ready = integrity_pass and refine_signal_pass
    decision = {
        "stage": "V3.1 pre-freeze decision",
        "status": "READY_TO_IMPLEMENT_V3.1" if ready else "MORE_DIAGNOSIS_REQUIRED",
        "selection_policy": "pre-declared non-inferiority and identity rules; not maximum-score selection",
        "thresholds": {
            "noninferiority_tolerance": args.noninferiority_tolerance,
            "identity_tolerance": args.identity_tolerance,
            "minimum_refine_gain": args.minimum_refine_gain,
        },
        "checkpoint_sha256": holdout_audit["checkpoint_sha256"],
        "ca_sha256": holdout_audit["ca_sha256"],
        "exact_val_images_excluded": expected_exclusions,
        "exact_holdout_images_excluded": expected_holdout_exclusions,
        "scope_results": scope_results,
        "integrity_pass": integrity_pass,
        "refine_signal_pass_both_scopes": refine_signal_pass,
        "formal_split_ready": formal_split_ready,
        "source_frame_overlap_count": source_frame_overlap_count,
        "fit_holdout_source_frame_overlap_count": fit_holdout_source_overlap_count,
        "primary_design": primary,
        "formal_evidence_boundary": {
            "current_checkpoint": "diagnostic only; it was not trained with the simplified V3.1 topology",
            "quality_auxiliary": "cannot be decided by inference switches; requires one matched retraining ablation",
            "validation": "clean-val is a sensitivity audit, not a replacement for a provenance-correct split",
            "test_used": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "v31_prefreeze_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Refine V3.1 冻结前验证结论",
        "",
        f"- 状态：**{decision['status']}**",
        f"- checkpoint SHA256：`{decision['checkpoint_sha256']}`",
        f"- clean-val 排除的精确重复图像：{expected_exclusions} 张",
        f"- clean-holdout 排除的 fit 精确重复图像：{expected_holdout_exclusions} 张",
        f"- 同一源帧跨 train/val：{source_frame_overlap_count} 组；正式证据划分可用：{formal_split_ready}",
        f"- 同一源帧跨 train-fit/holdout：{fit_holdout_source_overlap_count} 组",
        "- test：未读取。",
        "",
        "## 两个评估范围的预声明检查",
        "",
        "| 范围 | 短边非劣于完整双边 | 全 proposal 非劣于质量门控 | 去 re-NMS 恒等 | Refine 增益通过 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, result in scope_results.items():
        lines.append(
            f"| {name} | {result['short_branch_noninferior_to_full_all']} | "
            f"{result['all_proposals_noninferior_to_quality_gate']} | "
            f"{result['renms_identity_for_short_all']} | {result['short_all_refine_gain_pass']} |"
        )
    lines.extend(
        [
            "",
            "## 固定比较的具体差值",
            "",
        ]
    )
    for name, result in scope_results.items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- short-only-all − full-all：{format_deltas(result['short_branch_deltas_vs_full_all'])}",
                f"- short-only-all − short-only-gated：{format_deltas(result['all_proposals_deltas_vs_quality_gate'])}",
                f"- short-only-all − coarse：{format_deltas(result['short_all_refine_deltas_vs_coarse'])}",
                f"- short-only-all − no-reNMS 最大绝对差："
                f"{max(abs(value) for value in result['renms_deltas'].values()):.6g}",
                "",
            ]
        )
    lines.extend(
        [
            "## V3.1 主方案建议",
            "",
            f"- 几何输出：`{primary['geometry_output']}`",
            f"- proposal 策略：`{primary['proposal_policy']}`",
            f"- 是否重新执行 NMS：`{primary['rerun_nms']}`",
            f"- quality head：`{primary['quality_head']}`",
            "- 必须重新训练：是。现有 checkpoint 只能验证删减方向，不能充当简化拓扑的正式权重。",
            "",
            "## 证据边界",
            "",
            "- 本工具不按最高 mAP 自由选组合，只执行预先声明的非劣性、恒等性和最小增益判断。",
            "- clean-val 仅排除已证实的逐字节重复图像；场景级划分仍需结合数据来源核对。",
            "- 若任一源帧交叉计数大于0，可以继续实现 V3.1，但正式训练前必须按来源分组重建 fit/holdout 与 train/val。",
            "- quality 是否作为辅助监督保留，无法通过推理开关决定，需要一次等预算重训练消融。",
            "- test 继续封存，直到 V3.1 结构、训练规则、epoch 选择规则和多种子方案全部冻结。",
        ]
    )
    (output_dir / "v31_prefreeze_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_dir / "v31_prefreeze_decision.md")


if __name__ == "__main__":
    main()
