"""Audit three independent evaluation paths for the frozen low-gain setting."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from myscripts.V3_1_1_low_gain.config import DEFAULT_RESIDUAL_SCALE


METRICS = ("precision", "recall", "map50", "map50_95") + tuple(f"ap{x}" for x in range(50, 100, 5))
VARIANTS = ("coarse", "refined")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-batch8-dir", required=True)
    parser.add_argument("--amp-batch8-dir", required=True)
    parser.add_argument("--fp32-batch1-dir", required=True)
    parser.add_argument("--residual-scale", type=float, default=DEFAULT_RESIDUAL_SCALE)
    parser.add_argument("--metric-tolerance", type=float, default=1e-3)
    parser.add_argument("--strict", action="store_true", help="Return a non-zero status when the audit threshold fails")
    parser.add_argument("--output-dir", required=True)
    return parser


def _json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics(path: Path) -> dict[str, dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = {}
    for variant in VARIANTS:
        row = next((item for item in rows if item.get("variant") == variant), None)
        if row is None:
            raise RuntimeError(f"{variant} row missing from {path}")
        missing = [metric for metric in METRICS if metric not in row]
        if missing:
            raise RuntimeError(f"metrics missing from {path}: {missing}")
        output[variant] = {metric: float(row[metric]) for metric in METRICS}
    return output


def _differences(left: dict, right: dict) -> dict[str, float]:
    return {
        f"{variant}.{metric}": left[variant][metric] - right[variant][metric]
        for variant in VARIANTS
        for metric in METRICS
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.metric_tolerance <= 0:
        parser.error("metric-tolerance must be positive")
    if abs(args.residual_scale - DEFAULT_RESIDUAL_SCALE) > 1e-12:
        parser.error(f"formal low-gain audit is locked to residual-scale={DEFAULT_RESIDUAL_SCALE}")

    run_dirs = {
        "fp32_batch8": Path(args.fp32_batch8_dir),
        "amp_batch8": Path(args.amp_batch8_dir),
        "fp32_batch1": Path(args.fp32_batch1_dir),
    }
    audits = {name: _json(path / "low_gain_audit.json") for name, path in run_dirs.items()}
    metrics = {name: _metrics(path / "val_metrics.csv") for name, path in run_dirs.items()}
    expected = {
        "fp32_batch8": (False, 8),
        "amp_batch8": (True, 8),
        "fp32_batch1": (False, 1),
    }
    protocol_pass = {
        name: audit.get("architecture") == "OBBProposalRefinerV311"
        and audit.get("split") == "val"
        and bool(audit.get("amp")) == mode[0]
        and int(audit.get("batch", -1)) == mode[1]
        and float(audit.get("residual_scale", -1.0)) == args.residual_scale
        and audit.get("test_used") is False
        and audit.get("weights_modified") is False
        and audit.get("rerun_nms") is False
        and float(audit.get("identity_max_abs_metric_delta", 1.0)) == 0.0
        for name, audit in audits.items()
        for mode in (expected[name],)
    }
    checkpoint_hashes = {audit.get("checkpoint_sha256") for audit in audits.values()}
    ca_hashes = {audit.get("ca_sha256") for audit in audits.values()}
    hash_pass = len(checkpoint_hashes) == 1 and len(ca_hashes) == 1 and None not in checkpoint_hashes | ca_hashes
    amp_delta = _differences(metrics["amp_batch8"], metrics["fp32_batch8"])
    batch_delta = _differences(metrics["fp32_batch1"], metrics["fp32_batch8"])
    amp_max = max(map(abs, amp_delta.values()))
    batch_max = max(map(abs, batch_delta.values()))
    amp_pass = amp_max <= args.metric_tolerance
    batch_pass = batch_max <= args.metric_tolerance
    overall_pass = all(protocol_pass.values()) and hash_pass and amp_pass and batch_pass
    payload = {
        "stage": "Refine V3.1.1 low-gain three-path reproduction audit",
        "residual_scale": args.residual_scale,
        "metric_tolerance": args.metric_tolerance,
        "protocol_pass": protocol_pass,
        "hash_pass": hash_pass,
        "amp_batch8_minus_fp32_batch8": amp_delta,
        "amp_max_abs_metric_delta": amp_max,
        "amp_reproduction_pass": amp_pass,
        "fp32_batch1_minus_fp32_batch8": batch_delta,
        "batch_max_abs_metric_delta": batch_max,
        "batch_reproduction_pass": batch_pass,
        "overall_pass": overall_pass,
        "checkpoint_sha256": next(iter(checkpoint_hashes)) if len(checkpoint_hashes) == 1 else None,
        "ca_sha256": next(iter(ca_hashes)) if len(ca_hashes) == 1 else None,
        "test_used": False,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "reproduction_audit.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Refine V3.1.1 低指标三路径复现审计",
        "",
        f"- 总体通过：**{overall_pass}**",
        f"- 固定残差强度：{args.residual_scale}",
        f"- 三条路径协议通过：{all(protocol_pass.values())}",
        f"- CA/Refine 哈希一致：{hash_pass}",
        f"- AMP batch=8 与 FP32 batch=8 最大指标差：{amp_max:.8f}",
        f"- FP32 batch=1 与 batch=8 最大指标差：{batch_max:.8f}",
        "- 三条路径均为原验证集；未读取 test；未修改权重。",
        "",
    ]
    (output_dir / "reproduction_audit.md").write_text("\n".join(report), encoding="utf-8")
    print(json_path)
    if args.strict and not overall_pass:
        raise RuntimeError("low-gain reproduction audit failed; inspect reproduction_audit.json")


if __name__ == "__main__":
    main()
