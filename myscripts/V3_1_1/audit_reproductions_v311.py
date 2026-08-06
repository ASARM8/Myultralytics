"""Audit the locked three-run Refine V3.1.1 reproduction protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


METRICS = ("map50_95", "ap75", "ap90", "ap95")
VARIANTS = ("coarse", "refined")
ARCHITECTURE = "OBBProposalRefinerV311"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit V3.1.1 training and three independent val reproductions.")
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--amp-batch8-dir", required=True)
    parser.add_argument("--fp32-batch8-dir", required=True)
    parser.add_argument("--fp32-batch1-dir", required=True)
    parser.add_argument("--metric-tolerance", type=float, default=5e-4)
    parser.add_argument("--output-dir", required=True)
    return parser


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_metrics(path: Path) -> dict[str, dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for variant in VARIANTS:
        row = next((item for item in rows if item.get("variant") == variant), None)
        if row is None:
            raise RuntimeError(f"{variant} row missing from {path}")
        result[variant] = {metric: float(row[metric]) for metric in METRICS}
    return result


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_differences(left: dict, right: dict) -> dict[str, float]:
    return {
        f"{variant}.{metric}": left[variant][metric] - right[variant][metric]
        for variant in VARIANTS
        for metric in METRICS
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.metric_tolerance <= 0:
        parser.error("--metric-tolerance must be positive")

    training_dir = Path(args.training_dir)
    run_dirs = {
        "amp_batch8": Path(args.amp_batch8_dir),
        "fp32_batch8": Path(args.fp32_batch8_dir),
        "fp32_batch1": Path(args.fp32_batch1_dir),
    }
    training_acceptance = read_json(training_dir / "acceptance.json")
    training_metrics = read_metrics(training_dir / "val_metrics.csv")
    training_checkpoint_sha256 = sha256_file(training_dir / "checkpoints" / "best.pt")
    audits = {name: read_json(path / "validation_audit.json") for name, path in run_dirs.items()}
    metrics = {name: read_metrics(path / "val_metrics.csv") for name, path in run_dirs.items()}

    expected_modes = {
        "amp_batch8": (True, 8),
        "fp32_batch8": (False, 8),
        "fp32_batch1": (False, 1),
    }
    protocol_pass = {
        name: audit.get("architecture") == ARCHITECTURE
        and bool(audit.get("amp")) == amp
        and int(audit.get("batch", -1)) == batch
        and audit.get("split") == "val"
        and not bool(audit.get("test_used"))
        for name, audit in audits.items()
        for amp, batch in (expected_modes[name],)
    }
    hashes = {
        "checkpoint": {audit.get("checkpoint_sha256") for audit in audits.values()},
        "ca": {audit.get("ca_sha256") for audit in audits.values()},
    }
    hash_pass = (
        hashes["checkpoint"] == {training_checkpoint_sha256}
        and hashes["ca"] == {training_acceptance.get("ca_hash_before")}
    )
    training_protocol_pass = (
        training_acceptance.get("architecture") == ARCHITECTURE
        and training_acceptance.get("experiment") == "geometry_only"
        and training_acceptance.get("target_transform") == "smooth_compression"
        and training_acceptance.get("proposal_policy") == "all"
        and not bool(training_acceptance.get("rerun_nms"))
        and not bool(training_acceptance.get("test_used"))
    )

    amp_vs_training = metric_differences(metrics["amp_batch8"], training_metrics)
    fp32_batch1_vs_batch8 = metric_differences(metrics["fp32_batch1"], metrics["fp32_batch8"])
    amp_reproduction_pass = max(map(abs, amp_vs_training.values())) <= args.metric_tolerance
    fp32_batch_invariance_pass = max(map(abs, fp32_batch1_vs_batch8.values())) <= args.metric_tolerance
    individual_pass = {name: bool(audit.get("reproduction_pass")) for name, audit in audits.items()}
    overall_pass = all(
        (
            bool(training_acceptance.get("screening_pass")),
            training_protocol_pass,
            all(protocol_pass.values()),
            hash_pass,
            all(individual_pass.values()),
            amp_reproduction_pass,
            fp32_batch_invariance_pass,
        )
    )

    payload = {
        "stage": "Refine V3.1.1 locked reproduction audit",
        "architecture": ARCHITECTURE,
        "metric_tolerance": args.metric_tolerance,
        "training_screening_pass": bool(training_acceptance.get("screening_pass")),
        "training_protocol_pass": training_protocol_pass,
        "protocol_pass": protocol_pass,
        "hash_pass": hash_pass,
        "individual_reproduction_pass": individual_pass,
        "amp_vs_training_metric_delta": amp_vs_training,
        "amp_reproduction_pass": amp_reproduction_pass,
        "fp32_batch1_minus_batch8_metric_delta": fp32_batch1_vs_batch8,
        "fp32_batch_invariance_pass": fp32_batch_invariance_pass,
        "overall_pass": overall_pass,
        "checkpoint_sha256": next(iter(hashes["checkpoint"])) if len(hashes["checkpoint"]) == 1 else None,
        "ca_sha256": next(iter(hashes["ca"])) if len(hashes["ca"]) == 1 else None,
        "test_used": False,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reproduction_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# Refine V3.1.1 独立复现审计",
        "",
        f"- 总体通过：**{overall_pass}**",
        f"- 训练筛选通过：{payload['training_screening_pass']}",
        f"- 训练协议正确：{training_protocol_pass}",
        f"- 三次运行协议正确：{all(protocol_pass.values())}",
        f"- CA/Refine 权重哈希一致：{hash_pass}",
        f"- 三次单项验收通过：{all(individual_pass.values())}",
        f"- AMP batch=8 对训练结果最大差值：{max(map(abs, amp_vs_training.values())):.8f}",
        f"- FP32 batch=1/8 最大差值：{max(map(abs, fp32_batch1_vs_batch8.values())):.8f}",
        "- 数据划分：原 val；test 未使用。",
        "",
    ]
    (output_dir / "reproduction_audit.md").write_text("\n".join(report), encoding="utf-8")
    print(output_dir / "reproduction_audit.json")
    if not overall_pass:
        raise RuntimeError("Refine V3.1.1 reproduction audit failed; inspect reproduction_audit.json")


if __name__ == "__main__":
    main()
