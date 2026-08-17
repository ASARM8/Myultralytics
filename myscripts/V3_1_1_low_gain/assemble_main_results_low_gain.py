"""Assemble Baseline, CA, and conservative Refine metrics into one audited CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from myscripts.V3_1_1_low_gain.config import DEFAULT_RESIDUAL_SCALE


METRICS = ("precision", "recall", "map50", "map50_95") + tuple(f"ap{x}" for x in range(50, 100, 5))


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ca-csv", required=True)
    parser.add_argument("--low-gain-metrics-csv", required=True)
    parser.add_argument("--residual-scale", type=float, default=DEFAULT_RESIDUAL_SCALE)
    parser.add_argument("--ca-tolerance", type=float, default=1e-3)
    parser.add_argument("--output-csv", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if abs(args.residual_scale - DEFAULT_RESIDUAL_SCALE) > 1e-12:
        parser.error(f"formal low-gain result assembly is locked to residual-scale={DEFAULT_RESIDUAL_SCALE}")
    baseline_ca_path = Path(args.baseline_ca_csv)
    low_path = Path(args.low_gain_metrics_csv)
    baseline_ca = _read(baseline_ca_path)
    low_rows = _read(low_path)
    baseline = next((row for row in baseline_ca if row.get("method") == "Baseline"), None)
    ca = next((row for row in baseline_ca if row.get("method") == "CA"), None)
    coarse = next((row for row in low_rows if row.get("variant") == "coarse"), None)
    refined = next((row for row in low_rows if row.get("variant") == "refined"), None)
    if None in (baseline, ca, coarse, refined):
        raise RuntimeError("Baseline, CA, coarse, or refined row is missing")
    for row, label in ((baseline, "Baseline"), (ca, "CA"), (coarse, "coarse"), (refined, "refined")):
        missing = [metric for metric in METRICS if metric not in row]
        if missing:
            raise RuntimeError(f"{label} row misses metrics: {missing}")
    ca_differences = {metric: float(coarse[metric]) - float(ca[metric]) for metric in METRICS}
    max_ca_difference = max(map(abs, ca_differences.values()))
    if max_ca_difference > args.ca_tolerance:
        raise RuntimeError(f"CA validator mismatch exceeds tolerance: {max_ca_difference:.8f}")

    output_rows = []
    for label, source, row in (
        ("Baseline", str(baseline_ca_path), baseline),
        ("CA", str(baseline_ca_path), ca),
        ("CA + Refine", str(low_path), refined),
    ):
        output_rows.append(
            {
                "method": label,
                **{metric: float(row[metric]) for metric in METRICS},
                "residual_scale": args.residual_scale if label == "CA + Refine" else "",
                "source": source,
            }
        )
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    audit = {
        "baseline_ca_csv": str(baseline_ca_path),
        "low_gain_metrics_csv": str(low_path),
        "residual_scale": args.residual_scale,
        "ca_max_abs_metric_difference_between_validators": max_ca_difference,
        "ca_tolerance": args.ca_tolerance,
        "ca_consistency_pass": True,
        "complete_ap50_to_ap95": True,
        "test_used": False,
    }
    output_path.with_suffix(".audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output_path)


if __name__ == "__main__":
    main()
