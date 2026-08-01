"""Select a Refine checkpoint from a paired coarse/normal validation curve.

The selection rule is intentionally deterministic and paper-auditable:

1. require one coarse and one normal row per epoch;
2. require the frozen coarse metrics to remain within a configured tolerance;
3. filter checkpoints by predeclared mAP50-95/AP75/AP90/AP95 deltas;
4. select the largest mAP50-95 gain, breaking exact ties by the earlier epoch.

The script never reads the test split and never searches residual alpha values.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CONTROL_METRICS = (
    "precision",
    "recall",
    "map50",
    "map50_95",
    "ap55",
    "ap60",
    "ap65",
    "ap70",
    "ap75",
    "ap80",
    "ap85",
    "ap90",
    "ap95",
)
SELECTION_METRICS = ("map50_95", "ap75", "ap90", "ap95")
PAIR_METADATA = (
    "weights",
    "data",
    "imgsz",
    "split",
    "profile",
    "refine_version",
    "refine_experiment",
    "refine_delta_max",
    "refine_target_limit",
)


@dataclass(frozen=True)
class SelectionThresholds:
    """Predeclared validation-delta requirements."""

    min_map50_95: float = 0.002
    min_ap75: float = 0.0
    min_ap90: float = -0.002
    min_ap95: float = -0.001


def read_curve(path: Path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV without altering the source file."""
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"验证曲线为空: {path}")
    return rows


def parse_float(row: dict[str, str], column: str) -> float:
    """Parse one required finite metric."""
    try:
        value = float(row[column])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"列 {column!r} 缺失或不是数值: {row.get(column)!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"列 {column!r} 包含非有限值: {value}")
    return value


def parse_epoch(row: dict[str, str]) -> int:
    """Parse a positive integer plotted epoch."""
    value = parse_float(row, "epoch")
    epoch = int(value)
    if value != epoch or epoch <= 0:
        raise ValueError(f"epoch 必须是正整数，收到 {value}")
    return epoch


def require_expected_metadata(
    rows: list[dict[str, str]],
    *,
    expect_profile: str | None,
    expect_delta_max: float | None,
    expect_target_limit: float | None,
    tolerance: float = 1e-12,
) -> None:
    """Reject a curve generated from the wrong checkpoint semantics."""
    for row in rows:
        if expect_profile is not None and row.get("refine_experiment") != expect_profile:
            raise ValueError(
                "Refine profile 不符合预期: "
                f"expected={expect_profile!r}, actual={row.get('refine_experiment')!r}"
            )
        for column, expected in (
            ("refine_delta_max", expect_delta_max),
            ("refine_target_limit", expect_target_limit),
        ):
            if expected is None:
                continue
            actual = parse_float(row, column)
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
                raise ValueError(f"{column} 不符合预期: expected={expected}, actual={actual}")


def require_validation_context(rows: list[dict[str, str]]) -> None:
    """Prevent checkpoint selection on mixed settings or on the test split."""
    for column in (item for item in PAIR_METADATA if item != "weights"):
        values = {row.get(column) for row in rows}
        if None in values or len(values) != 1:
            raise ValueError(f"曲线跨行元数据不一致或缺失: {column}={sorted(map(str, values))}")
    if rows[0]["split"] != "val":
        raise ValueError("checkpoint 只能在 validation split 上选择，禁止使用 test split")
    if rows[0]["profile"] != "curve":
        raise ValueError(f"checkpoint 选择要求 profile=curve，收到 {rows[0]['profile']!r}")
    if parse_float(rows[0], "imgsz") != 640.0:
        raise ValueError(f"创新点一 checkpoint 选择固定 imgsz=640，收到 {rows[0]['imgsz']!r}")
    if any(row.get("fresh_load", "").lower() not in {"true", "1"} for row in rows):
        raise ValueError("checkpoint 选择要求 fresh_load=True，以排除跨 variant 状态污染")


def pair_epochs(rows: list[dict[str, str]]) -> list[tuple[int, dict[str, str], dict[str, str]]]:
    """Build and validate the coarse/normal pair for every epoch."""
    grouped: dict[int, dict[str, dict[str, str]]] = {}
    for row in rows:
        epoch = parse_epoch(row)
        variant = row.get("variant")
        if variant not in {"coarse", "normal"}:
            raise ValueError(f"只允许 coarse/normal curve，收到 variant={variant!r}")
        variants = grouped.setdefault(epoch, {})
        if variant in variants:
            raise ValueError(f"epoch={epoch} 出现重复 variant={variant}")
        variants[variant] = row

    pairs = []
    for epoch, variants in sorted(grouped.items()):
        missing = {"coarse", "normal"} - variants.keys()
        if missing:
            raise ValueError(f"epoch={epoch} 缺少配对行: {sorted(missing)}")
        coarse, normal = variants["coarse"], variants["normal"]
        mismatches = [column for column in PAIR_METADATA if coarse.get(column) != normal.get(column)]
        if mismatches:
            raise ValueError(f"epoch={epoch} coarse/normal 元数据不一致: {mismatches}")
        pairs.append((epoch, coarse, normal))
    return pairs


def check_coarse_identity(
    pairs: list[tuple[int, dict[str, str], dict[str, str]]],
    tolerance: float,
) -> dict[str, float]:
    """Require the frozen CA path to remain invariant across checkpoints."""
    if tolerance < 0:
        raise ValueError("coarse tolerance 不能为负数")
    ranges = {}
    for metric in CONTROL_METRICS:
        values = [parse_float(coarse, metric) for _, coarse, _ in pairs]
        metric_range = max(values) - min(values)
        ranges[metric] = metric_range
        if metric_range > tolerance:
            raise ValueError(
                f"coarse-only 跨 checkpoint 漂移: metric={metric}, range={metric_range:.9g}, "
                f"tolerance={tolerance:.9g}"
            )
    return ranges


def analyze_curve(
    rows: list[dict[str, str]],
    *,
    thresholds: SelectionThresholds = SelectionThresholds(),
    coarse_tolerance: float = 5e-4,
    expect_profile: str | None = None,
    expect_delta_max: float | None = None,
    expect_target_limit: float | None = None,
) -> dict[str, Any]:
    """Validate the curve, calculate paired deltas, and select one checkpoint."""
    require_expected_metadata(
        rows,
        expect_profile=expect_profile,
        expect_delta_max=expect_delta_max,
        expect_target_limit=expect_target_limit,
    )
    require_validation_context(rows)
    pairs = pair_epochs(rows)
    coarse_ranges = check_coarse_identity(pairs, coarse_tolerance)

    epoch_rows = []
    for epoch, coarse, normal in pairs:
        deltas = {
            metric: parse_float(normal, metric) - parse_float(coarse, metric)
            for metric in SELECTION_METRICS
        }
        passes = (
            deltas["map50_95"] >= thresholds.min_map50_95
            and deltas["ap75"] >= thresholds.min_ap75
            and deltas["ap90"] >= thresholds.min_ap90
            and deltas["ap95"] >= thresholds.min_ap95
        )
        epoch_rows.append(
            {
                "epoch": epoch,
                "weights": normal["weights"],
                "passes": passes,
                "normal": {metric: parse_float(normal, metric) for metric in SELECTION_METRICS},
                "coarse": {metric: parse_float(coarse, metric) for metric in SELECTION_METRICS},
                "delta": deltas,
            }
        )

    candidates = [row for row in epoch_rows if row["passes"]]
    if not candidates:
        raise ValueError("没有 checkpoint 同时满足预声明的四项验证门槛")
    selected = sorted(candidates, key=lambda row: (-row["delta"]["map50_95"], row["epoch"]))[0]
    metadata = {column: rows[0].get(column) for column in PAIR_METADATA if column != "weights"}
    return {
        "status": "PASS",
        "thresholds": asdict(thresholds),
        "coarse_tolerance": coarse_tolerance,
        "coarse_metric_ranges": coarse_ranges,
        "metadata": metadata,
        "epoch_count": len(epoch_rows),
        "candidate_count": len(candidates),
        "selected": selected,
        "epochs": epoch_rows,
    }


def write_json(path: Path, result: dict[str, Any]) -> None:
    """Write an auditable selection record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_env(path: Path, result: dict[str, Any]) -> None:
    """Write shell-safe variables for subsequent validation commands."""
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = result["selected"]
    content = (
        f"SELECTED_EPOCH={selected['epoch']}\n"
        f"SELECTED_V22={shlex.quote(selected['weights'])}\n"
    )
    path.write_text(content, encoding="utf-8")


def print_summary(result: dict[str, Any]) -> None:
    """Print the decision and enough evidence for terminal review."""
    selected = result["selected"]
    delta = selected["delta"]
    print("Refine checkpoint selection: PASS")
    print(f"epochs={result['epoch_count']}, eligible={result['candidate_count']}")
    print(f"selected_epoch={selected['epoch']}")
    print(f"selected_weights={selected['weights']}")
    print(
        "delta: "
        f"mAP50-95={delta['map50_95']:+.6f}, "
        f"AP75={delta['ap75']:+.6f}, "
        f"AP90={delta['ap90']:+.6f}, "
        f"AP95={delta['ap95']:+.6f}"
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve-csv", type=Path, required=True)
    parser.add_argument("--min-map50-95", type=float, default=0.002)
    parser.add_argument("--min-ap75", type=float, default=0.0)
    parser.add_argument("--min-ap90", type=float, default=-0.002)
    parser.add_argument("--min-ap95", type=float, default=-0.001)
    parser.add_argument("--coarse-tolerance", type=float, default=5e-4)
    parser.add_argument("--expect-profile")
    parser.add_argument("--expect-delta-max", type=float)
    parser.add_argument("--expect-target-limit", type=float)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-env", type=Path)
    args = parser.parse_args()

    thresholds = SelectionThresholds(
        min_map50_95=args.min_map50_95,
        min_ap75=args.min_ap75,
        min_ap90=args.min_ap90,
        min_ap95=args.min_ap95,
    )
    try:
        result = analyze_curve(
            read_curve(args.curve_csv),
            thresholds=thresholds,
            coarse_tolerance=args.coarse_tolerance,
            expect_profile=args.expect_profile,
            expect_delta_max=args.expect_delta_max,
            expect_target_limit=args.expect_target_limit,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.exit(2, f"Refine checkpoint selection: FAIL\n{error}\n")

    if args.output_json is not None:
        write_json(args.output_json, result)
    if args.output_env is not None:
        write_env(args.output_env, result)
    print_summary(result)


if __name__ == "__main__":
    main()
