"""Build traceable CSV/Markdown tables for the innovation-one paper without inventing missing values."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from myscripts.paper_visuals.common import h1_records, h2_records, load_h1h2


PAPER_BINS = ["<100", "100-200", "200-300", "300-500", ">500"]
LEVELS = ["P3", "P4", "P5"]


def merge_under_100_h1(frame: pd.DataFrame) -> pd.DataFrame:
    """Merge <50 and 50-100 counts exactly; leave nonlinear P95 blank rather than approximate it."""
    labels = set(frame["long_bin"].astype(str))
    if "<100" in labels:
        return frame
    children = frame[frame["long_bin"].isin(["<50", "50-100"])]
    if len(children) != 2:
        return frame
    total = int(children["total"].sum())
    overflow = int(children["overflow"].sum())
    weighted_mean = float(np.average(children["dreq_mean"], weights=children["total"])) if total else np.nan
    merged = {
        "method": children.iloc[0]["method"],
        "long_bin": "<100",
        "total": total,
        "overflow": overflow,
        "overflow_rate": 100 * overflow / total if total else np.nan,
        "dreq_mean": weighted_mean,
        "dreq_p95": np.nan,
    }
    warnings.warn("<100 的计数、溢出率和均值已精确合并；P95 无法由两个子区间 P95 反推，已留空。")
    return pd.concat(
        [pd.DataFrame([merged]), frame[~frame["long_bin"].isin(["<50", "50-100"])]], ignore_index=True
    )


def merge_under_100_h2(frame: pd.DataFrame) -> pd.DataFrame:
    """Merge H2 through total-weighted level percentages."""
    labels = set(frame["long_bin"].astype(str))
    if "<100" in labels:
        return frame
    children = frame[frame["long_bin"].isin(["<50", "50-100"])]
    if len(children) != 2:
        return frame
    total = int(children["total"].sum())
    merged = {"method": children.iloc[0]["method"], "long_bin": "<100", "total": total}
    for level in LEVELS:
        count = (children[level] * children["total"] / 100).sum()
        merged[level] = 100 * count / total if total else 0.0
    return pd.concat(
        [pd.DataFrame([merged]), frame[~frame["long_bin"].isin(["<50", "50-100"])]], ignore_index=True
    )


def order_bins(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort table rows in the manuscript's long-edge order."""
    order = {label: index for index, label in enumerate(PAPER_BINS)}
    result = frame.copy()
    result["_order"] = result["long_bin"].map(order).fillna(999)
    return result.sort_values(["_order", "long_bin"]).drop(columns="_order").reset_index(drop=True)


def table01(baseline: dict) -> pd.DataFrame:
    """Original-assignment DFL reachability diagnostics."""
    frame = order_bins(merge_under_100_h1(pd.DataFrame(h1_records(baseline, "Baseline"))))
    return frame[["long_bin", "total", "overflow_rate", "dreq_mean", "dreq_p95"]].rename(
        columns={
            "long_bin": "long_edge_bin_px",
            "total": "positive_samples",
            "overflow_rate": "overflow_rate_percent",
            "dreq_mean": "d_req_mean_grid",
            "dreq_p95": "d_req_p95_grid",
        }
    )


def table02(baseline: dict) -> pd.DataFrame:
    """Original positive-sample FPN distribution."""
    frame = order_bins(merge_under_100_h2(pd.DataFrame(h2_records(baseline, "Baseline"))))
    frame = frame[frame["long_bin"] != "<100"]
    return frame[["long_bin", *LEVELS]].rename(
        columns={"long_bin": "long_edge_bin_px", **{x: f"{x}_percent" for x in LEVELS}}
    )


def table03() -> pd.DataFrame:
    """Theoretical DFL reachability under reg_max=16/32."""
    rows = []
    for stride in (8, 16, 32):
        rows.append(
            {
                "feature_level": f"P{int(math.log2(stride))}",
                "stride": stride,
                "reg_max_16_dmax_px": stride * 15,
                "reg_max_32_dmax_px": stride * 31,
            }
        )
    return pd.DataFrame(rows)


def select_dataset_summary(path: Path, name: str | None) -> dict:
    """Select one dataset entry from target_coverage_summary.json."""
    summaries = json.loads(path.read_text(encoding="utf-8")).get("summaries", [])
    if not summaries:
        raise ValueError(f"No summaries found in {path}")
    if name is None:
        if len(summaries) > 1:
            warnings.warn(f"{path} 包含多个数据集，默认使用第一项；可用 --dataset-name 指定。")
        return summaries[0]
    for summary in summaries:
        if summary.get("name") == name:
            return summary
    raise KeyError(f"Dataset name not found: {name}")


def table04(dataset: dict | None) -> pd.DataFrame:
    """Dataset/preprocessing table; unavailable source fields remain blank."""
    splits = dataset.get("splits", {}) if dataset else {}

    def split_count(key: str):
        value = splits.get(key, "") if isinstance(splits, dict) else ""
        return value.get("total_images", value.get("images", "")) if isinstance(value, dict) else value

    split_values = list(splits.values()) if isinstance(splits, dict) else []
    sliced = sum(
        int(value.get("total_images", value.get("images", 0))) if isinstance(value, dict) else int(value or 0)
        for value in split_values
    )
    split_text = "/".join(str(split_count(key)) for key in ("train", "val", "test")) if dataset else ""
    return pd.DataFrame(
        [
            ["原始图像数", ""],
            ["切片后图像数", sliced if dataset else ""],
            ["训练/验证/测试图像数", split_text],
            ["电线实例数", dataset.get("object_count", "") if dataset else ""],
            ["输入尺寸", ""],
            ["类别数", len(dataset.get("class_counts", {})) if dataset and dataset.get("class_counts") else ""],
        ],
        columns=["item", "value"],
    )


def value_for_bin(frame: pd.DataFrame, label: str, column: str) -> float:
    """Return one bin value or NaN."""
    selected = frame.loc[frame["long_bin"] == label, column]
    return float(selected.iloc[0]) if len(selected) else np.nan


def table07(baseline: dict, ca: dict, extra: Path | None) -> pd.DataFrame:
    """CA mechanism diagnostics from H1/H2 plus optional dataset-level extra values."""
    frames_h1, frames_h2 = {}, {}
    for label, data in (("Baseline", baseline), ("CA", ca)):
        frames_h1[label] = order_bins(merge_under_100_h1(pd.DataFrame(h1_records(data, label))))
        frames_h2[label] = order_bins(merge_under_100_h2(pd.DataFrame(h2_records(data, label))))

    extra_values: dict[tuple[str, str], float | str] = {}
    if extra and extra.exists():
        for row in pd.read_csv(extra).to_dict("records"):
            extra_values[(str(row.get("metric")), str(row.get("method")))] = row.get("value", "")

    values: dict[str, dict[str, float | str]] = {"Baseline": {}, "CA": {}}
    for method in ("Baseline", "CA"):
        h1, h2 = frames_h1[method], frames_h2[method]
        total, overflow = int(h1["total"].sum()), int(h1["overflow"].sum())
        values[method] = {
            "all_positive_overflow_rate_percent": 100 * overflow / total if total else np.nan,
            ">500_overflow_rate_percent": value_for_bin(h1, ">500", "overflow_rate"),
            ">500_P3_percent": value_for_bin(h2, ">500", "P3"),
            ">500_P4_percent": value_for_bin(h2, ">500", "P4"),
            ">500_P5_percent": value_for_bin(h2, ">500", "P5"),
            "fallback_gt_ratio_percent": extra_values.get(("fallback_gt_ratio_percent", method), ""),
        }

    labels = [
        ("all_positive_overflow_rate_percent", "全体正样本 DFL 溢出率/%"),
        (">500_overflow_rate_percent", "长边 >500 px 溢出率/%"),
        (">500_P3_percent", "极端目标 P3 占比/%"),
        (">500_P4_percent", "极端目标 P4 占比/%"),
        (">500_P5_percent", "极端目标 P5 占比/%"),
        ("fallback_gt_ratio_percent", "无候选回退 GT 比例/%"),
    ]
    rows = []
    for key, label in labels:
        baseline_value, ca_value = values["Baseline"][key], values["CA"][key]
        change = ""
        if baseline_value != "" and ca_value != "":
            change = float(ca_value) - float(baseline_value)
        rows.append([label, baseline_value, ca_value, change])
    return pd.DataFrame(rows, columns=["diagnostic", "Baseline", "CA", "change_ca_minus_baseline"])


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    """Write a Word-friendly CSV and a review-friendly Markdown file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    path.with_suffix(".md").write_text(frame.to_markdown(index=False), encoding="utf-8")


def copy_manual_tables(manual_dir: Path | None, output_dir: Path) -> None:
    """Copy nonempty manual-input tables into the output bundle."""
    if manual_dir is None:
        return
    for number in (5, 6, 8, 9, 10, 11):
        matches = sorted(manual_dir.glob(f"table{number:02d}_*.csv"))
        if not matches:
            continue
        frame = pd.read_csv(matches[0])
        if frame.empty:
            warnings.warn(f"{matches[0]} 只有表头，未输出为正式表格。")
            continue
        write_frame(frame, output_dir / matches[0].name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-h1h2", type=Path)
    parser.add_argument("--ca-h1h2", type=Path)
    parser.add_argument("--dataset-summary", type=Path)
    parser.add_argument("--dataset-name")
    parser.add_argument("--table07-extra", type=Path)
    parser.add_argument("--manual-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_frame(table03(), args.output_dir / "table03_theoretical_dmax.csv")
    dataset = select_dataset_summary(args.dataset_summary, args.dataset_name) if args.dataset_summary else None
    write_frame(table04(dataset), args.output_dir / "table04_dataset_preprocessing.csv")

    baseline = load_h1h2(args.baseline_h1h2) if args.baseline_h1h2 else None
    ca = load_h1h2(args.ca_h1h2) if args.ca_h1h2 else None
    if baseline:
        write_frame(table01(baseline), args.output_dir / "table01_original_dfl_diagnostics.csv")
        write_frame(table02(baseline), args.output_dir / "table02_original_level_distribution.csv")
    if baseline and ca:
        write_frame(table07(baseline, ca, args.table07_extra), args.output_dir / "table07_ca_mechanism_diagnostics.csv")
    copy_manual_tables(args.manual_dir, args.output_dir)
    print(args.output_dir)


if __name__ == "__main__":
    main()
