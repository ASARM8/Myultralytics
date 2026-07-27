"""Generate Fig. 3: DFL overflow rate by long-edge bin for Baseline and CA."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from myscripts.paper_visuals.common import (
    COLORS,
    configure_matplotlib,
    ensure_output_dir,
    h1_records,
    load_h1h2,
    natural_bin_key,
    require_columns,
    require_nonempty,
    save_figure,
    wilson_interval,
)


configure_matplotlib()
import matplotlib.pyplot as plt


def merge_under_100(frame: pd.DataFrame) -> pd.DataFrame:
    """Merge bins below 100 px using exact counts; this is valid for rates, not quantiles."""
    rows = []
    for method, group in frame.groupby("method", sort=False):
        small = group[group["long_bin"].isin(["<50", "50-100", "<100"])]
        other = group[~group.index.isin(small.index)]
        if not small.empty:
            total = int(small["total"].sum())
            overflow = int(small["overflow"].sum())
            rows.append(
                {
                    "method": method,
                    "long_bin": "<100",
                    "total": total,
                    "overflow": overflow,
                    "overflow_rate": 100.0 * overflow / total if total else np.nan,
                }
            )
        rows.extend(other.to_dict("records"))
    return pd.DataFrame(rows)


def load_frame(args) -> pd.DataFrame:
    """Load either machine-readable H1/H2 JSON files or a tidy CSV."""
    if args.csv:
        frame = pd.read_csv(args.csv)
        require_nonempty(frame, args.csv)
        require_columns(frame, ["method", "long_bin"], args.csv)
        if "overflow_rate" not in frame.columns:
            require_columns(frame, ["total", "overflow"], args.csv)
            frame["overflow_rate"] = 100.0 * frame["overflow"] / frame["total"]
    else:
        if not args.baseline_json or not args.ca_json:
            raise ValueError("请同时提供 --baseline-json 与 --ca-json，或改用 --csv")
        records = h1_records(load_h1h2(args.baseline_json), args.baseline_label)
        records.extend(h1_records(load_h1h2(args.ca_json), args.ca_label))
        frame = pd.DataFrame(records)
    require_columns(frame, ["method", "long_bin", "overflow_rate"], args.csv or "H1/H2 JSON")
    if args.merge_under_100:
        require_columns(frame, ["total", "overflow"], args.csv or "H1/H2 JSON")
        frame = merge_under_100(frame)
    return frame


def build_figure(frame: pd.DataFrame):
    """Build a colorblind-safe grouped bar chart with Wilson intervals when counts exist."""
    methods = list(dict.fromkeys(frame["method"].astype(str)))
    if len(methods) != 2:
        raise ValueError(f"图3需要恰好两个方法，当前为: {methods}")
    bins = sorted(frame["long_bin"].astype(str).unique(), key=natural_bin_key)
    x = np.arange(len(bins), dtype=float)
    width = 0.36
    palette = [COLORS["baseline"], COLORS["ca"]]
    hatches = ["////", ""]

    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    for index, (method, color, hatch) in enumerate(zip(methods, palette, hatches)):
        subset = frame[frame["method"].astype(str) == method].set_index("long_bin")
        values = np.array(
            [float(subset.loc[label, "overflow_rate"]) if label in subset.index else np.nan for label in bins]
        )
        positions = x + (index - 0.5) * width
        yerr = None
        if {"total", "overflow"}.issubset(subset.columns):
            lower, upper = [], []
            for label, value in zip(bins, values):
                if label not in subset.index or np.isnan(value):
                    lower.append(np.nan)
                    upper.append(np.nan)
                    continue
                lo, hi = wilson_interval(int(subset.loc[label, "overflow"]), int(subset.loc[label, "total"]))
                lower.append(value - lo)
                upper.append(hi - value)
            yerr = np.array([lower, upper])
        bars = ax.bar(
            positions,
            values,
            width,
            label=method,
            color=color,
            edgecolor=COLORS["ink"],
            linewidth=0.55,
            hatch=hatch,
            yerr=yerr,
            capsize=2.3 if yerr is not None else 0,
            error_kw={"elinewidth": 0.75, "capthick": 0.75},
        )
        upper_errors = yerr[1] if yerr is not None else np.zeros_like(values)
        for bar, value, upper_error in zip(bars, values, upper_errors):
            if np.isnan(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.0 if np.isnan(upper_error) else upper_error)
                + max(0.45, np.nanmax(frame["overflow_rate"]) * 0.015),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.0,
                color=COLORS["ink"],
            )

    ax.set_xticks(x, bins)
    ax.set_xlabel("GT 长边分桶 / px")
    ax.set_ylabel("DFL 溢出率 / %")
    ax.grid(axis="y", color=COLORS["light_gray"], linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.margins(x=0.025)
    ymax = np.nanmax(frame["overflow_rate"].to_numpy(dtype=float))
    ax.set_ylim(0, max(5.0, ymax * 1.20 + 1.0))
    fig.subplots_adjust(left=0.105, right=0.995, top=0.98, bottom=0.19)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv", type=Path, help="长表格式 CSV")
    source.add_argument("--baseline-json", type=Path, help="Baseline 的 h1h2_data.json")
    parser.add_argument("--ca-json", type=Path, help="CA 的 h1h2_data.json")
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument("--ca-label", default="CA")
    parser.add_argument("--merge-under-100", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    frame = load_frame(args)
    fig = build_figure(frame)
    paths = save_figure(fig, ensure_output_dir(args.output_dir), "fig3_dfl_overflow_by_length")
    plt.close(fig)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
