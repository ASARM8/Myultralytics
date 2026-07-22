"""Generate Fig. 5: P3/P4/P5 positive-sample distribution before and after CA."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from myscripts.paper_visuals.common import (
    COLORS,
    configure_matplotlib,
    ensure_output_dir,
    h2_records,
    load_h1h2,
    natural_bin_key,
    require_columns,
    require_nonempty,
    save_figure,
)


configure_matplotlib()
import matplotlib.pyplot as plt


def load_frame(args) -> pd.DataFrame:
    """Load H2 distributions from diagnostic JSON or a tidy CSV."""
    if args.csv:
        frame = pd.read_csv(args.csv)
        require_nonempty(frame, args.csv)
    else:
        if not args.baseline_json or not args.ca_json:
            raise ValueError("请同时提供 --baseline-json 与 --ca-json，或改用 --csv")
        records = h2_records(load_h1h2(args.baseline_json), args.baseline_label)
        records.extend(h2_records(load_h1h2(args.ca_json), args.ca_label))
        frame = pd.DataFrame(records)
    require_columns(frame, ["method", "long_bin", "P3", "P4", "P5"], args.csv or "H1/H2 JSON")
    level_sum = frame[["P3", "P4", "P5"]].sum(axis=1)
    count_rows = level_sum.gt(0) & ~np.isclose(level_sum, 100.0, atol=0.5)
    if count_rows.any():
        frame.loc[count_rows, ["P3", "P4", "P5"]] = (
            frame.loc[count_rows, ["P3", "P4", "P5"]].div(level_sum[count_rows], axis=0) * 100.0
        )
    if args.exclude_under_100:
        frame = frame[~frame["long_bin"].isin(["<50", "50-100", "<100"])].copy()
    require_nonempty(frame, args.csv or "H1/H2 JSON")
    return frame


def build_figure(frame: pd.DataFrame):
    """Build matched 100% stacked-bar panels with identical bin ordering."""
    methods = list(dict.fromkeys(frame["method"].astype(str)))
    if len(methods) != 2:
        raise ValueError(f"图5需要恰好两个方法，当前为: {methods}")
    bins = sorted(frame["long_bin"].astype(str).unique(), key=natural_bin_key)
    layers = ["P3", "P4", "P5"]
    colors = [COLORS["p3"], COLORS["p4"], COLORS["p5"]]
    hatches = ["////", "....", ""]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.15), sharey=True)
    for panel, (ax, method) in enumerate(zip(axes, methods)):
        subset = frame[frame["method"].astype(str) == method].set_index("long_bin")
        x = np.arange(len(bins))
        bottom = np.zeros(len(bins), dtype=float)
        for layer, color, hatch in zip(layers, colors, hatches):
            values = np.array([float(subset.loc[label, layer]) if label in subset.index else 0.0 for label in bins])
            bars = ax.bar(
                x,
                values,
                width=0.66,
                bottom=bottom,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                hatch=hatch,
                label=layer,
            )
            for bar, value, base in zip(bars, values, bottom):
                if value >= 8.0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        base + value / 2,
                        f"{value:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=6.9,
                        color=COLORS["dark"],
                    )
            bottom += values

        ax.set_xticks(x, bins, rotation=20, ha="right")
        ax.set_xlabel("GT 长边分桶 / px")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color=COLORS["light_gray"], lw=0.65)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"({'ab'[panel]}) {method}", loc="left", fontsize=9.2)
    axes[0].set_ylabel("正样本层级占比 / %")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.53, 1.02))
    fig.subplots_adjust(left=0.085, right=0.995, top=0.86, bottom=0.22, wspace=0.12)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv", type=Path)
    source.add_argument("--baseline-json", type=Path)
    parser.add_argument("--ca-json", type=Path)
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument("--ca-label", default="CA")
    parser.add_argument("--exclude-under-100", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    frame = load_frame(args)
    fig = build_figure(frame)
    paths = save_figure(fig, ensure_output_dir(args.output_dir), "fig5_positive_level_distribution")
    plt.close(fig)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
