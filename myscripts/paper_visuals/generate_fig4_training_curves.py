"""Generate Fig. 4: mAP50-95 training curves from Ultralytics results CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from myscripts.paper_visuals.common import (
    COLORS,
    configure_matplotlib,
    ensure_output_dir,
    require_columns,
    require_nonempty,
    save_figure,
)


configure_matplotlib()
import matplotlib.pyplot as plt


DEFAULT_METRIC = "metrics/mAP50-95(B)"


def parse_mapping(values: list[str], option_name: str) -> dict[str, str]:
    """Parse repeated ``label=value`` CLI arguments."""
    mapping = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"{option_name} 必须使用 label=value 格式: {item}")
        label, value = item.split("=", 1)
        mapping[label.strip()] = value.strip()
    return mapping


def load_wide_series(series: dict[str, str], metric: str, offsets: dict[str, int]) -> pd.DataFrame:
    """Load one Ultralytics results CSV per method."""
    rows = []
    for label, path_text in series.items():
        path = Path(path_text)
        frame = pd.read_csv(path)
        frame.columns = [column.strip() for column in frame.columns]
        require_nonempty(frame, path)
        require_columns(frame, ["epoch", metric], path)
        offset = offsets.get(label, 0)
        for _, row in frame.iterrows():
            if pd.isna(row[metric]):
                continue
            rows.append({"epoch": int(row["epoch"]) + offset, "method": label, "map50_95": float(row[metric])})
    return pd.DataFrame(rows)


def load_tidy_series(path: Path) -> pd.DataFrame:
    """Load a long-form curve file, including output from ``collect_refine_ab_curve.py``."""
    frame = pd.read_csv(path)
    frame.columns = [column.strip() for column in frame.columns]
    require_nonempty(frame, path)
    require_columns(frame, ["epoch", "method", "map50_95"], path)
    return frame[["epoch", "method", "map50_95"]].copy()


def build_figure(frame: pd.DataFrame, smooth_window: int):
    """Plot raw curves lightly and an optional rolling mean prominently."""
    require_nonempty(frame, "curve data")
    methods = list(dict.fromkeys(frame["method"].astype(str)))
    fallback = [COLORS["purple"], COLORS["sky"], COLORS["vermillion"]]

    def series_style(label: str, index: int) -> tuple[str, object, str, float]:
        """Bind scientific meaning to a stable visual style, independent of CSV ordering."""
        key = label.lower().replace("_", " ")
        if "baseline" in key:
            return COLORS["baseline"], "-", "o", 1.35
        if "coarse" in key:
            return COLORS["ca"], ":", "s", 1.85
        if "normal" in key or ("refine" in key and "coarse" not in key):
            return COLORS["refine"], "-", "^", 1.65
        if "continue" in key:
            return COLORS["p4"], "--", "D", 1.55
        if key.strip() == "ca" or key.startswith("ca "):
            return COLORS["ca"], "-", "o", 1.45
        return fallback[index % len(fallback)], "-.", "v", 1.35

    fig, ax = plt.subplots(figsize=(6.8, 3.25))
    for index, method in enumerate(methods):
        subset = frame[frame["method"].astype(str) == method].sort_values("epoch")
        x = subset["epoch"].to_numpy(dtype=float)
        y = subset["map50_95"].to_numpy(dtype=float)
        if np.nanmax(np.abs(y)) <= 1.5:
            y = y * 100.0
        color, linestyle, marker, linewidth = series_style(method, index)
        if smooth_window > 1 and len(y) >= smooth_window:
            ax.plot(x, y, color=color, lw=0.65, alpha=0.24)
            smooth = pd.Series(y).rolling(smooth_window, center=True, min_periods=1).mean().to_numpy()
            ax.plot(x, smooth, color=color, lw=linewidth, ls=linestyle, label=method)
            plotted = smooth
        else:
            ax.plot(x, y, color=color, lw=linewidth, ls=linestyle, label=method)
            plotted = y
        best = int(np.nanargmax(plotted))
        ax.scatter(
            x[best],
            plotted[best],
            s=19,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=5,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP50–95 / %")
    ax.grid(color=COLORS["light_gray"], lw=0.65)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="lower right", columnspacing=1.1, handlelength=2.4)
    ax.margins(x=0.01)

    # Only state equality when the supplied values prove it at shared checkpoints.
    ca_labels = [method for method in methods if method.strip().lower() == "ca"]
    coarse_labels = [method for method in methods if "coarse" in method.lower()]
    if ca_labels and coarse_labels:
        ca = frame[frame["method"].astype(str) == ca_labels[0]].set_index("epoch")["map50_95"]
        coarse = frame[frame["method"].astype(str) == coarse_labels[0]].set_index("epoch")["map50_95"]
        shared = ca.index.intersection(coarse.index)
        if len(shared) >= 2 and np.allclose(ca.loc[shared], coarse.loc[shared], rtol=0.0, atol=1e-10):
            ax.text(
                0.015,
                0.975,
                "CA 与 coarse-only 在同 checkpoint 重合",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.2,
                color=COLORS["muted"],
                bbox={"facecolor": COLORS["paper"], "edgecolor": COLORS["border"], "pad": 2.4},
            )
    fig.subplots_adjust(left=0.10, right=0.995, top=0.98, bottom=0.17)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series",
        action="append",
        default=[],
        metavar="LABEL=CSV",
        help="Ultralytics results.csv；可重复传入",
    )
    parser.add_argument("--tidy-csv", type=Path, action="append", default=[], help="epoch,method,map50_95 长表")
    parser.add_argument("--epoch-offset", action="append", default=[], metavar="LABEL=N")
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--smooth-window", type=int, default=5, help="移动平均窗口；原始曲线仍以浅色显示")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    offsets = {label: int(value) for label, value in parse_mapping(args.epoch_offset, "--epoch-offset").items()}
    frames = []
    series = parse_mapping(args.series, "--series")
    if series:
        frames.append(load_wide_series(series, args.metric, offsets))
    frames.extend(load_tidy_series(path) for path in args.tidy_csv)
    if not frames:
        raise ValueError("请至少提供一个 --series LABEL=CSV 或 --tidy-csv")
    frame = pd.concat(frames, ignore_index=True)
    if frame["method"].nunique() < 2:
        raise ValueError("图4至少需要两个方法系列")

    fig = build_figure(frame, max(1, args.smooth_window))
    paths = save_figure(fig, ensure_output_dir(args.output_dir), "fig4_map5095_training_curves")
    plt.close(fig)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
