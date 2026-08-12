"""Generate the paper's proposal-refinement figure from audited CSV files.

The left panel visualizes checkpoint selection on the training holdout subset.  The
right panel reports the independent FP32/batch=8 validation comparison used in the
paper.  No metric is recomputed by this script; it only reads exported CSV values.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

# Keep Matplotlib's cache inside the workspace on locked-down Windows systems.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache" / "matplotlib"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    ("map50_95", "mAP@0.50:0.95"),
    ("ap75", "AP@0.75"),
    ("ap90", "AP@0.90"),
    ("ap95", "AP@0.95"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--selected-epoch", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="fig4_refine_v311_curve")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial"],
            "font.size": 10.5,
            "axes.unicode_minus": False,
            "axes.edgecolor": "#8A96A3",
            "axes.linewidth": 0.8,
            "axes.titleweight": "bold",
            "axes.titlepad": 10,
            "xtick.color": "#425466",
            "ytick.color": "#425466",
            "text.color": "#243240",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def main() -> None:
    args = parse_args()
    holdout = read_csv(args.holdout_csv)
    validation = read_csv(args.val_csv)

    coarse_holdout = {int(r["epoch"]): float(r["map50_95"]) for r in holdout if r["variant"] == "coarse"}
    refined_holdout = {int(r["epoch"]): float(r["map50_95"]) for r in holdout if r["variant"] == "refined"}
    epochs = sorted(set(coarse_holdout) & set(refined_holdout))
    if not epochs:
        raise ValueError("holdout CSV has no paired coarse/refined rows")
    if args.selected_epoch not in epochs:
        raise ValueError(f"selected epoch {args.selected_epoch} not present")

    val_by_variant = {r["variant"]: r for r in validation}
    if "coarse" not in val_by_variant or "refined" not in val_by_variant:
        raise ValueError("validation CSV must contain coarse and refined rows")

    configure_style()
    en = args.language == "en"
    text = {
        "coarse": "CA (coarse)",
        "refined": "CA + Refine",
        "selected": f"Selected epoch {args.selected_epoch}" if en else f"选定 epoch {args.selected_epoch}",
        "panel_a": "(a) Checkpoint selection on the training holdout" if en else "(a) Holdout 上的检查点选择",
        "epoch": "Epoch" if en else "训练轮次 / epoch",
        "panel_b": "(b) Independent validation (FP32, batch=8)" if en else "(b) 独立验证集定位指标（FP32，batch=8）",
        "metric": "Metric value" if en else "指标值",
        "title": "Checkpoint selection and independent validation of proposal-level refinement" if en else "候选框级几何精修的训练选择与独立验证结果",
    }
    colors = {"coarse": "#5B6770", "refined": "#2F80ED", "accent": "#F2994A"}
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.65), gridspec_kw={"width_ratios": [1.08, 1]})

    # (a) Holdout checkpoint curve.
    ax = axes[0]
    y_coarse = [coarse_holdout[e] for e in epochs]
    y_refined = [refined_holdout[e] for e in epochs]
    ax.plot(epochs, y_coarse, color=colors["coarse"], lw=1.8, ls="--", label=text["coarse"])
    ax.plot(epochs, y_refined, color=colors["refined"], lw=2.2, marker="o", ms=4.2, label=text["refined"])
    selected_y = refined_holdout[args.selected_epoch]
    ax.scatter([args.selected_epoch], [selected_y], s=90, zorder=5, color=colors["accent"], edgecolor="white", linewidth=1.4)
    ax.annotate(
        f"{text['selected']}\n{selected_y:.4f}",
        xy=(args.selected_epoch, selected_y),
        xytext=(args.selected_epoch + 1.0, selected_y + 0.012),
        arrowprops={"arrowstyle": "->", "color": colors["accent"], "lw": 1.2},
        color="#A44F12",
        fontsize=9.5,
    )
    ax.set_title(text["panel_a"])
    ax.set_xlabel(text["epoch"])
    ax.set_ylabel("mAP@0.50:0.95")
    ax.set_xticks(epochs)
    ax.set_ylim(min(y_coarse) - 0.015, max(y_refined) + 0.035)
    ax.grid(axis="y", color="#DCE3E8", lw=0.7, alpha=0.9)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#D0D7DE")

    # (b) Independent validation metrics.
    ax = axes[1]
    x = np.arange(len(METRICS))
    width = 0.34
    coarse_values = np.array([float(val_by_variant["coarse"][key]) for key, _ in METRICS])
    refined_values = np.array([float(val_by_variant["refined"][key]) for key, _ in METRICS])
    bars_c = ax.bar(x - width / 2, coarse_values, width, color=colors["coarse"], label=text["coarse"], zorder=3)
    bars_r = ax.bar(x + width / 2, refined_values, width, color=colors["refined"], label=text["refined"], zorder=3)
    ax.set_title(text["panel_b"])
    ax.set_ylabel(text["metric"])
    ax.set_xticks(x, [label for _, label in METRICS])
    ax.set_ylim(0.0, 0.66)
    ax.grid(axis="y", color="#DCE3E8", lw=0.7, alpha=0.9, zorder=0)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#D0D7DE")

    for bars in (bars_c, bars_r):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.012, f"{height:.3f}", ha="center", va="bottom", fontsize=8.7)
    for idx, (coarse, refined) in enumerate(zip(coarse_values, refined_values)):
        delta = refined - coarse
        top = max(coarse, refined)
        ax.text(idx, top + 0.055, f"Δ {delta:+.3f}", ha="center", va="bottom", fontsize=9.2, color="#A44F12" if delta < 0 else "#166534")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(text["title"], fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout(w_pad=2.6)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / f"{args.stem}.png"
    svg_path = args.output_dir / f"{args.stem}.svg"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    summary_path = args.output_dir / f"{args.stem}_data.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "coarse", "refined", "delta_refined_minus_coarse", "source"])
        for (key, label), coarse, refined in zip(METRICS, coarse_values, refined_values):
            writer.writerow([label, f"{coarse:.12f}", f"{refined:.12f}", f"{refined - coarse:.12f}", str(args.val_csv)])

    print(png_path)
    print(svg_path)
    print(summary_path)


if __name__ == "__main__":
    main()
