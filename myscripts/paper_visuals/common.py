"""Shared plotting, validation, and export helpers for paper visuals."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "mydocs" / "创新点一" / "paper_visuals" / "outputs"
MPL_CACHE = Path(tempfile.gettempdir()) / "myultralytics-paper-visuals-mpl"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

FONT_CN_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_CN_BOLD_PATH = Path(r"C:\Windows\Fonts\msyhbd.ttc")
FONT_EN_PATH = Path(r"C:\Windows\Fonts\times.ttf")
FONT_EN_BOLD_PATH = Path(r"C:\Windows\Fonts\timesbd.ttf")

COLORS = {
    # Shared semantic palette.  These values intentionally match the editable
    # PowerPoint architecture figure so that mechanism and data figures read as
    # one visual system in Word.
    "background": "#FBF8F1",
    "paper": "#FFFDF8",
    "ink": "#24313D",
    "muted": "#66717E",
    "border": "#D8DDE2",
    "baseline": "#77838F",
    "ca": "#397ED1",
    "ca_light": "#E3EFFC",
    "refine": "#F06A3B",
    "refine_light": "#FDE5DA",
    "success": "#1B9C68",
    "p3": "#55A7D7",
    "p4": "#F1A45B",
    "p5": "#48B99A",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "gray": "#6B7280",
    "light_gray": "#E5E9ED",
    "dark": "#24313D",
}


def configure_matplotlib():
    """Configure deterministic, Chinese-capable, publication-oriented Matplotlib defaults."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams

    for path in (FONT_CN_PATH, FONT_CN_BOLD_PATH, FONT_EN_PATH, FONT_EN_BOLD_PATH):
        if path.exists():
            font_manager.fontManager.addfont(str(path))

    rcParams.update(
        {
            "font.family": ["Times New Roman", "Microsoft YaHei", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.unicode_minus": False,
            "axes.linewidth": 0.8,
            "figure.facecolor": COLORS["background"],
            "axes.facecolor": COLORS["paper"],
            "savefig.facecolor": COLORS["background"],
            "axes.edgecolor": COLORS["border"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "text.color": COLORS["ink"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "stix",
        }
    )
    return matplotlib


def ensure_output_dir(path: str | Path | None) -> Path:
    """Create and return an output directory."""
    output = Path(path) if path else DEFAULT_OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    return output


def save_figure(fig, output_dir: str | Path, stem: str, *, dpi: int = 600) -> list[Path]:
    """Save vector masters plus a high-resolution PNG preview."""
    output = ensure_output_dir(output_dir)
    paths = []
    for suffix, options in (
        (".svg", {}),
        (".pdf", {"metadata": {"Creator": "Myultralytics paper_visuals"}}),
        (".png", {"dpi": dpi}),
    ):
        path = output / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.035, **options)
        paths.append(path)
    return paths


def read_json(path: str | Path) -> dict:
    """Read a UTF-8 JSON object."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_h1h2(path: str | Path) -> dict:
    """Load and minimally validate ``check_h1h2_stats.py`` output."""
    data = read_json(path)
    required = {"config", "h1", "h2"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"{path} 缺少字段: {sorted(missing)}")
    return data


def natural_bin_key(label: str) -> tuple[float, float, str]:
    """Return a stable sort key for labels such as '<100', '100-200', and '>500'."""
    values = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", label)]
    if label.startswith("<"):
        return -math.inf, values[0] if values else math.inf, label
    if label.startswith(">") or label.startswith("≥"):
        return values[0] if values else math.inf, math.inf, label
    if len(values) >= 2:
        return values[0], values[1], label
    return values[0] if values else math.inf, values[0] if values else math.inf, label


def h1_records(data: dict, method: str) -> list[dict]:
    """Convert H1 JSON entries into plotting records."""
    records = []
    for label, row in sorted(data["h1"].items(), key=lambda item: natural_bin_key(item[0])):
        total = int(row.get("total", 0))
        overflow = int(row.get("overflow", 0))
        records.append(
            {
                "method": method,
                "long_bin": label,
                "total": total,
                "overflow": overflow,
                "overflow_rate": 100.0 * overflow / total if total else np.nan,
                "dreq_mean": float(row.get("dreq_mean", np.nan)),
                "dreq_p95": float(row.get("dreq_p95", np.nan)),
            }
        )
    return records


def h2_records(data: dict, method: str) -> list[dict]:
    """Convert H2 JSON entries into normalized P3/P4/P5 distribution records."""
    strides = [int(value) for value in data["config"].get("strides", [8, 16, 32])]
    records = []
    for label, counts in sorted(data["h2"].items(), key=lambda item: natural_bin_key(item[0])):
        values = {stride: int(counts.get(str(stride), 0)) for stride in strides}
        total = sum(values.values())
        row = {"method": method, "long_bin": label, "total": total}
        for stride, count in values.items():
            row[f"P{int(math.log2(stride))}"] = 100.0 * count / total if total else 0.0
        records.append(row)
    return records


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a Wilson score interval as percentages without requiring SciPy."""
    if total <= 0:
        return np.nan, np.nan
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    return 100 * max(0.0, center - radius), 100 * min(1.0, center + radius)


def require_columns(frame, columns: Iterable[str], source: str | Path) -> None:
    """Raise a concise error when a tabular input is missing required columns."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} 缺少列: {missing}")


def require_nonempty(frame, source: str | Path) -> None:
    """Reject empty templates before plotting."""
    if frame.empty:
        raise ValueError(f"{source} 只有表头，没有可绘制数据")
