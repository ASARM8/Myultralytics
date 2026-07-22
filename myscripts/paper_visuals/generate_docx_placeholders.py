"""Generate clean manuscript placeholders for figures that still require real experiment data."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CANVAS = (2400, 760)
BG = "#FBFCFE"
BORDER = "#B7C0CC"
INK = "#263442"
MUTED = "#617083"
BLUE = "#2878B5"
PALE_BLUE = "#EAF3FA"


PLACEHOLDERS = {
    3: (
        "DFL 溢出率分桶图",
        "等待 Baseline 与 CA 的 h1h2_data.json",
        "正式图只使用 imgsz=640 下的真实正样本统计，并给出 Wilson 95% 置信区间。",
    ),
    4: (
        "mAP50-95 训练/验证曲线",
        "等待 Baseline、CA、CA-continue 及 Refine A/B 数据",
        "coarse-only 与 CA 按同一推理定义展示；normal refine 的效果采用保守表述。",
    ),
    5: (
        "P3 / P4 / P5 正样本层级分布",
        "等待 Baseline 与 CA 的 h1h2_data.json",
        "正式图比较相同长边分桶下的层级占比，不使用预设趋势替代真实结果。",
    ),
    6: (
        "定性检测对比",
        "等待统一 ROI 的原图、GT 与三组预测 OBB",
        "四列依次为 Ground Truth、Baseline、CA、CA+Refine（normal refine）。",
    ),
}


def _font(candidates: list[Path], size: int) -> ImageFont.FreeTypeFont:
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    raise FileNotFoundError(f"Required publication font not found: {candidates}")


def _centered_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: str) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1] - (box[3] - box[1]) / 2), text, font=font, fill=fill)


def render_placeholder(index: int, output_path: Path) -> None:
    title, required, note = PLACEHOLDERS[index]
    image = Image.new("RGB", CANVAS, BG)
    draw = ImageDraw.Draw(image)

    font_root = Path("C:/Windows/Fonts")
    cn_bold = _font([font_root / "msyhbd.ttc", font_root / "simhei.ttf"], 78)
    cn_regular = _font([font_root / "msyh.ttc", font_root / "simsun.ttc"], 43)
    cn_small = _font([font_root / "msyh.ttc", font_root / "simsun.ttc"], 34)

    draw.rounded_rectangle((18, 18, CANVAS[0] - 18, CANVAS[1] - 18), radius=26, outline=BORDER, width=5)
    draw.rounded_rectangle((78, 72, 520, 142), radius=34, fill=PALE_BLUE, outline=BLUE, width=2)
    _centered_text(draw, (299, 105), "DATA PENDING · 数据待补", cn_small, BLUE)

    _centered_text(draw, (CANVAS[0] // 2, 278), title, cn_bold, INK)
    draw.line((200, 385, CANVAS[0] - 200, 385), fill="#D6DCE4", width=3)
    _centered_text(draw, (CANVAS[0] // 2, 485), required, cn_regular, MUTED)
    _centered_text(draw, (CANVAS[0] // 2, 600), note, cn_small, MUTED)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True, dpi=(300, 300))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for index in PLACEHOLDERS:
        render_placeholder(index, args.output_dir / f"figure_placeholder_{index}_current.png")


if __name__ == "__main__":
    main()
