"""Render the bounded topological-staging control used in the supplement.

The figure intentionally reports agreement with one sequential-topological
reference for one monotone update.  It does not present the experiment as a
barrier lower bound or as evidence about global step-entry snapshots.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, pstdev

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "artifacts" / "topological_staging_results.json"
OUTPUT = ROOT / "figs" / "plot11.png"
SCALE = 2
WIDTH, HEIGHT = 1500, 721


def px(value: float) -> int:
    return round(value * SCALE)


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, px(size))
    return ImageFont.load_default(size=px(size))


def centered(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, face, fill):
    x, y = map(px, xy)
    box = draw.textbbox((0, 0), text, font=face)
    draw.text((x - (box[2] - box[0]) / 2, y), text, font=face, fill=fill)


def vertical_label(image: Image.Image, center: tuple[float, float], text: str) -> None:
    face = font(20)
    box = face.getbbox(text)
    label = Image.new(
        "RGBA", (box[2] - box[0] + px(12), box[3] - box[1] + px(12)), (255, 255, 255, 0)
    )
    ImageDraw.Draw(label).text((px(6), px(4)), text, font=face, fill=(20, 20, 20, 255))
    label = label.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    x, y = map(px, center)
    image.alpha_composite(label, (round(x - label.width / 2), round(y - label.height / 2)))


def draw_panel_axes(
    draw: ImageDraw.ImageDraw,
    bounds: tuple[float, float, float, float],
    centers: list[float],
    depths: list[int],
) -> None:
    left, top, right, bottom = bounds
    grid = (220, 220, 220, 255)
    ink = (30, 30, 30, 255)
    tick_face = font(16)
    for index in range(6):
        value = index / 5
        y = bottom - value * (bottom - top)
        draw.line((px(left), px(y), px(right), px(y)), fill=grid, width=px(1))
        label = f"{value:.1f}"
        box = draw.textbbox((0, 0), label, font=tick_face)
        draw.text(
            (px(left - 10) - (box[2] - box[0]), px(y) - (box[3] - box[1]) / 2),
            label,
            font=tick_face,
            fill=ink,
        )
    draw.line((px(left), px(top), px(left), px(bottom)), fill=ink, width=px(2))
    draw.line((px(left), px(bottom), px(right), px(bottom)), fill=ink, width=px(2))
    for center, depth in zip(centers, depths):
        draw.line((px(center), px(bottom), px(center), px(bottom + 6)), fill=ink, width=px(2))
        centered(draw, (center, bottom + 11), str(depth), tick_face, ink)


def main() -> None:
    payload = json.loads(INPUT.read_text())
    rows = payload["rows"]
    depths = sorted({int(row["ell"]) for row in rows})
    grouped = {depth: [row for row in rows if int(row["ell"]) == depth] for depth in depths}

    image = Image.new("RGBA", (px(WIDTH), px(HEIGHT)), "white")
    draw = ImageDraw.Draw(image)
    ink = (25, 25, 25, 255)
    gray = (85, 85, 85, 255)
    red = (222, 38, 40, 255)
    purple = (126, 52, 153, 255)

    centered(
        draw,
        (WIDTH / 2, 12),
        "Generated-DAG check against a sequential-topological reference (90 DAGs)",
        font(25),
        ink,
    )
    centered(
        draw,
        (WIDTH / 2, 47),
        "Specific monotone update; staged-execution control, not a general lower-bound result",
        font(17),
        gray,
    )

    legend_y = 83
    draw.rectangle((px(470), px(legend_y), px(494), px(legend_y + 18)), fill=red)
    draw.text((px(504), px(legend_y - 2)), "longest-path staging", font=font(16), fill=ink)
    draw.rectangle((px(770), px(legend_y), px(794), px(legend_y + 18)), fill=purple)
    draw.text((px(804), px(legend_y - 2)), "deepest layers merged", font=font(16), fill=ink)

    left_bounds = (120, 180, 705, 610)
    right_bounds = (835, 180, 1430, 610)
    left_centers = [205 + 110 * index for index in range(len(depths))]
    right_centers = [920 + 112 * index for index in range(len(depths))]
    draw_panel_axes(draw, left_bounds, left_centers, depths)
    draw_panel_axes(draw, right_bounds, right_centers, depths)

    draw.text((px(86), px(126)), "a", font=font(24, bold=True), fill=ink)
    centered(draw, (412, 126), "Graph-level exact matches", font(21), ink)
    draw.text((px(800), px(126)), "b", font=font(24, bold=True), fill=ink)
    centered(draw, (1132, 126), "Cell-level agreement", font(21), ink)

    vertical_label(image, (45, 395), "fraction of generated graphs")
    vertical_label(image, (760, 395), "fraction of cells matching reference")
    centered(draw, (412, 669), "longest directed path length", font(20), ink)
    centered(draw, (1132, 669), "longest directed path length", font(20), ink)

    def y_for(value: float, bounds: tuple[float, float, float, float]) -> float:
        _, top, _, bottom = bounds
        return bottom - value * (bottom - top)

    bar_width = 30
    for center, depth in zip(left_centers, depths):
        cases = grouped[depth]
        full_exact = sum(math.isclose(row["full_correct"], 1.0) for row in cases)
        short_exact = sum(math.isclose(row["short_correct"], 1.0) for row in cases)
        for offset, value, color, count in (
            (-bar_width, full_exact / len(cases), red, full_exact),
            (0, short_exact / len(cases), purple, short_exact),
        ):
            x0 = center + offset
            x1 = x0 + bar_width
            y = y_for(value, left_bounds)
            draw.rectangle((px(x0), px(y), px(x1), px(left_bounds[3])), fill=color)
            if value == 0:
                draw.line(
                    (px(x0), px(left_bounds[3] - 2), px(x1), px(left_bounds[3] - 2)),
                    fill=color,
                    width=px(4),
                )
                label_y = left_bounds[3] - 28
                label_color = purple
            else:
                label_y = y + 8
                label_color = (255, 255, 255, 255)
            centered(draw, ((x0 + x1) / 2, label_y), f"{count}/{len(cases)}", font(13, bold=True), label_color)

    for center, depth in zip(right_centers, depths):
        cases = grouped[depth]
        full_values = [float(row["full_correct"]) for row in cases]
        short_values = [float(row["short_correct"]) for row in cases]
        for offset, values, color in (
            (-bar_width, full_values, red),
            (0, short_values, purple),
        ):
            value = mean(values)
            spread = pstdev(values)
            x0 = center + offset
            x1 = x0 + bar_width
            xmid = (x0 + x1) / 2
            y = y_for(value, right_bounds)
            draw.rectangle((px(x0), px(y), px(x1), px(right_bounds[3])), fill=color)
            high = y_for(min(1.0, value + spread), right_bounds)
            low = y_for(max(0.0, value - spread), right_bounds)
            draw.line((px(xmid), px(high), px(xmid), px(low)), fill=ink, width=px(2))
            draw.line((px(xmid - 6), px(high), px(xmid + 6), px(high)), fill=ink, width=px(2))
            draw.line((px(xmid - 6), px(low), px(xmid + 6), px(low)), fill=ink, width=px(2))

    centered(draw, (1132, 695), "bars: mean; whiskers: population SD across 18 graphs per depth", font(14), gray)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    final = image.convert("RGB").resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    final.save(OUTPUT, optimize=True)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
