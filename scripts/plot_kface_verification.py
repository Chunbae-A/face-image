#!/usr/bin/env python3
"""K-FACE 3장·5장 test 점수 분포를 개인정보 없이 PNG로 그린다."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


FONT_PATHS = (
    Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_PATHS:
        if path.is_file():
            try:
                return ImageFont.truetype(str(path), size=size, index=1 if bold else 0)
            except OSError:
                continue
    return ImageFont.load_default()


def _panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    title: str,
    metrics: dict[str, Any],
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=18, fill="#F8FAFC", outline="#CBD5E1", width=2)
    draw.text((left + 24, top + 18), title, font=_font(26, bold=True), fill="#0F172A")
    subtitle = (
        f"AUC {metrics['roc_auc']:.4f}  |  TAR {metrics['tar']:.4f}  |  "
        f"FAR {metrics['far']:.6f}  |  EER {metrics['eer']:.4f}"
    )
    draw.text((left + 24, top + 58), subtitle, font=_font(16), fill="#334155")

    plot_left, plot_top = left + 55, top + 105
    plot_right, plot_bottom = right - 25, bottom - 52
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#64748B", width=2)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#64748B", width=2)
    genuine = metrics["genuine_histogram"]["counts"]
    impostor = metrics["impostor_histogram"]["counts"]
    bins = len(genuine)
    genuine_total = max(sum(genuine), 1)
    impostor_total = max(sum(impostor), 1)
    genuine_frequency = [value / genuine_total for value in genuine]
    impostor_frequency = [value / impostor_total for value in impostor]
    maximum = max(
        max(genuine_frequency, default=0), max(impostor_frequency, default=0), 1e-9
    )
    width = (plot_right - plot_left) / bins
    for index, frequency in enumerate(impostor_frequency):
        height = (plot_bottom - plot_top) * frequency / maximum
        x0 = plot_left + index * width
        draw.rectangle((x0, plot_bottom - height, x0 + max(1, width), plot_bottom), fill="#94A3B8")
    for index, frequency in enumerate(genuine_frequency):
        height = (plot_bottom - plot_top) * frequency / maximum
        x0 = plot_left + index * width
        draw.rectangle((x0, plot_bottom - height, x0 + max(1, width), plot_bottom), fill="#22C55E")

    threshold = float(metrics["threshold"])
    threshold_x = plot_left + (threshold + 1.0) / 2.0 * (plot_right - plot_left)
    draw.line((threshold_x, plot_top, threshold_x, plot_bottom), fill="#EF4444", width=4)
    draw.text(
        (max(plot_left, threshold_x - 55), plot_top + 4),
        f"기준 {threshold:.3f}",
        font=_font(15, bold=True),
        fill="#B91C1C",
    )
    draw.text((plot_left, plot_bottom + 10), "-1.0", font=_font(14), fill="#475569")
    draw.text((plot_right - 24, plot_bottom + 10), "1.0", font=_font(14), fill="#475569")
    legend_y = top + 82
    draw.rectangle((right - 250, legend_y, right - 232, legend_y + 13), fill="#94A3B8")
    draw.text((right - 225, legend_y - 4), "타인", font=_font(14), fill="#475569")
    draw.rectangle((right - 160, legend_y, right - 142, legend_y + 13), fill="#22C55E")
    draw.text((right - 135, legend_y - 4), "동일인", font=_font(14), fill="#475569")


def plot(payload: dict[str, Any], output: Path) -> None:
    image = Image.new("RGB", (1800, 1180), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    draw.text((70, 45), "K-FACE 400명 ArcFace 저·중화질 검증", font=_font(42, bold=True), fill="#0F172A")
    draw.text(
        (70, 102),
        "중화질 등록 3장·5장, subject-disjoint test 점수 분포",
        font=_font(23),
        fill="#475569",
    )
    boxes = (
        (70, 165, 870, 620),
        (930, 165, 1730, 620),
        (70, 665, 870, 1120),
        (930, 665, 1730, 1120),
    )
    panels = (
        ("3장 등록 · 저화질 질의", "references_3", "low"),
        ("3장 등록 · 중화질 질의", "references_3", "medium"),
        ("5장 등록 · 저화질 질의", "references_5", "low"),
        ("5장 등록 · 중화질 질의", "references_5", "medium"),
    )
    for box, (title, protocol, resolution) in zip(boxes, panels):
        _panel(
            draw,
            box,
            title=title,
            metrics=payload["protocols"][protocol]["conditions"][resolution]["test"],
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    plot(payload, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
