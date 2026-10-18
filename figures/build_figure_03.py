#!/usr/bin/env python3
"""Build the separate SSI and supplementary M1 plots from saved aggregates.

Requires ReportLab. PDF and SVG exports are vector graphics. A PNG preview is
also written when pdftoppm is available. No statistical models are fitted.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from copn_analysis.private_io import private_input, private_output, private_directory
import shutil
import subprocess

from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Circle, Drawing, Group, Line, PolyLine, Rect, String
from reportlab.lib.colors import HexColor, white

NAVY = HexColor("#173B5E")
CORAL = HexColor("#C85C5C")
INK = HexColor("#1F2933")
GRID = HexColor("#D8DEE4")
FRAME = HexColor("#AEB8C2")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def text(drawing, x, y, value, *, size=10, color=INK, anchor="middle", bold=False):
    drawing.add(String(
        x, y, value, fontName="Helvetica-Bold" if bold else "Helvetica",
        fontSize=size, fillColor=color, textAnchor=anchor,
    ))


def vertical_text(drawing, x, y, value, color):
    label = Group(String(
        0, 0, value, fontName="Helvetica", fontSize=12,
        fillColor=color, textAnchor="middle",
    ))
    label.transform = (0, 1, -1, 0, x, y)
    drawing.add(label)


def line_series(drawing, points, color, square=False):
    if len(points) > 1:
        drawing.add(PolyLine(
            [coordinate for point in points for coordinate in point],
            strokeColor=color, strokeWidth=1.8, fillColor=None,
        ))
    for x, y in points:
        marker = (
            Rect(x - 3, y - 3, 6, 6, fillColor=color, strokeColor=color)
            if square else Circle(x, y, 3, fillColor=color, strokeColor=color)
        )
        drawing.add(marker)


def export(drawing: Drawing, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    renderPDF.drawToFile(drawing, str(path))
    renderSVG.drawToFile(drawing, str(path.with_suffix(".svg")))
    poppler = shutil.which("pdftoppm")
    if poppler:
        subprocess.run([
            poppler, "-png", "-singlefile", "-scale-to", "1800",
            str(path), str(path.with_suffix("")),
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def plot(rows, *, title, right_field, right_label, left_limits, left_ticks,
         right_limits, right_ticks, x_label, path, target=None, x_tick_labels=None):
    drawing = Drawing(560, 340)
    drawing.add(Rect(0, 0, 560, 340, fillColor=white, strokeColor=None))
    left, bottom, width, height = 70, 75, 418, 205
    count = len(rows)
    x = lambda index: left + width * index / (count - 1)
    y_left = lambda value: bottom + height * (value - left_limits[0]) / (
        left_limits[1] - left_limits[0]
    )
    y_right = lambda value: bottom + height * (value - right_limits[0]) / (
        right_limits[1] - right_limits[0]
    )

    text(drawing, 280, 321, title, size=14, color=NAVY, bold=True)
    for tick in left_ticks:
        y = y_left(tick)
        drawing.add(Line(left, y, left + width, y, strokeColor=GRID, strokeWidth=0.6))
        text(drawing, left - 10, y - 3, f"{tick:.1f}", size=11, color=NAVY, anchor="end")
    for tick in right_ticks:
        y = y_right(tick)
        drawing.add(Line(left + width, y, left + width + 4, y,
                         strokeColor=FRAME, strokeWidth=0.7))
        text(drawing, left + width + 9, y - 3, f"{tick:g}",
             size=11, color=CORAL, anchor="start")
    for index, row in enumerate(rows):
        drawing.add(Line(x(index), bottom, x(index), bottom - 4,
                         strokeColor=FRAME, strokeWidth=0.7))
        tick_label = x_tick_labels[index] if x_tick_labels is not None else row["display_group"]
        text(drawing, x(index), bottom - 18, tick_label, size=11.5)
    for x1, y1, x2, y2 in [
        (left, bottom, left, bottom + height),
        (left, bottom, left + width, bottom),
        (left + width, bottom, left + width, bottom + height),
    ]:
        drawing.add(Line(x1, y1, x2, y2, strokeColor=FRAME, strokeWidth=0.8))
    vertical_text(drawing, 22, bottom + height / 2, "A2 mean set size (labels)", NAVY)
    vertical_text(drawing, 544, bottom + height / 2, right_label, CORAL)
    text(drawing, left + width / 2, 27, x_label, size=12)

    if target is not None:
        drawing.add(Line(
            left, y_right(target), left + width, y_right(target),
            strokeColor=CORAL, strokeWidth=0.8, strokeDashArray=[3, 3],
        ))
        text(drawing, left + width - 5, y_right(target) + 6,
             f"{target:g}% target", size=10, color=CORAL, anchor="end")
    a2 = [(x(i), y_left(float(row["a2_mean_set_size"]))) for i, row in enumerate(rows)]
    right = [
        (x(i), y_right(100 * float(row[right_field])))
        for i, row in enumerate(rows) if row[right_field] != ""
    ]
    assert all(math.isfinite(value) for point in a2 + right for value in point)
    line_series(drawing, a2, NAVY)
    line_series(drawing, right, CORAL, square=True)

    for legend_x, color, label, square in [
        (82, NAVY, "A2 mean set size", False),
        (292, CORAL, right_label.replace(" (%)", ""), True),
    ]:
        line_series(drawing, [(legend_x, 298), (legend_x + 18, 298)], color, square)
        text(drawing, legend_x + 26, 294.5, label, size=11, anchor="start")
    export(drawing, path)


def build_ssi(data: Path, output_dir: Path) -> None:
    data = private_input(data)
    output_dir = private_directory(output_dir)
    ssi = read_rows(data)
    assert [row["source_group"] for row in ssi] == [f"Q{i}" for i in range(1, 6)]
    assert [row["display_group"] for row in ssi] == [
        "Lowest variation", "Lower variation", "Middle variation",
        "Higher variation", "Highest variation",
    ]
    plot(
        ssi, title="Variation across scenarios at J3",
        right_field="a3_singleton_error_rate", right_label="A3 single-label error (%)",
        left_limits=(1.7, 2.7), left_ticks=(1.8, 2.0, 2.2, 2.4, 2.6),
        right_limits=(0, 20), right_ticks=(0, 5, 10, 15, 20),
        x_label="Variation in the possible diagnoses returned across scenarios",
        x_tick_labels=["Lowest", "Lower", "Middle", "Higher", "Highest"],
        path=output_dir / "figure_03_labels.pdf",
    )
    # Keep the established manuscript PNG name for previews and archive consumers.
    preview = output_dir / "figure_03_labels.png"
    if preview.exists():
        shutil.copyfile(preview, output_dir / "figure_03.png")


def build_m1(data: Path, output_dir: Path) -> None:
    data = private_input(data)
    output_dir = private_directory(output_dir)
    m1 = read_rows(data)
    assert [row["display_group"] for row in m1] == ["D1", "D2", "D3", "D4"]
    plot(
        m1, title="Earlier M1-based deficit at J3",
        right_field="a5_hc_coverage", right_label="A5 HC core coverage (%)",
        left_limits=(1.5, 3.05), left_ticks=(1.5, 2.0, 2.5, 3.0),
        right_limits=(75, 102), right_ticks=(75, 80, 85, 90, 95, 100),
        x_label="M1 deficit group (increasing deficit)",
        path=output_dir / "figure_s02_m1_deficit.pdf", target=90,
    )


def build(ssi: Path, m1: Path, output_dir: Path) -> None:
    build_ssi(ssi, output_dir)
    build_m1(m1, output_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssi", type=Path, required=True)
    parser.add_argument("--m1", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.ssi, args.m1, args.output_dir)
