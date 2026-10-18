#!/usr/bin/env python3
"""Plot verified aggregate development and ensemble results with ReportLab.

The adjacent CSV preserves unrounded results, and the palette JSON stores the
user's J1-J4 colors. No participant-level data are needed to rebuild the figure.
"""

import csv
from pathlib import Path
from copn_analysis.private_io import private_input, private_output, private_directory

from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import reportlab


INK = HexColor('#222222')
MUTED = HexColor('#555555')
GRID = HexColor('#E5E5E5')
FONTS = Path(reportlab.__file__).resolve().parent / 'fonts'
pdfmetrics.registerFont(TTFont('BudgetSans', str(FONTS / 'Vera.ttf')))
pdfmetrics.registerFont(TTFont('BudgetSansBold', str(FONTS / 'VeraBd.ttf')))


def text(drawing, x, y, value, size=10, bold=False, anchor='start', color=INK):
    drawing.add(String(x, y, value, fontName='BudgetSansBold' if bold else 'BudgetSans',
                       fontSize=size, textAnchor=anchor, fillColor=color))


def panel(drawing, palette, *, x, width, title_y, plot_y, plot_height,
          letter, title, subtitle, budgets, values, maximum, step, delta_note):
    text(drawing, x, title_y, letter, 11.5, bold=True)
    text(drawing, x + 17, title_y, title, 11, bold=True)
    text(drawing, x + 17, title_y - 15, subtitle, 9.6, color=MUTED)

    chart = VerticalBarChart()
    chart.x = x + 29
    chart.y = plot_y
    chart.width = width - 38
    chart.height = plot_height
    chart.data = [values]
    chart.barWidth = 14
    chart.groupSpacing = 17
    chart.barSpacing = 0
    chart.strokeColor = None
    chart.fillColor = None
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = maximum
    chart.valueAxis.valueStep = step
    chart.valueAxis.labelTextFormat = '%.1f'
    chart.valueAxis.labels.fontName = 'BudgetSans'
    chart.valueAxis.labels.fontSize = 9
    chart.valueAxis.labels.fillColor = MUTED
    chart.valueAxis.labels.dx = -4
    chart.valueAxis.visibleTicks = False
    chart.valueAxis.visibleAxis = False
    chart.valueAxis.visibleGrid = False
    for index in range(1, round(maximum / step) + 1):
        y = plot_y + plot_height * index * step / maximum
        drawing.add(Line(chart.x, y, chart.x + chart.width, y,
                         strokeColor=GRID, strokeWidth=0.45))
    chart.categoryAxis.categoryNames = budgets
    chart.categoryAxis.labels.fontName = 'BudgetSans'
    chart.categoryAxis.labels.fontSize = 10
    chart.categoryAxis.labels.fillColor = INK
    chart.categoryAxis.labels.dy = -5
    chart.categoryAxis.strokeColor = HexColor('#8A8A8A')
    chart.categoryAxis.strokeWidth = 0.6
    chart.categoryAxis.visibleTicks = False
    chart.barLabelFormat = '%.4f'
    chart.barLabels.fontName = 'BudgetSans'
    chart.barLabels.fontSize = 10
    chart.barLabels.fillColor = INK
    chart.barLabels.dy = 5
    for index, budget in enumerate(budgets):
        chart.bars[(0, index)].fillColor = HexColor(palette[budget])
        chart.bars[(0, index)].strokeColor = INK if budget == 'J3' else None
        chart.bars[(0, index)].strokeWidth = 1.05 if budget == 'J3' else 0
        if budget == 'J3':
            chart.categoryAxis.labels[index].fontName = 'BudgetSansBold'
    drawing.add(chart)
    text(drawing, x + width / 2, plot_y - 37, delta_note, 9.2,
         anchor='middle', color=MUTED)


def build(data: Path, output_dir: Path):
    data = private_input(data)
    output_dir = private_directory(output_dir)
    with data.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    palette = {'J1': '#98C897', 'J2': '#9C6BA3', 'J3': '#9DBAD2', 'J4': '#FE9C9D'}
    if set(palette) != {'J1', 'J2', 'J3', 'J4'}:
        raise ValueError('The palette must assign exactly one color to each budget.')
    values = {(row['stage'], row['model'], row['metric'], row['budget']):
              float(row['value']) for row in rows}
    if len(rows) != 14 or len(values) != 14:
        raise ValueError('Expected exactly 14 unique source values.')

    drawing = Drawing(480, 430)
    drawing.add(Rect(0, 0, 480, 430, fillColor=white, strokeColor=None))
    text(drawing, 240, 412, 'Performance across assessment budgets', 13,
         bold=True, anchor='middle')
    text(drawing, 12, 392, 'Component models during development', 11.5, bold=True)
    for x, letter, model, name in [(12, 'A', 'M2', 'M2 - CatBoost'),
                                   (250, 'B', 'M4', 'M4 - Hierarchical classifier')]:
        budgets = ['J1', 'J2', 'J3', 'J4']
        scores = [values[('development', model, 'balanced_accuracy', b)] for b in budgets]
        delta = 100 * (scores[2] - scores[1])
        panel(drawing, palette, x=x, width=218, title_y=371,
              plot_y=220, plot_height=123, letter=letter, title=name,
              subtitle='Balanced accuracy (higher is better)', budgets=budgets,
              values=scores, maximum=0.8, step=0.2,
              delta_note=f'J2 to J3: {delta:+.2f} percentage points')

    drawing.add(Line(12, 170, 468, 170, strokeColor=GRID, strokeWidth=0.8))
    text(drawing, 12, 153, 'Final probability ensemble', 11.5, bold=True)
    specifications = [
        (12, 'C', 'log_loss', 'Log loss', 'Lower is better', 0.4, 0.1),
        (172, 'D', 'brier', 'Brier score', 'Lower is better', 0.2, 0.1),
        (332, 'E', 'ap_pr_auc', 'AP PR-AUC', 'Higher is better', 0.4, 0.1),
    ]
    for x, letter, metric, name, subtitle, maximum, step in specifications:
        budgets = ['J3', 'J4']
        scores = [values[('final_ensemble', 'ensemble', metric, b)] for b in budgets]
        delta = scores[1] - scores[0]
        panel(drawing, palette, x=x, width=136, title_y=132,
              plot_y=44, plot_height=63, letter=letter, title=name,
              subtitle=subtitle, budgets=budgets, values=scores,
              maximum=maximum, step=step,
              delta_note=f'J4 - J3: {delta:+.4f}')

    renderPDF.drawToFile(drawing, str(output_dir / 'figure_budget_performance.pdf'))
    renderSVG.drawToFile(drawing, str(output_dir / 'figure_budget_performance.svg'))
    print('Exported vector PDF and SVG with 14 bars and the user palette.')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.data, args.output_dir)
