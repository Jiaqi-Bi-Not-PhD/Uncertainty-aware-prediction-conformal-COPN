#!/usr/bin/env python3
"""Build the manuscript's editable vector overview with ReportLab."""

from pathlib import Path
from copn_analysis.private_io import private_input, private_output, private_directory
from reportlab.lib.colors import HexColor
from reportlab.pdfgen import canvas

NAVY = '#173B5E'
TEAL = '#1F8A8A'
INK = '#1F2933'


def label(c, x, y, text, size, color=INK, bold=False, center=False):
    c.setFillColor(HexColor(color))
    c.setFont('Helvetica-Bold' if bold else 'Helvetica', size)
    (c.drawCentredString if center else c.drawString)(x, y, text)


def build(output: Path):
    output = private_output(output)
    c = canvas.Canvas(str(output), pagesize=(752, 380))
    label(c, 376, 353, 'Prediction uncertainty in incomplete C-OPN records',
          16, NAVY, True, True)
    cards = [
        (20, '#DCEAF2', NAVY, ('Incomplete C-OPN', 'records'),
         ('Observed clinical values', 'Structured missing states',
          'PD / AP / HC classification', 'Outcome-independent inputs')),
        (204, '#DDF1EF', TEAL, ('Training and', 'calibration'),
         ('Target-excluded Q estimation', 'Missingness scenarios',
          'Class-conditional label sets', 'Prespecified evaluation criteria')),
        (388, '#F7EDD1', '#9A6B00', ('Uncertainty', 'evaluation'),
         ('Prediction-set coverage', 'Set size and scenario stability',
          'Error-risk ranking', 'Information-deficit score Q')),
        (572, '#F5DEDE', '#C85C5C', ('Study', 'findings'),
         ('Coverage and set breadth', 'Uncertainty by information deficit',
          'Rare-class precision', 'Limits of internal validation')),
    ]
    for x, fill, border, title, bullets in cards:
        c.setFillColor(HexColor(fill))
        c.setStrokeColor(HexColor(border))
        c.setLineWidth(1.5)
        c.roundRect(x, 157, 160, 165, 10, fill=1, stroke=1)
        for y, text in zip((298, 282), title):
            label(c, x + 80, y, text, 12, border, True, True)
        for y, text in zip((252, 229, 206, 183), bullets):
            c.setFillColor(HexColor(INK))
            c.circle(x + 10, y + 3, 1.25, fill=1, stroke=0)
            label(c, x + 16, y, text, 8.3)
        if x != 572:
            c.setStrokeColor(HexColor('#5F6B76'))
            c.setLineWidth(1.4)
            c.line(x + 163, 239, x + 181, 239)
            c.line(x + 181, 239, x + 175, 243)
            c.line(x + 181, 239, x + 175, 235)
    c.setFillColor(HexColor('#F4F7FA'))
    c.setStrokeColor(HexColor(NAVY))
    c.roundRect(60, 64, 632, 63, 10, fill=1, stroke=1)
    label(c, 376, 107, 'Scope of interpretation', 12, NAVY, True, True)
    label(c, 376, 89, 'The analysis compares prediction coverage, set breadth and model uncertainty.',
          9.5, center=True)
    label(c, 376, 74, 'Clinician usefulness and clinical benefit were not evaluated.',
          9.5, center=True)
    label(c, 376, 27, 'Evaluating prediction uncertainty in incomplete clinical records',
          12, TEAL, True, True)
    c.showPage()
    c.save()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.output)
