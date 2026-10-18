#!/usr/bin/env python3
"""Plot the reported reliability curves from authorized private inputs.

Requires numpy, pandas and matplotlib. The caller supplies confidential curve and summary inputs outside this checkout.
Curve area uses the deterministic within-fold tie ordering.
"""
from pathlib import Path
from copn_analysis.private_io import private_input, private_output, private_directory
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ssc_reliability_mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "ssc_reliability_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

STEM = "figure_reliability_risk_coverage"
SIMPLE = "rowwise_training_selected_simple_score"
COMBINED = "multicomponent_six_feature_monotone"
COLORS = {SIMPLE: "#176A80", COMBINED: "#C85F37"}
LABELS = {SIMPLE: "Simple score", COMBINED: "Combined score (six signals)"}


def build(data_path: Path, summary_path: Path, output_dir: Path):
    data_path = private_input(data_path)
    summary_path = private_input(summary_path)
    output_dir = private_directory(output_dir)
    data = pd.read_csv(data_path, float_precision="round_trip")
    saved = json.loads(summary_path.read_text())
    summaries = {(r["budget"], r["score"]): r for r in saved["metrics"]}
    contrasts = {r["budget"]: r for r in saved["contrasts"]}
    curves = {}
    for budget in (3, 4):
        for score in (SIMPLE, COMBINED):
            frame = data[data["budget"].eq(budget) & data["score"].eq(score)]
            x = frame["retained_fraction_right_endpoint"].to_numpy(float)
            y = frame["observed_prefix_error_rate"].to_numpy(float)
            assert x[0] == 0 and x[-1] == 1 and np.all(np.diff(x) > 0)
            area = float(np.diff(x) @ y[1:])
            r = summaries[budget, score]
            assert np.isclose(area, r["aurc"], rtol=0, atol=1e-13)
            assert np.isclose(y[-1], r["errors"] / r["rows"], rtol=0, atol=1e-13)
            curves[budget, score] = (x, y)
        delta = summaries[budget, COMBINED]["aurc"] - summaries[budget, SIMPLE]["aurc"]
        assert np.isclose(delta, contrasts[budget]["estimate"], rtol=0, atol=1e-13)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.labelcolor": "#263945", "text.color": "#263945",
                         "xtick.color": "#53636D", "ytick.color": "#53636D",
                         "axes.edgecolor": "#CDD5DA", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(6.4, 3.15), facecolor="white")
    grid = fig.add_gridspec(1, 2, left=0.115, right=0.97,
                           bottom=0.18, top=0.80, wspace=0.24)
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    for ax, budget, letter in zip(axes, (3, 4), ("A", "B")):
        ax.set_title(f"{letter}  J{budget}" + (" (primary)" if budget == 3 else " (additional assessments)"),
                     loc="left", fontsize=9.2, weight="bold", pad=9)
        xs, ys = curves[budget, SIMPLE]
        xc, yc = curves[budget, COMBINED]
        assert np.array_equal(xs, xc)
        ax.fill_between(xs, ys, yc, step="pre", color=COLORS[COMBINED], alpha=0.12, linewidth=0)
        for score in (COMBINED, SIMPLE):
            x, y = curves[budget, score]
            ax.step(x, y, where="pre", lw=1.6, color=COLORS[score], zorder=3)
        r = summaries[budget, SIMPLE]
        ax.axhline(r["errors"] / r["rows"], color="#9AA5AD", lw=0.8, ls=(0, (4, 3)))
        ax.text(0.04, 0.84, f"{r['rows']:,} prediction rows\n{r['errors']} errors in all rows",
                transform=ax.transAxes, fontsize=7, color="#687780", zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
        ax.text(0.04, 0.64, "Area (AURC)", transform=ax.transAxes, fontsize=7.2, color="#687780")
        for ypos, score in [(0.54, SIMPLE), (0.44, COMBINED)]:
            short = "Simple" if score == SIMPLE else "Combined"
            ax.text(0.04, ypos, f"{short}  {summaries[budget, score]['aurc']:.6f}",
                    transform=ax.transAxes, fontsize=7.5, color=COLORS[score], weight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.10)
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.set_yticks(np.arange(0, .101, .02))
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.tick_params(labelsize=7.5, length=3)
        ax.set_xlabel("Predictions kept", fontsize=8.5, labelpad=5)
        if budget == 3:
            ax.set_ylabel("Error rate among\npredictions kept", fontsize=8.5, labelpad=6)
        ax.grid(axis="y", color="#E9EEF0", lw=0.65)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    handles = [Line2D([], [], color=COLORS[s], lw=2, label=LABELS[s]) for s in (SIMPLE, COMBINED)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, .988),
               ncol=2, frameon=False, fontsize=8.5, handlelength=2.5, columnspacing=1.7)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(output_dir / f"{STEM}.{ext}", dpi=220, facecolor="white")
    plt.close(fig)
    print("Built two-panel reliability figure; all four curve areas and both contrasts match saved results.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.data, args.summary, args.output_dir)
