"""Scientific routines used by the manuscript; unused workflow/report code omitted."""

from __future__ import annotations

from pathlib import Path
from .private_io import private_output

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

NAVY = "#173B5E"

TEAL = "#1F8A8A"

GOLD = "#D9A441"

CORAL = "#C85C5C"

INK = "#1F2933"

GRID = "#D8DEE4"

Q_LABELS = {
    "frontier_q_dimnorm_primary_v1": "Primary\ndimension-normalized",
    "frontier_q_rawnorm_primarymap_v1": "Raw-norm\nweight sensitivity",
    "frontier_q_dimnorm_lowdeficit_v1": "Low-deficit\nstate sensitivity",
    "frontier_q_dimnorm_highdeficit_v1": "High-deficit\nstate sensitivity",
}

CONTEXT_LABELS = {
    "ordinary_cal25__cross_holdout_C0_dual_use": "Ordinary 25% / C0",
    "ordinary_cal30": "Ordinary 30%",
    "ordinary_cal35": "Ordinary 35%",
    "cross_holdout_C1": "Cross holdout C1",
    "cross_holdout_C2": "Cross holdout C2",
    "cross_holdout_C3": "Cross holdout C3",
}

def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.edgecolor": "#AEB8C2",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )

def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(private_output(path), bbox_inches="tight", facecolor="white")
    plt.close(fig)

def _row_lookup(rows: list[dict], key: str, value: str, q_definition: str | None = None) -> dict:
    for row in rows:
        if row.get(key) == value and (q_definition is None or row.get("q_definition") == q_definition):
            return row
    raise KeyError((key, value, q_definition))

def figure_frontier_q(q: dict, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.7))
    colors = [TEAL, NAVY, GOLD, CORAL]

    ax = axes[0, 0]
    definitions = list(Q_LABELS)
    for i, (definition, color) in enumerate(zip(definitions, colors)):
        stats = q["overall_q_summary"][definition]
        qq = stats["quantiles"]
        ax.plot([qq["q05"], qq["q95"]], [i, i], color=color, linewidth=2)
        ax.plot([qq["q25"], qq["q75"]], [i, i], color=color, linewidth=8, solid_capstyle="butt")
        ax.scatter([qq["q50"]], [i], color="white", edgecolor=color, linewidth=1.5, s=45, zorder=3)
    ax.set_yticks(range(4), [Q_LABELS[d] for d in definitions])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Information-deficit Q")
    ax.set_title("Four prespecified Q definitions\n(5th–95th, IQR, median)", color=NAVY, fontweight="bold")
    ax.grid(axis="x", color=GRID, linewidth=0.7)

    ax = axes[0, 1]
    primary = "frontier_q_dimnorm_primary_v1"
    contexts = list(CONTEXT_LABELS)
    for i, context in enumerate(contexts):
        row = _row_lookup(q["q_summary_by_context"], "physical_context_id", context, primary)
        qq = row["quantiles"]
        ax.plot([qq["q05"], qq["q95"]], [i, i], color=TEAL, linewidth=1.8)
        ax.plot([qq["q25"], qq["q75"]], [i, i], color=TEAL, linewidth=7)
        ax.scatter([qq["q50"]], [i], color="white", edgecolor=TEAL, linewidth=1.4, s=38, zorder=3)
    ax.set_yticks(range(6), [CONTEXT_LABELS[c] for c in contexts])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Primary Q")
    ax.set_title("Primary Q across analysis settings", color=NAVY, fontweight="bold")
    ax.grid(axis="x", color=GRID, linewidth=0.7)

    ax = axes[1, 0]
    matrix = np.array(
        [[q["participant_context_spearman_correlation"][r][c] for c in contexts] for r in contexts]
    )
    image = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="YlGnBu")
    short = ["25%/C0", "30%", "35%", "C1", "C2", "C3"]
    ax.set_xticks(range(6), short, rotation=35, ha="right")
    ax.set_yticks(range(6), short)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    fontsize=7.8, color="white" if matrix[i, j] < 0 else INK)
    ax.set_title("Correlation of participant rankings\n(Spearman correlation)", color=NAVY, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 1]
    stability = q["participant_stability_summary"]
    labels = ["Within-participant IQR", "Within-participant range"]
    keys = ["iqr_primary_q", "range_primary_q"]
    for i, (label, key, color) in enumerate(zip(labels, keys, [NAVY, CORAL])):
        qq = stability[key]["quantiles"]
        ax.plot([qq["q05"], qq["q95"]], [i, i], color=color, linewidth=2)
        ax.plot([qq["q25"], qq["q75"]], [i, i], color=color, linewidth=9)
        ax.scatter([qq["q50"]], [i], color="white", edgecolor=color, linewidth=1.5, s=48, zorder=3)
        ax.text(qq["q95"] + 0.01, i, f"median {qq['q50']:.3f}", va="center", fontsize=8.5, color=color)
    ax.set_yticks(range(2), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Variation in primary Q across analysis coordinates")
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_title("Within-participant variation across settings", color=NAVY, fontweight="bold")

    fig.suptitle(
        "Information-deficit score distributions and sensitivity",
        fontsize=14.2, color=NAVY, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    save(fig, path)

def figure_selector_patterns(q: dict, path: Path) -> None:
    rows = pd.DataFrame(q["selector_summary"]["candidate_frequency"])
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    pivot = rows.pivot_table(index="gamma", columns="AP_weight_multiplier", values="count", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(index=[0.5, 1.0, 2.0], columns=[1.0, 1.5, 2.0], fill_value=0)
    ax = axes[0]
    im = ax.imshow(pivot.to_numpy(), cmap="YlGnBu")
    ax.set_xticks(range(3), ["1.0", "1.5", "2.0"])
    ax.set_yticks(range(3), ["0.5", "1.0", "2.0"])
    ax.set_xlabel("AP loss multiplier")
    ax.set_ylabel("Gamma")
    ax.set_title("Selected loss block", color=NAVY, fontweight="bold")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(int(pivot.iloc[i, j])), ha="center", va="center",
                    color="white" if pivot.iloc[i, j] > pivot.to_numpy().max() * 0.5 else INK)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)

    ax = axes[1]
    alpha = rows.groupby("alpha", observed=True)["count"].sum().reindex([0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    bars = ax.bar([str(v) for v in alpha.index], alpha.to_numpy(), color=TEAL)
    ax.set_xlabel("Elastic-net alpha")
    ax.set_ylabel("Selections")
    ax.set_title("Regularization mix", color=NAVY, fontweight="bold")
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    for b, v in zip(bars, alpha):
        ax.text(b.get_x() + b.get_width()/2, v + 3, str(int(v)), ha="center", fontsize=8)

    ax = axes[2]
    freq = q["selector_summary"]["coordinate_unique_position_count_frequency"]
    xs = sorted(int(value) for value in freq)
    ys = [int(freq.get(str(x), 0)) for x in xs]
    bars = ax.bar([str(x) for x in xs], ys, color=[GOLD, NAVY, CORAL])
    ax.set_xlabel("Unique candidates across 5 scopes")
    ax.set_ylabel("Coordinates")
    ax.set_title("Selections across target scopes", color=NAVY, fontweight="bold")
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    for b, v in zip(bars, ys):
        ax.text(b.get_x() + b.get_width()/2, v + 1, str(v), ha="center", fontsize=9)
    fig.suptitle("Target-excluded tuning selections",
                 fontsize=13.5, color=NAVY, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, path)
