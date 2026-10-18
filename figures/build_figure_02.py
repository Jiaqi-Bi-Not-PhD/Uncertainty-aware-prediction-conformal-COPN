"""Plot the reported prediction-set metrics from an authorized private aggregate CSV. No data are bundled."""
from pathlib import Path
from copn_analysis.private_io import private_input, private_output, private_directory
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
NAVY = '#173B5E'
INK = '#1F2933'
GRID = '#D8DEE4'
SET_POLICY_COLORS = {'A2': '#94B5D8', 'A3': '#ACD2C7', 'A5': '#D57B70'}
SET_CLASS_COLORS = {'PD': '#94B5D8', 'AP': '#ACD2C7', 'HC': '#D57B70'}

def configure() -> None:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.titlesize': 11, 'axes.labelsize': 9.5, 'axes.edgecolor': '#AEB8C2', 'axes.linewidth': 0.8, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'figure.dpi': 160, 'savefig.dpi': 220, 'savefig.bbox': 'tight', 'savefig.facecolor': 'white'})

def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def figure_efficiency_safety(stage5: pd.DataFrame, path: Path, *, additional_paths: tuple[Path, ...]=()) -> None:
    data = stage5.loc[(stage5['budget'] == 3) & (stage5['stratum'] == 'overall') & (stage5['level'] == 'all') & stage5['method_id'].isin(['A2', 'A3', 'A5'])].set_index('method_id').loc[['A2', 'A3', 'A5']]
    labels = ['A2 union', 'A3 aggregate', 'A5 constrained']
    colors = [SET_POLICY_COLORS[method] for method in data.index]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))
    for ax, column, title, ylim in [(axes[0, 0], 'mean_set_size', 'Mean core set size (lower is narrower)', (0, 3.0)), (axes[0, 1], 'mean_SIG', 'Mean set information gain (bits)', (0, 1.0)), (axes[1, 0], 'full_set_rate', 'Full three label set rate', (0, 0.38))]:
        values = data[column].to_numpy(dtype=float)
        bars = ax.bar(labels, values, color=colors, width=0.66)
        ax.set_ylim(*ylim)
        ax.grid(axis='y', color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_title(title, color=NAVY, fontweight='bold')
        ax.tick_params(axis='x', rotation=0)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + ylim[1] * 0.025, f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    ax = axes[1, 1]
    x = np.arange(3)
    width = 0.22
    class_cols = [('PD', 'core_coverage_PD'), ('AP', 'core_coverage_AP'), ('HC', 'core_coverage_HC')]
    for j, (class_id, column) in enumerate(class_cols):
        ax.bar(x + (j - 1) * width, data[column].to_numpy(dtype=float), width=width, label=class_id, color=SET_CLASS_COLORS[class_id])
    ax.axhline(0.9, color='#7A8793', linewidth=1, linestyle='--')
    ax.axhline(0.95, color='#56616C', linewidth=1, linestyle=(0, (2, 2)))
    ax.set_ylim(0.78, 1.065)
    ax.set_yticks([0.8, 0.85, 0.9, 0.95, 1.0])
    ax.set_xticks(x, labels)
    ax.tick_params(axis='x', rotation=0)
    ax.set_title('Class conditional core coverage (safety check)', color=NAVY, fontweight='bold')
    ax.grid(axis='y', color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc='upper left')
    harmful = data['dangerous_HC_singleton_count'].astype(int).tolist()
    ax.text(0.02, 0.08, f'Dangerous HC singleton rows: {harmful[0]} / {harmful[1]} / {harmful[2]}', transform=ax.transAxes, fontsize=8.1, color=INK, fontweight='bold', bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.82, 'pad': 1.5})
    fig.suptitle('Prediction set coverage and efficiency', fontsize=14.5, color=NAVY, fontweight='bold', y=1.01)
    fig.tight_layout()
    for additional_path in additional_paths:
        fig.savefig(additional_path, bbox_inches='tight', facecolor='white')
    save(fig, path)

def build(data_path: Path, output_dir: Path) -> None:
    data_path = private_input(data_path)
    output_dir = private_directory(output_dir)
    configure()
    plt.rcParams.update({'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    data = pd.read_csv(data_path, float_precision='round_trip')
    figure_efficiency_safety(data, output_dir / 'figure_02_labels.pdf', additional_paths=(output_dir / 'figure_02.png', output_dir / 'figure_02.svg'))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.data, args.output_dir)
