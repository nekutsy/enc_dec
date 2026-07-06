"""Plot train_loss, val_loss, lr for current runs (sessions/runs/)."""

import csv
import os
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SESSIONS_DIR = Path(__file__).resolve().parent.parent / 'sessions'
RUNS_DIR = SESSIONS_DIR / 'runs'
OUT_DIR = SESSIONS_DIR / 'plots'


def read_csv(csv_path):
    data = defaultdict(list)
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                try:
                    data[k].append(float(v))
                except (ValueError, TypeError):
                    data[k].append(np.nan)
    return data


def plot_run(run_id, csv_path, meta_path, out_dir):
    data = read_csv(csv_path)

    # Load meta info
    label = run_id
    noise = '?'
    if meta_path.exists():
        import json
        with open(meta_path) as f:
            meta = json.load(f)
        tc = meta.get('train_config', {})
        noise = tc.get('noise_prob', '?')
        label = f'{run_id[:6]} (noise={noise})'

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{label}', fontsize=12, fontweight='bold')

    metrics = ['train_loss', 'val_loss', 'lr']
    colors = ['#1f77b4', '#d62728', '#2ca02c']

    x = data.get('total_samples', [])

    for ax, metric, color in zip(axes, metrics, colors):
        y = data.get(metric, [])
        valid = [(xv, yv) for xv, yv in zip(x, y) if not np.isnan(yv)]
        if not valid:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, color='gray')
            ax.set_title(metric)
            continue
        xv, yv = zip(*valid)
        ax.plot(xv, yv, color=color, linewidth=1.2)
        ax.set_title(metric)
        ax.set_xlabel('total_samples')
        ax.grid(True, alpha=0.3)

        # Final value annotation
        ax.annotate(f'{yv[-1]:.6f}' if metric != 'lr' else f'{yv[-1]:.2e}',
                    xy=(xv[-1], yv[-1]), xytext=(10, 0),
                    textcoords='offset points', fontsize=8, color=color,
                    va='center')

    plt.tight_layout()
    out_path = out_dir / f'{run_id[:6]}.png'
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  {out_path}')


def main():
    out_dir = OUT_DIR
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    runs = sorted(os.listdir(RUNS_DIR))
    print(f'Plotting {len(runs)} runs:\n')

    for run_id in runs:
        csv_path = RUNS_DIR / run_id / 'log.csv'
        meta_path = RUNS_DIR / run_id / 'meta.json'
        if csv_path.exists():
            plot_run(run_id, csv_path, meta_path, out_dir)

    print(f'\nDone. Plots in: {out_dir}')


if __name__ == '__main__':
    main()
