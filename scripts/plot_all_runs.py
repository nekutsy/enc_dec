"""Plot train_loss, val_loss, lr for all runs grouped by experiment."""

import csv
import os
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SESSIONS_DIR = Path(__file__).resolve().parent.parent / 'sessions'
OUT_DIR = SESSIONS_DIR / 'plots'


def read_csv(csv_path):
    """Read log.csv → {col: [values]}."""
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


def plot_group(group_name, runs, out_dir):
    """Plot train_loss, val_loss, lr for a group of runs."""
    n = len(runs)
    if n == 0:
        return

    cols = 3
    fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 5))
    fig.suptitle(group_name, fontsize=13, fontweight='bold', y=0.98)

    metrics = ['train_loss', 'val_loss', 'lr']
    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 1)))

    for ax, metric in zip(axes, metrics):
        for i, (label, csv_path) in enumerate(runs):
            data = read_csv(csv_path)
            x = data.get('total_samples', data.get('epoch', []))
            y = data.get(metric, [])

            # Drop NaN values
            valid = [(xv, yv) for xv, yv in zip(x, y) if not np.isnan(yv)]
            if not valid:
                continue
            x_valid, y_valid = zip(*valid)
            ax.plot(x_valid, y_valid, color=colors[i % len(colors)],
                    linewidth=1.0, alpha=0.85, label=label)

        ax.set_title(metric)
        ax.set_xlabel('total_samples' if 'total_samples' in data else 'epoch')
        ax.grid(True, alpha=0.3)
        if n <= 10:
            ax.legend(fontsize=7, loc='best')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = out_dir / f'{group_name}.png'
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out_path}')


def collect_runs():
    """Collect all log.csv files grouped by experiment/parent dir."""
    groups = defaultdict(list)

    # Scan sessions dir for all log.csv
    for csv_path in SESSIONS_DIR.rglob('log.csv'):
        rel = csv_path.relative_to(SESSIONS_DIR)

        # Determine group name from path
        parts = rel.parts

        if parts[0] == 'runs':
            # Current runs — group all together as "current"
            run_id = parts[1]
            label = run_id[:6]
            groups['current_runs'].append((label, csv_path))
        elif parts[0] == 'archive':
            # e.g. archive/csv_logs/rect_sweep/sweep_n2/log.csv
            if len(parts) >= 4:
                group = parts[2]  # rect_sweep, interleaved_sweep, etc.
                subdir = parts[3]  # sweep_n2, sweep_noise_prob0.1, etc.
                groups[group].append((subdir, csv_path))
            elif len(parts) == 3:
                # e.g. archive/csv_logs/n8_rect_bn160_50M/n8_s128/log.csv
                group = parts[2]
                subdir = parts[3] if len(parts) > 3 else 'run'
                groups[group].append((subdir, csv_path))

    return groups


def main():
    out_dir = OUT_DIR
    # Clean old plots
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    groups = collect_runs()
    print(f'Found {sum(len(v) for v in groups.values())} runs in {len(groups)} groups\n')

    for group_name in sorted(groups):
        runs = groups[group_name]
        print(f'  {group_name} ({len(runs)} runs)')
        plot_group(group_name, runs, out_dir)

    print(f'\nDone. Plots in: {out_dir}')


if __name__ == '__main__':
    main()
