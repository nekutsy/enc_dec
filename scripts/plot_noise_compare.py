"""Plot train_loss, val_loss, lr for noise=0.025 vs 0.25."""

import csv
import os
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SESSIONS_DIR = Path(__file__).resolve().parent.parent / 'sessions'
OUT_DIR = SESSIONS_DIR / 'plots'


def _find_run_csv(run_id: str):
    """Find log.csv for a run_id, handling new hash-model_name naming."""
    runs_dir = SESSIONS_DIR / 'runs'
    # Exact match
    direct = runs_dir / run_id / 'log.csv'
    if direct.exists():
        return direct
    # Prefix match (hash-model_name)
    for entry in runs_dir.iterdir():
        if entry.is_dir() and not entry.is_symlink() and entry.name.startswith(run_id):
            csv_path = entry / 'log.csv'
            if csv_path.exists():
                return csv_path
    return None


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


def main(argv: list[str] | None = None):
    """Compare two noise-level runs.

    Usage: enc-dec plot noise LABEL1 RUN_ID1 LABEL2 RUN_ID2
    """
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) < 4:
        print("Usage: enc-dec plot noise LABEL1 RUN_ID1 LABEL2 RUN_ID2")
        print("Example: enc-dec plot noise 0.025 bbeda7548d05 0.25 c4cb0acad82f")
        return

    RUNS = {
        argv[0]: argv[1],
        argv[2]: argv[3],
    }
    os.makedirs(OUT_DIR, exist_ok=True)

    runs_data = {}
    for label, run_id in RUNS.items():
        # Resolve run dir (handles new hash-model_name naming)
        csv_path = _find_run_csv(run_id)
        if csv_path is None:
            print(f"  ⚠ Run '{run_id}' not found — skip")
            continue
        runs_data[label] = read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    colors = ['#d62728', '#1f77b4']  # distinct for 0.025 vs 0.25
    markers_at = [8_000_000]

    for ax, metric, title in zip(axes,
                                   ['train_loss', 'val_loss', 'lr'],
                                   ['train_loss', 'val_loss', 'lr']):
        for (label, data), color in zip(runs_data.items(), colors):
            x = data.get('total_samples', [])
            y = data.get(metric, [])
            valid = [(xv, yv) for xv, yv in zip(x, y) if not np.isnan(yv)]
            if not valid:
                continue
            xv, yv = zip(*valid)
            ax.plot(xv, yv, color=color, linewidth=1.2, label=f'noise={label}')

            # Mark 8M resume point
            for m in markers_at:
                if m > xv[0] and m < xv[-1]:
                    idx = min(range(len(xv)), key=lambda i: abs(xv[i] - m))
                    ax.axvline(x=m, color=color, linestyle=':', alpha=0.5, linewidth=0.8)

            # Annotate final value
            fmt = f'{yv[-1]:.6f}' if metric != 'lr' else f'{yv[-1]:.2e}'
            ax.annotate(fmt, xy=(xv[-1], yv[-1]), xytext=(12, 2),
                        textcoords='offset points', fontsize=8, color=color,
                        va='bottom', fontweight='bold')

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('total_samples')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    fig.suptitle('noise=0.025 vs noise=0.25 — 12M samples (→ resume at 8M)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    out_path = OUT_DIR / 'noise_0025_vs_025.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')

    # ── Summary table ──
    print(f'\n{"noise":>6}  {"train@12M":>10}  {"val@12M":>10}  {"val@8M":>10}  {"Δval":>8}')
    print('-' * 52)
    for label, data in runs_data.items():
        x = data['total_samples']
        train = data['train_loss']
        val = data['val_loss']
        train_12 = train[-1]
        val_12 = val[-1]
        # val at 8M
        idx_8m = min(range(len(x)), key=lambda i: abs(x[i] - 8_000_000))
        val_8 = val[idx_8m]
        delta = (val_12 - val_8) / val_8 * 100
        print(f'{label:>6}  {train_12:>10.6f}  {val_12:>10.6f}  {val_8:>10.6f}  {delta:>+7.1f}%')


if __name__ == '__main__':
    main(sys.argv[1:])
