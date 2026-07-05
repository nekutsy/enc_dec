"""Registry status viewer — list experiments, runs, summaries.

Usage:
  python cli/status.py                    # overview
  python cli/status.py --experiments      # list all experiments
  python cli/status.py --runs             # recent runs
  python cli/status.py --exp noise_sweep  # experiment details
  python cli/status.py --run abc123       # single run details
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from registry import Registry


def _fmt_loss(v):
    if v is None:
        return '     —'
    return f'{v:.6f}'


def _fmt_params(n):
    if n is None:
        return '    —'
    if n >= 1_000_000:
        return f'{n / 1_000_000:.1f}M'
    return f'{n:,}'


def _fmt_samples(n):
    if n is None:
        return '       —'
    if n >= 1_000_000:
        return f'{n / 1_000_000:.1f}M'
    return f'{n:,}'


def _fmt_duration(s):
    if s is None:
        return '    —'
    if s < 60:
        return f'{s:.0f}s'
    if s < 3600:
        return f'{s / 60:.1f}m'
    return f'{s / 3600:.1f}h'


def cmd_overview(reg: Registry):
    s = reg.summary()
    print(f"Registry: {s['db_path']}")
    print(f"  Experiments: {s['total_experiments']}")
    print(f"  Runs: {s['done_runs']} done / {s['running_runs']} running / "
          f"{s['total_runs']} total\n")

    # Recent done runs
    runs = reg.list_recent_runs(limit=10, status='done')
    if runs:
        print(f"{'run_id':>12}  {'model':>30}  {'loss':>10}  {'samples':>10}  {'dur':>8}")
        print('-' * 80)
        for r in runs:
            print(f"{r['id']:>12}  {r['model_name']:>30}  "
                  f"{_fmt_loss(r['final_train_loss']):>10}  "
                  f"{_fmt_samples(r['total_samples']):>10}  "
                  f"{_fmt_duration(r['duration_seconds']):>8}")


def cmd_experiments(reg: Registry, limit: int = 50):
    exps = reg.list_experiments(limit=limit)
    if not exps:
        print("No experiments yet.")
        return
    print(f"{'name':>40}  {'strategy':>8}  {'vary':>16}  {'status':>10}")
    print('-' * 85)
    for e in exps:
        print(f"{e['name']:>40}  {e['strategy']:>8}  "
              f"{e['vary_param']:>16}  {e['status']:>10}")


def cmd_experiment_detail(reg: Registry, exp_name: str):
    results = reg.get_experiment_results(exp_name)
    if not results:
        print(f"No results for experiment '{exp_name}'.")
        return
    print(f"Experiment: {exp_name}")
    print(f"  Strategy: {results[0]['strategy']}  "
          f"Vary: {results[0]['vary_param']}\n")
    print(f"{'vary_value':>12}  {'run_id':>12}  {'loss':>10}  "
          f"{'samples':>10}  {'dur':>8}  {'status':>10}")
    print('-' * 75)
    for r in results:
        print(f"{r['vary_value']:>12}  {r['run_id']:>12}  "
              f"{_fmt_loss(r['final_train_loss']):>10}  "
              f"{_fmt_samples(r['total_samples']):>10}  "
              f"{_fmt_duration(r['duration_seconds']):>8}  "
              f"{r['run_status']:>10}")


def cmd_run_detail(reg: Registry, run_id: str):
    run = reg.get_run(run_id)
    if not run:
        print(f"Run '{run_id}' not found.")
        return
    print(f"Run: {run_id}")
    print(f"  Model: {run['model_name']}")
    print(f"  Status: {run['status']}")
    print(f"  Architecture FP: {run['architecture_fp']}")
    print(f"  Training Hash: {run['training_hash']}")
    print(f"  Loss: {_fmt_loss(run['final_train_loss'])}")
    print(f"  Samples: {_fmt_samples(run['total_samples'])}")
    print(f"  Duration: {_fmt_duration(run['duration_seconds'])}")
    if run.get('error_message'):
        print(f"  Error: {run['error_message']}")


def main():
    parser = argparse.ArgumentParser(description='Registry status viewer')
    parser.add_argument('--experiments', '-e', action='store_true',
                        help='List all experiments')
    parser.add_argument('--runs', '-r', action='store_true',
                        help='List recent runs')
    parser.add_argument('--exp', default=None, help='Experiment details by name')
    parser.add_argument('--run', default=None, help='Run details by id')
    parser.add_argument('--limit', type=int, default=50, help='Max items')
    args = parser.parse_args()

    reg = Registry()

    if args.exp:
        cmd_experiment_detail(reg, args.exp)
    elif args.run:
        cmd_run_detail(reg, args.run)
    elif args.runs:
        runs = reg.list_recent_runs(limit=args.limit)
        if not runs:
            print("No runs yet.")
            return
        print(f"{'run_id':>12}  {'model':>30}  {'loss':>10}  {'samples':>10}  "
              f"{'status':>10}  {'dur':>8}")
        print('-' * 90)
        for r in runs:
            print(f"{r['id']:>12}  {r['model_name']:>30}  "
                  f"{_fmt_loss(r['final_train_loss']):>10}  "
                  f"{_fmt_samples(r['total_samples']):>10}  "
                  f"{r['status']:>10}  "
                  f"{_fmt_duration(r['duration_seconds']):>8}")
    elif args.experiments:
        cmd_experiments(reg, args.limit)
    else:
        cmd_overview(reg)


if __name__ == '__main__':
    main()
