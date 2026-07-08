#!/usr/bin/env python3
"""enc-dec — unified CLI for autoencoder training & experiments.

Usage:
  enc-dec status              Registry overview (experiments, runs)
  enc-dec status --runs       Recent runs
  enc-dec status --exp NAME   Experiment details
  enc-dec status --run ID     Single run details

  enc-dec train [opts]        Train a single model
  enc-dec sweep run --config CONFIG [opts]
  enc-dec sweep grid --vary n=2,4,6 --solve b --budget 40M [opts]
  enc-dec sweep binary --vary n --range 1 16 [opts]

  enc-dec infer [--gpu]       Interactive inference REPL
  enc-dec overfit [opts]      Overfit test (debug architecture)
  enc-dec lr-find [opts]      LR range test

  enc-dec plot runs           Plot all runs in sessions/runs/
  enc-dec plot noise LABEL1 ID1 LABEL2 ID2   Compare two noise levels

  enc-dec resume [--target 12M]   Resume all runs to target samples
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    if len(sys.argv) < 2:
        _print_help()
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == 'status':
        _run_status(sys.argv[2:])
    elif cmd == 'train':
        _run_train(sys.argv[2:])
    elif cmd == 'sweep':
        _run_sweep(sys.argv[2:])
    elif cmd == 'infer':
        _run_infer(sys.argv[2:])
    elif cmd == 'overfit':
        _run_overfit(sys.argv[2:])
    elif cmd == 'lr-find':
        _run_lr_find(sys.argv[2:])
    elif cmd == 'plot':
        _run_plot(sys.argv[2:])
    elif cmd == 'resume':
        _run_resume(sys.argv[2:])
    elif cmd in ('-h', '--help', 'help'):
        _print_help()
    else:
        print(f"Unknown command: {cmd}")
        _print_help()
        sys.exit(1)


def _print_help():
    print(__doc__)
    print("Run 'enc-dec <command> --help' for per-command options.")


# ── Dispatchers ──────────────────────────────────────────────

def _run_status(argv):
    sys.argv = ['enc-dec status'] + argv
    import cli.status
    cli.status.main()


def _run_train(argv):
    sys.argv = ['enc-dec train'] + argv
    import cli.train
    cli.train.main()


def _run_sweep(argv):
    sys.argv = ['enc-dec sweep'] + argv
    import cli.sweep
    cli.sweep.main()


def _run_infer(argv):
    sys.argv = ['enc-dec infer'] + argv
    import cli.infer
    cli.infer.main()


def _run_overfit(argv):
    sys.argv = ['enc-dec overfit'] + argv
    import cli.overfit_test
    cli.overfit_test.main()


def _run_lr_find(argv):
    sys.argv = ['enc-dec lr-find'] + argv
    import cli.lr_finder
    cli.lr_finder.main()


def _run_plot(argv):
    if not argv:
        print("Usage: enc-dec plot <runs|noise> [opts]")
        sys.exit(1)
    sub = argv[0]

    if sub == 'runs':
        sys.argv = ['enc-dec plot'] + argv
        import scripts.plot_all_runs
        scripts.plot_all_runs.main()
    elif sub == 'noise':
        if len(argv) < 5:
            print("Usage: enc-dec plot noise LABEL1 RUN_ID1 LABEL2 RUN_ID2")
            sys.exit(1)
        sys.argv = ['enc-dec plot'] + argv[1:]
        import scripts.plot_noise_compare
        scripts.plot_noise_compare.main()
    else:
        print(f"Unknown plot subcommand: {sub}")
        print("Usage: enc-dec plot <runs|noise> [opts]")


def _run_resume(argv):
    sys.argv = ['enc-dec resume'] + argv
    import scripts.resume_to_12m
    scripts.resume_to_12m.main()


if __name__ == '__main__':
    main()
