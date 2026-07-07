"""Resume all runs in sessions/runs/ to 12M samples.

Reuses existing checkpoints (weights, optimizer, schedulers) — no re-init.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from experiment.config import ModelConfig, TrainConfig
from experiment.context import setup_runtime, OutputConfig
from orchestration import Run, Workspace
from registry import Registry
from logger import LoggerConfig
from utils import gpu_health_check, cuda_safe_cleanup

TARGET = 12_000_000


def _extract_run_id(dir_name: str) -> str:
    """Extract run_id from dir name: pure hash or hash-model_name."""
    return dir_name[:12] if len(dir_name) >= 12 else dir_name


def main():
    ws = Workspace()
    reg = Registry()
    runtime = setup_runtime(OutputConfig(device='auto'))

    if not gpu_health_check():
        print('GPU not available')
        sys.exit(1)

    dir_names = sorted(os.listdir(ws.root / 'runs'))

    for dir_name in dir_names:
        run_dir = ws.root / 'runs' / dir_name
        if not run_dir.is_dir():
            continue
        run_id = _extract_run_id(dir_name)
        meta_path = ws.meta_path(run_id)
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        old_target = meta['train_config'].get('target_samples', 0)
        if old_target >= TARGET:
            print(f'[{run_id[:6]}] already at {old_target:,} — skip')
            continue

        # Reconstruct configs from meta
        mc = ModelConfig(**meta['model_config'])
        tc = TrainConfig(**meta['train_config'])
        tc.target_samples = TARGET

        arch = {
            'sizes': meta['layer_sizes'],
            'n_params': meta['n_params'],
            'n': len([s for s in meta['layer_sizes']
                      if s == meta['layer_sizes'][1]]) // 2,  # rough but works
            'b': round(meta['layer_sizes'][1] / meta['layer_sizes'][0], 6),
            'hidden_dim': meta['layer_sizes'][1],
        }

        model_name = meta.get('model_name', '')
        exp_name = meta.get('experiment', '')

        print(f'\n{"=" * 55}')
        print(f'[{run_id[:6]}] target: {old_target:,} → {TARGET:,}  noise={tc.noise_prob}')

        # Bypass find_or_create — use existing run_id directly
        run = Run(run_id, arch, mc, tc, reg, ws,
                  exp_name=exp_name, model_name=model_name)

        try:
            result = run.execute(runtime, no_val=True)
            cuda_safe_cleanup()

            if result.final_train_loss is not None:
                print(f'  ✅ done: train={result.final_train_loss:.6f} '
                      f'val={result.final_val_loss} samples={result.total_samples:,}')
            else:
                print(f'  ❌ failed: {result.status}')

        except Exception as e:
            print(f'  ❌ error: {e}')
            cuda_safe_cleanup()

    print(f'\n{"=" * 55}')
    print('All done.')


if __name__ == '__main__':
    main()
