"""Workspace — deterministic file layout for runs and experiments.

Every run gets its own directory:  sessions/runs/{run_id}/
  model.pth     — latest checkpoint
  best.pth      — best checkpoint by validation loss
  model.opt     — optimizer state
  model.sch     — scheduler state (checkpoint-level)
  model.step_sch — scheduler state (step-level)
  log.csv       — per-step training metrics
  meta.json     — fingerprint, arch, training config, experiment reference
  result.json   — final metrics after completion

Experiments get:  sessions/experiments/{exp_name}/
  config.json   — copy of SweepConfig at start
  summary.csv   — auto-exported from registry after completion
"""

import json
import os
from dataclasses import asdict
from pathlib import Path


class Workspace:
    """Deterministic file paths for registry-based runs."""

    def __init__(self, root: str = 'sessions'):
        self.root = Path(root)

    # ── Run paths ──────────────────────────────────────────

    def run_dir(self, run_id: str) -> Path:
        d = self.root / 'runs' / run_id
        os.makedirs(d, exist_ok=True)
        return d

    def model_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / 'model.pth'

    def best_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / 'best.pth'

    def optimizer_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / 'model.opt'

    def scheduler_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / 'model.sch'

    def step_scheduler_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / 'model.step_sch'

    def log_csv_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / 'log.csv'

    def log_txt_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / 'train.log'

    def meta_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / 'meta.json'

    def result_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / 'result.json'

    # ── Experiment paths ───────────────────────────────────

    def exp_dir(self, exp_name: str) -> Path:
        d = self.root / 'experiments' / exp_name
        os.makedirs(d, exist_ok=True)
        return d

    def config_path(self, exp_name: str) -> Path:
        return self.exp_dir(exp_name) / 'config.json'

    def summary_csv_path(self, exp_name: str) -> Path:
        return self.exp_dir(exp_name) / 'summary.csv'

    # ── Metadata writers ───────────────────────────────────

    def write_meta(self, run_id: str, arch: dict, mc, tc, exp_name: str = '') -> Path:
        """Write meta.json for a run. Returns the path."""
        path = self.meta_path(run_id)
        meta = {
            'run_id': run_id,
            'experiment': exp_name,
            'layer_sizes': arch.get('sizes', []),
            'n_params': arch.get('n_params', 0),
            'model_config': asdict(mc) if hasattr(mc, '__dataclass_fields__') else mc,
            'train_config': asdict(tc) if hasattr(tc, '__dataclass_fields__') else tc,
        }
        with open(path, 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        return path

    def write_result(self, run_id: str, result) -> Path:
        """Write result.json after run completion."""
        path = self.result_path(run_id)
        d = asdict(result) if hasattr(result, '__dataclass_fields__') else vars(result)
        with open(path, 'w') as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        return path

    def write_config(self, exp_name: str, config) -> Path:
        """Write config.json for an experiment."""
        path = self.config_path(exp_name)
        cfg_dict = config.to_dict() if hasattr(config, 'to_dict') else config
        with open(path, 'w') as f:
            json.dump(cfg_dict, f, indent=2, ensure_ascii=False)
        return path
