"""Workspace — deterministic file layout with I/O for runs and experiments.

Delegates path resolution to PathResolver. Focused on read/write of
metadata files. mkdir is the only filesystem-mutating call outside of
explicit write_* methods.
"""

import json
import os
from dataclasses import asdict
from pathlib import Path

from orchestration.paths import PathResolver


class Workspace:
    """File I/O for registry-based runs and experiments.

    Path resolution is delegated to PathResolver.
    _find_run_dir handles backward-compat lookup in existing filesystem.
    """

    def __init__(self, root: str = 'sessions'):
        self._paths = PathResolver(root)
        self.root = self._paths.root

    # ── Path accessors (delegated) ─────────────────────────

    def run_dir(self, run_id: str, model_name: str = '') -> Path:
        existing = self._find_run_dir(run_id)
        if existing is not None:
            return existing
        d = self._paths.run_dir(run_id, model_name)
        os.makedirs(d, exist_ok=True)
        return d

    def _find_run_dir(self, run_id: str) -> Path | None:
        """Look up existing run directory by run_id prefix."""
        runs_root = self._paths.root / 'runs'
        if not runs_root.is_dir():
            return None
        prefix = f"{run_id}-"
        for entry in runs_root.iterdir():
            if entry.is_dir() and entry.name.startswith(prefix):
                return entry
        exact = runs_root / run_id
        if exact.is_dir():
            return exact
        return None

    def model_path(self, run_id: str, model_name: str = '') -> Path:
        d = self.run_dir(run_id, model_name)
        return d / 'model.pth'

    def best_path(self, run_id: str, model_name: str = '') -> Path:
        d = self.run_dir(run_id, model_name)
        return d / 'best.pth'

    def optimizer_path(self, run_id: str, model_name: str = '') -> Path:
        d = self.run_dir(run_id, model_name)
        return d / 'model.opt'

    def scheduler_path(self, run_id: str, model_name: str = '') -> Path:
        d = self.run_dir(run_id, model_name)
        return d / 'model.sch'

    def step_scheduler_path(self, run_id: str, model_name: str = '') -> Path:
        d = self.run_dir(run_id, model_name)
        return d / 'model.step_sch'

    def log_csv_path(self, run_id: str, model_name: str = '') -> Path:
        d = self.run_dir(run_id, model_name)
        return d / 'log.csv'

    def log_txt_path(self, run_id: str, model_name: str = '') -> Path:
        d = self.run_dir(run_id, model_name)
        return d / 'train.log'

    def meta_path(self, run_id: str, model_name: str = '') -> Path:
        d = self.run_dir(run_id, model_name)
        return d / 'meta.json'

    def result_path(self, run_id: str, model_name: str = '') -> Path:
        d = self.run_dir(run_id, model_name)
        return d / 'result.json'

    # ── Experiment paths ───────────────────────────────────

    def exp_dir(self, exp_name: str) -> Path:
        d = self._paths.exp_dir(exp_name)
        os.makedirs(d, exist_ok=True)
        return d

    def config_path(self, exp_name: str) -> Path:
        return self.exp_dir(exp_name) / 'config.json'

    def summary_csv_path(self, exp_name: str) -> Path:
        return self.exp_dir(exp_name) / 'summary.csv'

    # ── Metadata writers ───────────────────────────────────

    def write_meta(self, run_id: str, arch: dict, mc, tc, exp_name: str = '',
                   model_name: str = '') -> Path:
        path = self.meta_path(run_id, model_name)
        meta = {
            'run_id': run_id,
            'experiment': exp_name,
            'model_name': model_name,
            'layer_sizes': arch.get('sizes', []),
            'n_params': arch.get('n_params', 0),
            'enc_n': arch.get('enc_n'),
            'dec_n': arch.get('dec_n'),
            'n': arch.get('n'),
            'model_config': asdict(mc) if hasattr(mc, '__dataclass_fields__') else mc,
            'train_config': asdict(tc) if hasattr(tc, '__dataclass_fields__') else tc,
        }
        with open(path, 'w') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        return path

    def write_result(self, run_id: str, result,
                     model_name: str = '') -> Path:
        path = self.result_path(run_id, model_name)
        d = asdict(result) if hasattr(result, '__dataclass_fields__') else vars(result)
        with open(path, 'w') as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        return path

    def write_config(self, exp_name: str, config) -> Path:
        path = self.config_path(exp_name)
        cfg_dict = config.to_dict() if hasattr(config, 'to_dict') else config
        with open(path, 'w') as f:
            json.dump(cfg_dict, f, indent=2, ensure_ascii=False)
        return path
