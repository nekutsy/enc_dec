"""PathResolver — pure path resolution without I/O.

Deterministic: same inputs → same Path objects. No filesystem access.
Workspace delegates to PathResolver for path building and adds I/O.
"""

from __future__ import annotations

from pathlib import Path


class PathResolver:
    """Pure functions: convert run_id, model_name, exp_name → Path objects.

    All methods are side-effect-free. No directories created, no files read.
    """

    def __init__(self, root: str = 'sessions'):
        self.root = Path(root)

    # ── Run paths ──────────────────────────────────────────

    def run_dir(self, run_id: str, model_name: str = '') -> Path:
        name = f"{run_id}-{model_name}" if model_name else run_id
        return self.root / 'runs' / name

    def model_path(self, run_id: str, model_name: str = '') -> Path:
        return self.run_dir(run_id, model_name) / 'model.pth'

    def best_path(self, run_id: str, model_name: str = '') -> Path:
        return self.run_dir(run_id, model_name) / 'best.pth'

    def optimizer_path(self, run_id: str, model_name: str = '') -> Path:
        return self.run_dir(run_id, model_name) / 'model.opt'

    def scheduler_path(self, run_id: str, model_name: str = '') -> Path:
        return self.run_dir(run_id, model_name) / 'model.sch'

    def step_scheduler_path(self, run_id: str, model_name: str = '') -> Path:
        return self.run_dir(run_id, model_name) / 'model.step_sch'

    def log_csv_path(self, run_id: str, model_name: str = '') -> Path:
        return self.run_dir(run_id, model_name) / 'log.csv'

    def log_txt_path(self, run_id: str, model_name: str = '') -> Path:
        return self.run_dir(run_id, model_name) / 'train.log'

    def meta_path(self, run_id: str, model_name: str = '') -> Path:
        return self.run_dir(run_id, model_name) / 'meta.json'

    def result_path(self, run_id: str, model_name: str = '') -> Path:
        return self.run_dir(run_id, model_name) / 'result.json'

    # ── Experiment paths ───────────────────────────────────

    def exp_dir(self, exp_name: str) -> Path:
        return self.root / 'experiments' / exp_name

    def config_path(self, exp_name: str) -> Path:
        return self.exp_dir(exp_name) / 'config.json'

    def summary_csv_path(self, exp_name: str) -> Path:
        return self.exp_dir(exp_name) / 'summary.csv'
