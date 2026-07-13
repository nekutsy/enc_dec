"""Sweep — grid and binary search orchestration with pluggable strategies.

Replaces cli/sweep.py SweepRunner. Uses Registry for deduplication across
experiments and pluggable SweepStrategy for extensibility.
"""

import json
import sys
from typing import Protocol

from experiment.context import setup_runtime
from utils import cuda_safe_cleanup, gpu_health_check
from model.architecture import resolve_architecture

from registry.db import Registry
from registry.fingerprint import arch_fingerprint, training_hash
from orchestration.workspace import Workspace
from orchestration.run import Run


# ── Strategy protocol ──────────────────────────────────────

class SweepStrategy(Protocol):
    """Plugin protocol for sweep strategies. Grid, Binary, Bayesian implement."""

    @property
    def vary_name(self) -> str: ...

    @property
    def vary_values(self) -> list: ...

    def next_candidate(self, completed: dict) -> tuple | None:
        """Return next (vary_name, vary_value) or None if done.

        completed: {vary_value: final_train_loss} — already-trained results.
        """
        ...


class GridStrategy:
    """Iterate over values in order, skip already-completed."""

    def __init__(self, values: list, vary_name: str):
        self._values = list(values)
        self.vary_name = vary_name

    @property
    def vary_values(self) -> list:
        return self._values

    def _key(self, v):
        """Normalize sweep value to a hashable key."""
        return tuple(v) if isinstance(v, list) else v

    def next_candidate(self, completed: dict) -> tuple | None:
        for v in self._values:
            if self._key(v) not in completed:
                return (self.vary_name, v)
        return None


class BinaryStrategy:
    """Binary search over a parameter range.

    Given [lo, hi] range, probes boundaries first, then bisects.
    """

    def __init__(self, values: list, vary_name: str):
        lo, hi = values[0], values[1]
        self.vary_name = vary_name
        self._lo = lo
        self._hi = hi
        self._is_float = isinstance(lo, float) or isinstance(hi, float)

        if self._is_float:
            def _decimals(v):
                s = f"{v:.10f}".rstrip('0')
                if '.' in s:
                    return len(s.split('.')[1])
                return 0
            self._step = 10 ** (-max(_decimals(lo), _decimals(hi)))
        else:
            self._step = 1

        self._iteration = 0
        self._boundaries_done = False
        self._converged = False

    @property
    def vary_values(self) -> list:
        return [self._lo, self._hi]

    def _to_step(self, v):
        """Round to the allowed step."""
        if self._is_float:
            return round(round(v / self._step) * self._step, 10)
        return int(round(v / self._step) * self._step)

    def next_candidate(self, completed: dict) -> tuple | None:
        if self._converged:
            return None

        # Phase 1: probe boundaries
        if not self._boundaries_done:
            for boundary in [self._lo, self._hi]:
                b = self._to_step(boundary)
                if b not in completed:
                    return (self.vary_name, b)
            self._boundaries_done = True

        self._iteration += 1

        valid = {k: v for k, v in completed.items() if v < 1e8}
        if len(valid) < 2:
            return None

        sorted_items = sorted(valid.items(), key=lambda x: x[1])
        best, best_val = sorted_items[0]
        second = sorted_items[1][0]

        left_neighbor = self._to_step(best - self._step)
        right_neighbor = self._to_step(best + self._step)
        has_left = best <= self._lo or left_neighbor in completed
        has_right = best >= self._hi or right_neighbor in completed

        if has_left and has_right:
            self._converged = True
            return None

        if abs(self._to_step(best - second)) <= self._step:
            if not has_left and left_neighbor not in completed:
                return (self.vary_name, left_neighbor)
            if not has_right and right_neighbor not in completed:
                return (self.vary_name, right_neighbor)
            self._converged = True
            return None

        mid = self._to_step((best + second) / 2)
        if mid in completed:
            lo_b, hi_b = min(best, second), max(best, second)
            candidate = lo_b + self._step
            while candidate < hi_b:
                c = self._to_step(candidate)
                if c not in completed:
                    return (self.vary_name, c)
                candidate += self._step
            if not has_left and left_neighbor not in completed:
                return (self.vary_name, left_neighbor)
            if not has_right and right_neighbor not in completed:
                return (self.vary_name, right_neighbor)
            self._converged = True
            return None

        return (self.vary_name, mid)


def _make_strategy(strategy: str, values: list, vary_name: str) -> SweepStrategy:
    if strategy == 'binary':
        return BinaryStrategy(values, vary_name)
    return GridStrategy(values, vary_name)


# ── Sweep ──────────────────────────────────────────────────

class Sweep:
    """Orchestrates a sweep using Registry for deduplication."""

    def __init__(self, config, registry: Registry | None = None,
                 workspace: Workspace | None = None):
        self.cfg = config
        self.mc = config.model
        self.tc = config.training
        self.sc = config.sweep
        self.oc = config.output
        self.registry = registry or Registry()
        self.ws = workspace or Workspace()

        self.exp_name = config.name
        self.strategy = _make_strategy(
            self.sc.strategy, self.sc.values, self.sc.vary)

        self._results: dict = {}

    # ── Main entry ─────────────────────────────────────────

    def run(self, no_val: bool = True) -> dict:
        """Run the sweep. Returns {vary_value: final_train_loss}.

        With registry deduplication, identical runs across experiments
        are automatically skipped.
        """
        cfg = self.cfg

        # ── Check GPU ──
        if not gpu_health_check():
            print('⚠ GPU not available — check nvidia-smi')
            return {}

        # ── Create experiment ──
        exp_id = self.registry.create_experiment(cfg)
        self.ws.write_config(self.exp_name, cfg)

        # ── Runtime ──
        runtime = setup_runtime(cfg.output, use_tf32=cfg.training.use_tf32)

        print(f'Sweep: {cfg.name}  ({cfg.sweep.strategy} over {cfg.sweep.vary})')
        print(f'  seq_len={cfg.model.seq_len}  target={cfg.training.target_samples // 1e6:.0f}M samples  '
              f'budget={cfg.sweep.budget // 1e6 if cfg.sweep.budget else "auto"}M')
        print()

        # ── Precompute architectures ──
        all_values = self.strategy.vary_values
        print(f'{"vary":>8}  {"enc_n":>5}  {"dec_n":>5}  {"b":>7}  {"params":>10}')
        for v in all_values:
            try:
                arch = resolve_architecture(v, self.sc.vary, cfg)
                fp = arch_fingerprint(arch['sizes'], self.mc)
                th = training_hash(self.tc)
                done = self.registry.get_completed_run(fp, th, min_samples=self.tc.target_samples)
                tag = ' ✓' if done else ''
                enc_n = arch.get('enc_n', arch.get('n', '?'))
                dec_n = arch.get('dec_n', arch.get('n', '?'))
                print(f'{str(v):>8}  {enc_n:>5}  {dec_n:>5}  {arch["b"]:>7.4g}  {arch["n_params"]:>10,}{tag}')
            except Exception as e:
                print(f'{str(v):>8}  ERROR: {e}')
        print()

        # ── Adaptive loop ──
        completed: dict = {}
        while True:
            candidate = self.strategy.next_candidate(completed)
            if candidate is None:
                break

            vary_name, vary_value = candidate
            print(f'{"─" * 50}')
            print(f'[{vary_name}={vary_value}]')

            try:
                arch = resolve_architecture(vary_value, vary_name, cfg)
            except Exception as e:
                print(f'  ⚠ resolve error: {e}')
                completed[self.strategy._key(vary_value)] = 1e9
                continue

            run, created = Run.find_or_create(
                arch, self.mc, self.tc, self.registry, self.ws, self.exp_name)

            if not created and run is not None:
                # Already fully done at target_samples or higher
                rd = self.registry.get_run(run.run_id)
                if rd and rd.get('status') == 'done' and rd.get('total_samples', 0) >= self.tc.target_samples:
                    val = rd.get('final_train_loss')
                    print(f'  already done (loss={val})')
                    self.registry.link_run(exp_id, run.run_id, vary_value)
                    completed[self.strategy._key(vary_value)] = val if val is not None else 1e9
                    continue
                elif rd and rd.get('status') == 'done':
                    stale = rd.get('total_samples', 0)
                    print(f'  done at {stale:,} samples < target {self.tc.target_samples:,} — retraining')
                else:
                    print(f'  run exists but not done (status={rd.get("status")}) — retrying')

            result = run.execute(runtime, no_val=no_val)
            cuda_safe_cleanup()

            self.registry.link_run(exp_id, run.run_id, vary_value)

            if result.final_train_loss is not None:
                completed[self.strategy._key(vary_value)] = result.final_train_loss
            else:
                completed[self.strategy._key(vary_value)] = 1e9

            print()

        # ── Finish ──
        self.registry.finish_experiment(exp_id)
        self.registry.export_csv(
            self.exp_name,
            str(self.ws.summary_csv_path(self.exp_name)),
        )

        self._results = {k: v for k, v in completed.items() if v < 1e8}
        self._print_summary()
        return self._results

    def _print_summary(self):
        if not self._results:
            return
        print(f'{"=" * 55}')
        print(f'RESULTS: {self.exp_name}')
        print(f'{"vary":>8}  {"train_loss":>10}')
        print('-' * 25)
        for v in sorted(self._results, key=lambda k: str(k)):
            print(f'{str(v):>8}  {self._results[v]:>10.6f}')
        best = min(self._results, key=self._results.get)
        print(f'\n  ★ best: {self.cfg.sweep.vary}={best}  train={self._results[best]:.6f}')
        print(f'{"=" * 55}')
