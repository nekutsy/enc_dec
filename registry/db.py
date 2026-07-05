"""Registry — SQLite-backed experiment tracking.

Single source of truth for all experiments, architectures, and runs.
Uses JSON blobs for config storage → new fields don't need schema migrations.

Usage:
    reg = Registry('sessions/registry.db')
    exp_id = reg.create_experiment(config)
    run_id = reg.find_or_create_run(arch_fp, train_hash, ...)
    reg.start_run(run_id)
    # ... train ...
    reg.finish_run(run_id, RunResult(...))
    reg.finish_experiment(exp_id)
"""

import json
import os
import sqlite3
import time as time_mod
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any

from registry.schema import DDL, SCHEMA_VERSION


# ── Lightweight result type ─────────────────────────────────

@dataclass
class RunResult:
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    total_samples: int = 0
    duration_seconds: float = 0.0
    status: str = 'done'
    error_message: str | None = None


# ── Registry class ──────────────────────────────────────────

class Registry:
    """Thread-safe SQLite registry for experiment tracking.

    All writes go through a single connection; sqlite3 module-level lock
    handles thread safety. For multi-process use, WAL mode is enabled.
    """

    def __init__(self, db_path: str = 'sessions/registry.db'):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
        self._ensure_schema()

    # ── Connection management ───────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(DDL)
            conn.execute(
                "INSERT OR IGNORE INTO _meta(key, value) VALUES (?, ?)",
                ('ddl_version', str(SCHEMA_VERSION)),
            )

    # ── Helpers ─────────────────────────────────────────────

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _generate_id() -> str:
        import uuid
        return uuid.uuid4().hex[:12]

    # ── Experiments ─────────────────────────────────────────

    def create_experiment(self, config: Any) -> str:
        """Create a new experiment record. Returns experiment id."""
        exp_id = self._generate_id()
        cfg_dict = config.to_dict() if hasattr(config, 'to_dict') else config
        s = getattr(config, 'sweep', None)
        strategy = getattr(s, 'strategy', 'grid') if s else 'grid'
        vary_param = getattr(s, 'vary', '') if s else ''
        vary_values = getattr(s, 'values', []) if s else []

        with self._connect() as conn:
            conn.execute("""
                INSERT INTO experiments (id, name, strategy, vary_param,
                    vary_values_json, config_json, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
            """, (
                exp_id,
                cfg_dict.get('name', 'unknown'),
                strategy,
                vary_param,
                json.dumps(vary_values),
                json.dumps(cfg_dict),
                self._now(),
            ))
        return exp_id

    def finish_experiment(self, exp_id: str, status: str = 'done') -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE experiments SET status = ?, finished_at = ? WHERE id = ?",
                (status, self._now(), exp_id),
            )

    def get_experiment(self, exp_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (exp_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_experiments(self, status: str | None = None,
                         limit: int = 50) -> list[dict]:
        query = "SELECT * FROM experiments"
        params: tuple = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY created_at DESC LIMIT ?"
        params = params + (limit,)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ── Architectures ───────────────────────────────────────

    def register_architecture(self, fingerprint: str, n_params: int,
                               seq_len: int, shape: str,
                               config_json: str) -> bool:
        """Register an architecture. Returns True if new, False if existed."""
        with self._connect() as conn:
            try:
                conn.execute("""
                    INSERT INTO architectures (fingerprint, n_params, seq_len,
                        shape, config_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (fingerprint, n_params, seq_len, shape, config_json,
                      self._now()))
                return True
            except sqlite3.IntegrityError:
                return False

    def get_architecture(self, fp: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM architectures WHERE fingerprint = ?", (fp,)
            ).fetchone()
        return dict(row) if row else None

    # ── Runs ────────────────────────────────────────────────

    def find_or_create_run(self, architecture_fp: str, training_hash_val: str,
                           model_name: str, arch_config_json: str,
                           train_config_json: str) -> tuple[str, bool]:
        """Find existing run or create new. Returns (run_id, created_new)."""
        with self._connect() as conn:
            # Check UNIQUE constraint first
            row = conn.execute(
                "SELECT id, status FROM runs WHERE architecture_fp = ? AND training_hash = ?",
                (architecture_fp, training_hash_val),
            ).fetchone()

            if row:
                return row['id'], False

            run_id = self._generate_id()
            conn.execute("""
                INSERT INTO runs (id, architecture_fp, training_hash, model_name,
                    arch_config_json, train_config_json, status)
                VALUES (?, ?, ?, ?, ?, ?, 'pending')
            """, (run_id, architecture_fp, training_hash_val, model_name,
                  arch_config_json, train_config_json))
            return run_id, True

    def start_run(self, run_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET status = 'running', started_at = ? WHERE id = ?",
                (self._now(), run_id),
            )

    def finish_run(self, run_id: str, result: RunResult) -> None:
        with self._connect() as conn:
            conn.execute("""
                UPDATE runs SET
                    final_train_loss = ?, final_val_loss = ?,
                    total_samples = ?, duration_seconds = ?,
                    status = ?, finished_at = ?, error_message = ?
                WHERE id = ?
            """, (
                result.final_train_loss, result.final_val_loss,
                result.total_samples, result.duration_seconds,
                result.status, self._now(), result.error_message,
                run_id,
            ))

    def get_run(self, run_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_completed_run(self, arch_fp: str, train_hash: str) -> dict | None:
        """Return a completed run if one exists, else None."""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT * FROM runs
                WHERE architecture_fp = ? AND training_hash = ? AND status = 'done'
            """, (arch_fp, train_hash)).fetchone()
        return dict(row) if row else None

    def list_recent_runs(self, limit: int = 20,
                          status: str | None = None) -> list[dict]:
        query = "SELECT * FROM run_summary"
        params: tuple = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY total_samples DESC LIMIT ?"
        params = params + (limit,)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_best_loss(self, arch_fp: str) -> float | None:
        """Best final_train_loss for a given architecture fingerprint."""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT final_train_loss FROM runs
                WHERE architecture_fp = ? AND status = 'done'
                ORDER BY final_train_loss ASC LIMIT 1
            """, (arch_fp,)).fetchone()
        return row['final_train_loss'] if row else None

    # ── Experiment ↔ Run linking ────────────────────────────

    def link_run(self, exp_id: str, run_id: str, vary_value: Any) -> None:
        with self._connect() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO experiment_runs (experiment_id, run_id, vary_value)
                VALUES (?, ?, ?)
            """, (exp_id, run_id, str(vary_value)))

    def get_experiment_runs(self, exp_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM experiment_results WHERE experiment =
                    (SELECT name FROM experiments WHERE id = ?)
                ORDER BY vary_value
            """, (exp_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_experiment_results(self, exp_name: str) -> list[dict]:
        """Get all run results for an experiment by name."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT * FROM experiment_results
                WHERE experiment = ?
                ORDER BY vary_value
            """, (exp_name,)).fetchall()
        return [dict(r) for r in rows]

    # ── Export ──────────────────────────────────────────────

    def export_csv(self, exp_name: str, path: str) -> str:
        """Export experiment results to CSV. Returns the path."""
        results = self.get_experiment_results(exp_name)
        if not results:
            return path

        import csv
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        fieldnames = list(results[0].keys())
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        return path

    # ── Summary ─────────────────────────────────────────────

    def summary(self) -> dict:
        """Quick summary: run counts, experiment counts, disk usage hint."""
        with self._connect() as conn:
            total_runs = conn.execute("SELECT COUNT(*) as n FROM runs").fetchone()
            done_runs = conn.execute(
                "SELECT COUNT(*) as n FROM runs WHERE status = 'done'"
            ).fetchone()
            running = conn.execute(
                "SELECT COUNT(*) as n FROM runs WHERE status = 'running'"
            ).fetchone()
            total_exps = conn.execute("SELECT COUNT(*) as n FROM experiments").fetchone()

        return {
            'total_runs': total_runs['n'],
            'done_runs': done_runs['n'],
            'running_runs': running['n'],
            'total_experiments': total_exps['n'],
            'db_path': self.db_path,
        }
