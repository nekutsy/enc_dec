"""Structured training logger with configurable fields, EMA smoothing.

Core components:
  LoggerConfig   — toggle individual metric fields
  TrainingLogger — per-model CSV + stdout + progress bar
  get_last_samples — read last checkpoint from CSV

Note: GlobalLogger removed; use registry.db for experiment tracking.
"""

import csv
import os
import time as time_mod
from dataclasses import dataclass
from typing import Optional


# ── LoggerConfig ──────────────────────────────────────────────

@dataclass
class LoggerConfig:
    """Which metrics to track and log in per-model CSV and stdout.

    Each bool toggles a column. All fields are optional — disable what you
    don't need.
    """
    epoch: bool = True
    total_samples: bool = True
    speed_sps: bool = True       # samples/sec over last interval
    train_loss: bool = True      # interval average loss
    train_loss_ema: bool = False  # EMA-smoothed loss (β=0.95)
    val_loss: bool = True
    lr: bool = True

    @classmethod
    def all_off(cls) -> 'LoggerConfig':
        return cls(False, False, False, False, False, False, False)

    @classmethod
    def minimal(cls) -> 'LoggerConfig':
        """Only essential metrics: samples + loss."""
        return cls(epoch=False, total_samples=True, speed_sps=False,
                   train_loss=True, train_loss_ema=False, val_loss=False, lr=False)

    @classmethod
    def full(cls) -> 'LoggerConfig':
        """All metrics except ema (noisy, distracting)."""
        return cls(epoch=False, total_samples=True, speed_sps=True,
                   train_loss=True, train_loss_ema=False, val_loss=True, lr=True)

    # Maps field name → CSV header string, in column order
    _COLUMN_MAP = [
        ('epoch',          'epoch'),
        ('total_samples',  'total_samples'),
        ('speed_sps',      'speed_sps'),
        ('train_loss',     'train_loss'),
        ('train_loss_ema', 'train_loss_ema'),
        ('val_loss',       'val_loss'),
        ('lr',             'lr'),
    ]

    def enabled_fields(self) -> list[str]:
        return [k for k, _ in self._COLUMN_MAP if getattr(self, k)]

    def csv_header(self) -> list[str]:
        mapping = dict(self._COLUMN_MAP)
        return [mapping[k] for k in self.enabled_fields()]


# ── TrainingLogger ────────────────────────────────────────────

class TrainingLogger:
    """Per-model training metrics tracker.

    Usage:
        logger = TrainingLogger('path/to/model_losses.csv',
                                config=LoggerConfig(), model_name='sweep_n4')
        for batch in loader:
            ...
            logger.on_batch_end(total_samples, loss.item(), lr=...)
            if checkpoint_reached:
                logger.log_checkpoint(total_samples, avg_loss, epoch_size,
                                     val_loss=val, lr=lr)
        logger.log_final(total_samples, train_loss, epoch_size, duration, status='done')

    Writes to:
      - Per-model CSV  (every checkpoint)
      - Stdout          (every checkpoint — formatted line)
      - Optional global summary CSV (on_final)
    """

    def __init__(self, csv_path: str, config: LoggerConfig | None = None,
                 ema_decay: float = 0.95, model_name: str = '',
                 log_path: str | None = None):
        self.csv_path = csv_path
        self.config = config or LoggerConfig()
        self.ema_decay = ema_decay
        self.model_name = model_name
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        self.log_path = log_path or csv_path.replace('.csv', '.log')

        # Internal state
        self._ema_loss: float | None = None
        self._last_log_time: float | None = None
        self._last_log_samples: int = 0
        self._train_start_time: float | None = None
        self._prog_last_time: float | None = None
        self._prog_last_samples: int = 0

    def log_header(self, lines: list[str]):
        """Write header to .log file. Skips if file already exists."""
        if os.path.isfile(self.log_path):
            return
        with open(self.log_path, 'w') as f:
            for line in lines:
                f.write(line + '\n')

    # ── Batch-level updates ─────────────────────────────────

    def on_batch_end(self, total_samples: int, loss: float, lr: float | None = None):
        """Update EMA loss and start clock on first call.

        Call every batch or at progress-interval frequency.
        """
        if self._train_start_time is None:
            self._train_start_time = time_mod.time()
        if self._ema_loss is None:
            self._ema_loss = loss
        else:
            self._ema_loss = self.ema_decay * self._ema_loss + (1 - self.ema_decay) * loss

    # ── Checkpoint logging ───────────────────────────────────

    def _build_row(self, total_samples: int, train_loss: float, epoch_size: int,
                   val_loss: float | None = None, lr: float | None = None) -> dict:
        """Build a row dict for the enabled fields."""
        now = time_mod.time()
        speed = 0.0
        if self._last_log_time is not None and now > self._last_log_time:
            speed = (total_samples - self._last_log_samples) / (now - self._last_log_time)
        self._last_log_time = now
        self._last_log_samples = total_samples

        ema = self._ema_loss

        row = {}
        fields = self.config.enabled_fields()
        for f in fields:
            if f == 'epoch':
                row[f] = round(total_samples / epoch_size, 4) if epoch_size > 0 else 0.0
            elif f == 'total_samples':
                row[f] = total_samples
            elif f == 'speed_sps':
                row[f] = round(speed, 1)
            elif f == 'train_loss':
                row[f] = train_loss
            elif f == 'train_loss_ema':
                row[f] = ema if ema is not None else train_loss
            elif f == 'val_loss':
                row[f] = val_loss if val_loss is not None else ''
            elif f == 'lr':
                row[f] = lr if lr is not None else ''
        return row

    def on_checkpoint(self, total_samples: int, train_loss: float,
                      epoch_size: int, val_loss: float | None = None,
                      lr: float | None = None) -> dict:
        """Write checkpoint row to per-model CSV. Returns the row dict."""
        row = self._build_row(total_samples, train_loss, epoch_size, val_loss, lr)
        header = self.config.csv_header()
        fields = self.config.enabled_fields()

        write_header = not os.path.isfile(self.csv_path)
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if write_header:
                writer.writeheader()
            out = {}
            for f, h in zip(fields, header):
                v = row.get(f, '')
                out[h] = v
            writer.writerow(out)

        return row

    def log_checkpoint(self, total_samples: int, train_loss: float,
                       epoch_size: int, val_loss: float | None = None,
                       lr: float | None = None, debug: dict | None = None):
        """Full checkpoint: write CSV + .log + print formatted stdout line."""
        row = self.on_checkpoint(total_samples, train_loss, epoch_size, val_loss, lr)
        line = self.format_line(row, debug=debug)
        print(line, flush=True)
        with open(self.log_path, 'a') as f:
            f.write(line + '\n')

    def log_final(self, total_samples: int, train_loss: float, epoch_size: int,
                  status: str = 'done', duration_seconds: float = 0.0,
                  val_loss: float | None = None, debug: dict | None = None):
        """Write final checkpoint and return summary dict for global CSV."""
        row = self.on_checkpoint(total_samples, train_loss, epoch_size, val_loss)
        line = self.format_line(row, debug=debug)
        print(line, flush=True)
        with open(self.log_path, 'a') as f:
            f.write(line + '\n')
        return row

    # ── Formatting ───────────────────────────────────────────

    @staticmethod
    def _is_present(row, key):
        v = row.get(key, '')
        return v is not None and v != ''

    def format_line(self, row: dict, debug: dict | None = None) -> str:
        """Format a checkpoint row as a single-line string."""
        parts = []
        cfg = self.config
        if 'epoch' in row and cfg.epoch:
            parts.append(f'epoch={row["epoch"]:>6.2f}')
        if 'total_samples' in row and cfg.total_samples:
            parts.append(f'samples={row["total_samples"]:>11,}')
        if 'speed_sps' in row and cfg.speed_sps:
            parts.append(f'speed={row["speed_sps"]:>6,.0f} sps')
        if 'train_loss' in row and cfg.train_loss:
            parts.append(f'loss={row["train_loss"]:.6f}')
        if 'train_loss_ema' in row and cfg.train_loss_ema:
            parts.append(f'ema={row["train_loss_ema"]:.6f}')
        if 'val_loss' in row and cfg.val_loss and self._is_present(row, 'val_loss'):
            parts.append(f'val={row["val_loss"]:.6f}')
        if 'lr' in row and cfg.lr and self._is_present(row, 'lr'):
            v = row['lr']
            parts.append(f'lr={v:.2e}' if isinstance(v, float) else f'lr={v}')
        if 'speed_sps' in row and cfg.speed_sps and row.get('speed_sps', 0) == 0:
            parts = [p for p in parts if not p.startswith('speed=')]
        if debug:
            D_v = debug.get('D')
            Dp_v = debug.get('Dprime')
            mult_v = debug.get('mult')
            delta_v = debug.get('lr_delta')
            if D_v is not None:
                parts.append(f'D={D_v:.2e}')
            if Dp_v is not None:
                parts.append(f"D'={Dp_v:.2e}")
            if mult_v is not None:
                parts.append(f'm={mult_v:.4f}')
            elif delta_v is not None:
                parts.append(f'Δlr={delta_v:+.2e}')

        ts = time_mod.strftime('%Y-%m-%d %H:%M:%S')
        name_str = f'{self.model_name} | ' if self.model_name else ''
        return f'{ts} | {name_str}{" | ".join(parts)}'

    def format_progress(self, total_samples: int, max_samples: int,
                        loss: float, epoch_size: int, lr: float | None = None,
                        debug: dict | None = None) -> str:
        """Format a progress line for stderr (in-place updates)."""
        now = time_mod.time()
        speed = 0.0
        if self._prog_last_time is not None and now > self._prog_last_time:
            samples_delta = total_samples - self._prog_last_samples
            speed = samples_delta / (now - self._prog_last_time)
        self._prog_last_time = now
        self._prog_last_samples = total_samples
        line = (f'\r\033[Ksamples={total_samples:>11,} | '
                f'loss={loss:.6f} | speed={speed:.0f} sps')
        if lr is not None:
            line += f' | lr={lr:.2e}'
        if debug:
            D_v = debug.get('D')
            Dp_v = debug.get('Dprime')
            mult_v = debug.get('mult')
            delta_v = debug.get('lr_delta')
            if D_v is not None:
                line += f' | D={D_v:.2e}'
            if Dp_v is not None:
                line += f" | D'={Dp_v:.2e}"
            if mult_v is not None:
                line += f' | m={mult_v:.4f}'
            elif delta_v is not None:
                line += f' | Δlr={delta_v:+.2e}'
        return line

    @property
    def ema_loss(self) -> float | None:
        return self._ema_loss

    @property
    def elapsed(self) -> float:
        if self._train_start_time is None:
            return 0.0
        return time_mod.time() - self._train_start_time


def get_last_samples(csv_path: str) -> int:
    """Read total_samples from the last row of a per-model CSV.

    Handles formats:
      New: first column is always 'total_samples' (any number of columns)
      Old (3 cols): total_symbols, train_loss, val_loss → returns 0
      Old (4 cols): total_samples, total_symbols, train_loss, val_loss

    Skips header row. Uses tail-read for efficiency on large files.
    """
    if not os.path.isfile(csv_path):
        return 0
    try:
        with open(csv_path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            tail = min(size, 16384)
            f.seek(size - tail)
            tail_bytes = f.read()
        lines = tail_bytes.decode('utf-8').strip().splitlines()
        if len(lines) < 2:
            return 0
        last_line = lines[-1]
        parts = last_line.split(',')
        first_val = int(float(parts[0]))
        if len(parts) >= 4:
            return first_val
        elif len(parts) == 3:
            return 0  # old 3-col format — can't infer without seq_len
        return 0
    except (ValueError, IndexError, UnicodeDecodeError):
        return 0


def get_last_symbols(csv_path: str) -> int:
    """Legacy: read total_symbols from old 3-column format CSV.

    Returns 0 for new-format CSVs (4+ columns).
    """
    if not os.path.isfile(csv_path):
        return 0
    try:
        with open(csv_path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            tail = min(size, 16384)
            f.seek(size - tail)
            tail_bytes = f.read()
        lines = tail_bytes.decode('utf-8').strip().splitlines()
        if len(lines) < 2:
            return 0
        last_line = lines[-1]
        parts = last_line.split(',')
        if len(parts) == 3 and int(float(parts[0])) > 50_000_000:
            return int(float(parts[0]))
        return 0
    except (ValueError, IndexError, UnicodeDecodeError):
        return 0
