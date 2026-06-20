"""CSV-based checkpoint logger and resume utilities."""

import csv
import os


class CSVLogger:
    """Minimal CSV logger for training metrics."""

    def __init__(self, csv_path):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)


def get_last_samples(csv_path):
    """Read total_samples from the last line of the CSV.

    New format: '{total_samples},{total_symbols},{train_loss},{val_loss}'
    Old format: '{total_symbols},{train_loss},{val_loss}'

    For old format, total_symbols is converted to samples via floor division
    (caller must provide seq_len, or we estimate it).
    """
    if not os.path.isfile(csv_path):
        return 0
    try:
        with open(csv_path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            tail = min(size, 8192)
            f.seek(size - tail)
            tail_bytes = f.read()
        lines = tail_bytes.decode('utf-8').strip().splitlines()
        if len(lines) < 2:
            return 0
        last_line = lines[-1]
        parts = last_line.split(',')
        first_val = int(parts[0])
        # Heuristic: if there are 4 columns, it's new format; 3 columns = old format.
        # New: total_samples,total_symbols,train_loss,val_loss
        # Old: total_symbols,train_loss,val_loss
        if len(parts) >= 4:
            return first_val
        elif len(parts) == 3:
            # Old format (total_symbols). Can't infer samples without seq_len.
            return 0
        return 0
    except (ValueError, IndexError, UnicodeDecodeError):
        return 0


def get_last_symbols(csv_path):
    """Legacy — read total_symbols from old-format CSV. Returns 0 for new format."""
    if not os.path.isfile(csv_path):
        return 0
    try:
        with open(csv_path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            tail = min(size, 8192)
            f.seek(size - tail)
            tail_bytes = f.read()
        lines = tail_bytes.decode('utf-8').strip().splitlines()
        if len(lines) < 2:
            return 0
        last_line = lines[-1]
        parts = last_line.split(',')
        if len(parts) >= 2 and int(parts[0]) > 50_000_000:
            # Old format: first col is total_symbols
            return int(parts[0])
        return 0
    except (ValueError, IndexError, UnicodeDecodeError):
        return 0
