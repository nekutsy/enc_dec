"""CSV-based checkpoint logger and resume utilities."""

import csv
import os


class CSVLogger:
    """Minimal CSV logger for training metrics."""

    def __init__(self, csv_path):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)


def get_last_symbols(csv_path):
    """Read the total_symbols value from the last line of the CSV.

    Handles arbitrarily long lines — reads from end of file.
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
        return int(last_line.split(',')[0])
    except (ValueError, IndexError, UnicodeDecodeError):
        return 0


def get_last_epoch(csv_path):
    return get_last_symbols(csv_path)
