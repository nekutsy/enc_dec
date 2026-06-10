import csv
import os

class CSVLogger:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

def get_last_symbols(csv_path):
    if not os.path.isfile(csv_path):
        return 0
    try:
        with open(csv_path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            if size < 256:
                f.seek(0)
            else:
                f.seek(size - 256)
            lines = f.read().decode('utf-8').strip().splitlines()
            if len(lines) < 2:
                return 0
            last_line = lines[-1]
            return int(last_line.split(',')[0])
    except (ValueError, IndexError):
        return 0

def get_last_epoch(csv_path):
    return get_last_symbols(csv_path)