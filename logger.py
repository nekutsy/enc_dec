# logger.py
import csv
import os

class CSVLogger:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

def get_last_symbols(csv_path):
    if not os.path.isfile(csv_path):
        return 0
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        last = 0
        for row in reader:
            try:
                symbols = int(row['total_symbols'])
                if symbols > last:
                    last = symbols
            except (ValueError, KeyError):
                pass
    return last