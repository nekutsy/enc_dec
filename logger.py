import csv
import os

class CSVLogger:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.file_exists = os.path.isfile(csv_path)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    def log(self, epoch, train_loss, val_loss):
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not self.file_exists:
                writer.writerow(['epoch', 'train_loss', 'val_loss'])
                self.file_exists = True
            writer.writerow([epoch, train_loss, val_loss])

def get_last_epoch(csv_path):
    if not os.path.isfile(csv_path):
        return -1
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        last = -1
        for row in reader:
            try:
                epoch = int(row['epoch'])
                if epoch > last:
                    last = epoch
            except ValueError:
                pass
    return last