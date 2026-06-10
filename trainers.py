# trainers.py
import torch
import os
import sys
import time
import csv
from torch.utils.data import TensorDataset, DataLoader

def run_training(start_symbols, max_symbols, model, optimizer, criterion,
                train_X, train_y, val_X, val_y, logger, model_path, batch_size,
                symbols_per_sample, report_interval_symbols=1_000_000):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    device = next(model.parameters()).device
    total_symbols_in_dataset = len(train_X) * symbols_per_sample

    train_dataset = TensorDataset(train_X, train_y if train_y is not None else train_X)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_dataset = TensorDataset(val_X, val_y if val_y is not None else val_X)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    total_symbols_processed = start_symbols
    LOG_INTERVAL = 100_000_000
    UPDATE_INTERVAL = 1_000_000

    interval_train_loss_sum = 0.0
    interval_train_samples = 0

    next_update = total_symbols_processed + UPDATE_INTERVAL
    next_log = total_symbols_processed + LOG_INTERVAL
    last_update_time = time.time()
    last_update_symbols = total_symbols_processed

    if not hasattr(logger, 'initialized'):
        logger.initialized = True
        with open(logger.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['total_symbols', 'train_loss', 'val_loss'])

    sys.stdout.write("\r\033[K")
    sys.stdout.flush()

    try:
        while total_symbols_processed < max_symbols:
            model.train()
            for x_batch, y_batch in train_loader:
                if total_symbols_processed >= max_symbols:
                    break
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                optimizer.zero_grad()
                if scaler is not None:
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        out = model(x_batch)
                        loss = criterion(out, y_batch)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    out = model(x_batch)
                    loss = criterion(out, y_batch)
                    loss.backward()
                    optimizer.step()

                batch_loss = loss.item()
                batch_size_actual = x_batch.size(0)
                batch_symbols = batch_size_actual * symbols_per_sample

                interval_train_loss_sum += batch_loss * batch_size_actual
                interval_train_samples += batch_size_actual
                total_symbols_processed += batch_symbols

                if total_symbols_processed >= next_update:
                    current_time = time.time()
                    time_delta = current_time - last_update_time
                    symbols_delta = total_symbols_processed - last_update_symbols
                    speed = symbols_delta / time_delta if time_delta > 0 else 0
                    remaining = max_symbols - total_symbols_processed
                    eta = remaining / speed if speed > 0 else 0
                    avg_loss = interval_train_loss_sum / interval_train_samples if interval_train_samples > 0 else 0
                    progress = (total_symbols_processed / max_symbols) * 100

                    sys.stdout.write(f"\r\033[KProgress: {progress:.1f}% | Loss: {avg_loss:.6f} | Speed: {speed:.0f} sym/s | ETA: {eta:.0f}s")
                    sys.stdout.flush()

                    next_update += UPDATE_INTERVAL
                    last_update_time = current_time
                    last_update_symbols = total_symbols_processed

                if total_symbols_processed >= next_log:
                    avg_train_loss = interval_train_loss_sum / interval_train_samples if interval_train_samples > 0 else 0

                    model.eval()
                    val_loss_sum = 0.0
                    processed_val = 0
                    with torch.no_grad():
                        for x_batch_val, y_batch_val in val_loader:
                            x_batch_val = x_batch_val.to(device, non_blocking=True)
                            y_batch_val = y_batch_val.to(device, non_blocking=True)
                            out_val = model(x_batch_val)
                            loss_val = criterion(out_val, y_batch_val)
                            val_loss_sum += loss_val.item() * x_batch_val.size(0)
                            processed_val += x_batch_val.size(0)
                    avg_val_loss = val_loss_sum / processed_val if processed_val > 0 else 0
                    model.train()

                    with open(logger.csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([total_symbols_processed, avg_train_loss, avg_val_loss])

                    sys.stdout.write(f"\r\033[K[{total_symbols_processed} symbols] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}\n")
                    sys.stdout.flush()

                    interval_train_loss_sum = 0.0
                    interval_train_samples = 0
                    next_log += LOG_INTERVAL

                    last_update_time = time.time()
                    last_update_symbols = total_symbols_processed
                    next_update = total_symbols_processed + UPDATE_INTERVAL

            if total_symbols_processed >= max_symbols:
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model...")
        torch.save(model.state_dict(), model_path)
        raise

    if interval_train_samples > 0:
        avg_train_loss = interval_train_loss_sum / interval_train_samples
        model.eval()
        val_loss_sum = 0.0
        processed_val = 0
        with torch.no_grad():
            for x_batch_val, y_batch_val in val_loader:
                x_batch_val = x_batch_val.to(device, non_blocking=True)
                y_batch_val = y_batch_val.to(device, non_blocking=True)
                out_val = model(x_batch_val)
                loss_val = criterion(out_val, y_batch_val)
                val_loss_sum += loss_val.item() * x_batch_val.size(0)
                processed_val += x_batch_val.size(0)
        avg_val_loss = val_loss_sum / processed_val if processed_val > 0 else 0
        with open(logger.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([total_symbols_processed, avg_train_loss, avg_val_loss])
        sys.stdout.write(f"\r\033[K[{total_symbols_processed} symbols] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}\n")
        sys.stdout.flush()

    torch.save(model.state_dict(), model_path)
    print(f"Training finished. Model saved to {model_path}")
    return total_symbols_processed