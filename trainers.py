import torch
import os
import sys
import time
from torch.utils.data import TensorDataset, DataLoader

def run_training(start_epoch, max_epochs, model, optimizer, criterion,
                train_X, train_y, val_X, val_y, logger, model_path, batch_size,
                symbols_per_sample, report_interval_symbols=1_000_000):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    device = next(model.parameters()).device
    total_symbols_per_epoch = len(train_X) * symbols_per_sample

    train_dataset = TensorDataset(train_X, train_y if train_y is not None else train_X)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_dataset = TensorDataset(val_X, val_y if val_y is not None else val_X)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)

    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    try:
        for epoch in range(start_epoch, max_epochs):
            model.train()
            train_loss_sum = 0.0
            processed_train = 0
            processed_symbols = 0
            next_report = report_interval_symbols
            epoch_start_time = time.time()
            last_report_time = epoch_start_time
            last_report_symbols = 0

            sys.stdout.write(f"\rEpoch {epoch:6d} | Progress: 0.0% | Loss: --- | Speed: --- sym/s | Total ETA: ---s")
            sys.stdout.flush()

            for x_batch, y_batch in train_loader:
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
                train_loss_sum += batch_loss * batch_size_actual
                processed_train += batch_size_actual

                processed_symbols += batch_size_actual * symbols_per_sample
                if processed_symbols >= next_report and processed_symbols < total_symbols_per_epoch:
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    current_time = time.time()
                    time_delta = current_time - last_report_time
                    symbols_delta = processed_symbols - last_report_symbols
                    speed = symbols_delta / time_delta if time_delta > 0 else 0

                    remaining_in_current = total_symbols_per_epoch - processed_symbols
                    remaining_full_epochs = max_epochs - epoch - 1
                    total_remaining_symbols = remaining_full_epochs * total_symbols_per_epoch + remaining_in_current
                    total_eta = total_remaining_symbols / speed if speed > 0 else 0

                    avg_loss = train_loss_sum / processed_train
                    sys.stdout.write(
                        f"\rEpoch {epoch:6d} | Progress: {processed_symbols/total_symbols_per_epoch*100:.1f}% "
                        f"| Loss: {avg_loss:.6f} | Speed: {speed:.0f} sym/s | Total ETA: {total_eta:.0f}s"
                    )
                    sys.stdout.flush()
                    next_report += report_interval_symbols
                    last_report_time = current_time
                    last_report_symbols = processed_symbols

            if device.type == 'cuda':
                torch.cuda.synchronize()
            avg_train_loss = train_loss_sum / processed_train if processed_train > 0 else 0.0
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_speed = total_symbols_per_epoch / epoch_duration if epoch_duration > 0 else 0

            remaining_epochs = max_epochs - epoch - 1
            total_remaining_symbols = remaining_epochs * total_symbols_per_epoch
            total_eta = total_remaining_symbols / epoch_speed if epoch_speed > 0 and remaining_epochs > 0 else 0

            model.eval()
            val_loss_sum = 0.0
            processed_val = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(device, non_blocking=True)
                    y_batch = y_batch.to(device, non_blocking=True)
                    out = model(x_batch)
                    loss = criterion(out, y_batch)
                    val_loss_sum += loss.item() * x_batch.size(0)
                    processed_val += x_batch.size(0)

            avg_val_loss = val_loss_sum / processed_val if processed_val > 0 else 0.0

            logger.log(epoch, avg_train_loss, avg_val_loss)
            if remaining_epochs > 0:
                sys.stdout.write(
                    f"\rEpoch {epoch:6d} finished | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Speed: {epoch_speed:.0f} sym/s | Total ETA: {total_eta:.0f}s\n"
                )
            else:
                sys.stdout.write(
                    f"\rEpoch {epoch:6d} finished | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Speed: {epoch_speed:.0f} sym/s\n"
                )
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model...")
        torch.save(model.state_dict(), model_path)
        raise
    torch.save(model.state_dict(), model_path)
    print(f"Training finished. Model saved to {model_path}")