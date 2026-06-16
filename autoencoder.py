"""Primary text autoencoder — compresses raw Russian text via Unicode-21 encoding."""

import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
from configs import PrimaryConfig, UNICODE_BITS
from model import Autoencoder
from data import load_text, prepare_data, split_into_chunks, vec2seq, export_latent_vectors
from trainers import run_training, build_scheduler, _save_checkpoint, _load_optimizer, _cuda_safe_cleanup
from logger import CSVLogger, get_last_symbols

torch.set_float32_matmul_precision('high')

# ── Layer size helpers ─────────────────────────────────────────


def _default_layer_sizes(config: PrimaryConfig) -> list[int]:
    """11-layer autoencoder: deep hourglass with wide middle."""
    h = config.hidden_dim
    return [
        config.input_dim,
        h * 2, h * 4, h * 2, h * 2,
        config.bottleneck,
        h * 2, h * 2, h * 4, h * 2,
        config.input_dim,
    ]


def _sweep_layer_sizes(input_dim: int, hidden_dim: int, bottleneck: int, n_hidden: int) -> list[int]:
    """Build symmetric layer sizes with `n_hidden` hidden layers each side.

    Encoder: input_dim → H (×n_hidden) → bottleneck
    Decoder: bottleneck → H (×n_hidden) → input_dim
    """
    return [input_dim] + [hidden_dim] * n_hidden + [bottleneck] + [hidden_dim] * n_hidden + [input_dim]


def _save_paths(layer_sizes: list[int], model_name: str, prefix: str = "sessions"):
    """Return (model_path, csv_path) for a given layer configuration.
    Uses encoder half only (decoder always mirrors) to keep filename short."""
    mid = len(layer_sizes) // 2
    encoder_half = layer_sizes[:mid + 1]  # includes bottleneck
    key = "_".join(map(str, encoder_half))
    if len(key) > 200:
        import hashlib
        key = hashlib.md5(key.encode()).hexdigest()[:16]
    base = f"{key}_{model_name}"
    os.makedirs(prefix, exist_ok=True)
    return os.path.join(prefix, f"{base}.pth"), os.path.join(prefix, f"training_losses_{base}.csv")


def _compile_model(model, device):
    """Compile model for GPU — skip for tiny models where overhead dominates."""
    if device.type == "cuda":
        n_params = sum(p.numel() for p in model.parameters())
        if n_params > 50_000:  # only compile models that benefit from it
            return torch.compile(model, mode="reduce-overhead")
    return model


def reconstruct_text(model, text: str, config, device) -> str:
    model.eval()
    max_bits = config.seq_len * UNICODE_BITS
    chunks = split_into_chunks(text, max_bits)
    reconstructed = []
    with torch.inference_mode():
        for orig_chunk, bits in chunks:
            inp = torch.tensor([bits], dtype=torch.float32).to(device)
            out = model(inp).squeeze(0).cpu().tolist()
            rec_str = vec2seq(out)
            reconstructed.append(rec_str)
    return ''.join(reconstructed)


def _train_setup(config, model, optimizer, csv_path, model_path, device):
    """Load checkpoint if available, return start_symbols."""
    start_symbols = get_last_symbols(csv_path)
    if start_symbols > 0:
        print(f"  Resuming from {start_symbols} symbols processed. Loading checkpoint...")
        state = torch.load(model_path, map_location=device, weights_only=True)
        # Detect prefix: old code saved compiled state with _orig_mod. prefix
        has_prefix = any(k.startswith('_orig_mod.') for k in state.keys())
        unwrapped = model._orig_mod if hasattr(model, '_orig_mod') else model
        if has_prefix:
            state = {k[len('_orig_mod.'):]: v for k, v in state.items()}
        unwrapped.load_state_dict(state)
        _load_optimizer(optimizer, model_path, device)
    return start_symbols


def _adaptive_batch_size(input_dim: int, n_hidden: int) -> int:
    """Pick batch size based on model size to maximise GPU utilisation.

    Tiny models leave 90%+ GPU idle — use huge batches to compensate.
    Deep models (h≥4) need tighter limits for VRAM headroom.
    """
    if input_dim <= 84:        # seq_len <= 4: tiny models, GPU mostly idle
        if n_hidden <= 1:     return 32768
        if n_hidden <= 3:     return 16384
        if n_hidden <= 4:     return 8192
        return 4096
    elif input_dim <= 336:     # seq_len <= 16
        if n_hidden <= 2:     return 8192
        if n_hidden <= 4:     return 4096
        return 2048
    elif input_dim <= 672:     # seq_len <= 32
        if n_hidden <= 2:     return 2048
        if n_hidden <= 4:     return 1024
        return 512
    elif input_dim <= 1344:    # seq_len <= 64
        if n_hidden <= 1:     return 1024
        if n_hidden <= 3:     return 512
        if n_hidden <= 5:     return 384
        return 256
    else:                       # seq_len = 128
        if n_hidden <= 1:     return 256
        if n_hidden <= 2:     return 128
        if n_hidden <= 4:     return 96
        return 64


def _log_sweep_header(log_path):
    """Write CSV header for master sweep log."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not os.path.isfile(log_path):
        with open(log_path, 'w', newline='') as f:
            import csv
            w = csv.writer(f)
            w.writerow([
                'seq_len', 'n_hidden', 'input_dim', 'hidden_dim', 'bottleneck',
                'params', 'batch_size', 'total_symbols', 'chars_processed', 'epochs',
                'final_train_loss', 'final_val_loss', 'status', 'duration_seconds'
            ])


def _log_sweep_row(log_path, row):
    import csv
    with open(log_path, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow(row)


def run_experiments():
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_type}")
    device = torch.device(device_type)

    if device_type == "cuda":
        torch.backends.cudnn.benchmark = False

    # ── Sweep parameters ──
    seq_lens = [1, 2, 4, 8, 16, 32, 64, 128]
    n_hidden_opts = [1, 2, 3, 4, 5, 6]
    TARGET_SYMBOLS = 150_000_000

    # Master log — all model results in one file
    SWEEP_LOG = "sessions/sweep_summary.csv"
    _log_sweep_header(SWEEP_LOG)

    text = load_text()
    total_configs = len(seq_lens) * len(n_hidden_opts)
    print(f"\n{'='*60}")
    print(f"Sweep: {len(seq_lens)} seq_lens × {len(n_hidden_opts)} hidden_depths = {total_configs} models")
    print(f"Target: {TARGET_SYMBOLS:,} symbols per model")
    print(f"Hidden dim: 4× input_dim | Bottleneck: seq_len")
    print(f"Master log: {SWEEP_LOG}")
    print(f"{'='*60}\n")

    model_index = 0
    for seq_len in seq_lens:
        input_dim = seq_len * UNICODE_BITS
        # seq_len=128 → 2× to avoid OOM (was 4×)
        hidden_mult = 2 if seq_len >= 128 else 4
        hidden_dim = hidden_mult * input_dim
        bottleneck = seq_len

        for n_hidden in n_hidden_opts:
            model_index += 1
            layer_sizes = _sweep_layer_sizes(input_dim, hidden_dim, bottleneck, n_hidden)
            batch_size = _adaptive_batch_size(input_dim, n_hidden)

            # Temporary model just to count params
            tmp_model = Autoencoder(layer_sizes)
            n_params = sum(p.numel() for p in tmp_model.parameters())
            del tmp_model

            model_name = f"sweep_s{seq_len}_h{n_hidden}"
            model_path, csv_path = _save_paths(layer_sizes, model_name, prefix="sessions/sweep")

            arch_str = "→".join(str(s) for s in layer_sizes)
            chars_per_pass = len(text)  # actual unique characters seen per pass
            total_symbols_per_pass = chars_per_pass * seq_len  # counted symbols (for trainer)
            epochs_needed = max(1, int(TARGET_SYMBOLS / chars_per_pass) + 1)
            max_symbols = epochs_needed * total_symbols_per_pass

            print(f"[{model_index}/{total_configs}] seq_len={seq_len}, n_hidden={n_hidden}")
            print(f"  Arch: ({arch_str})")
            print(f"  Params: {n_params:,}  |  Hidden: {hidden_dim}  |  Batch: {batch_size}")
            print(f"  Epochs: {epochs_needed}  |  Chars/pass: {chars_per_pass:,}  |  Symbols/pass: {total_symbols_per_pass:,}")

            config = PrimaryConfig(
                seq_len=seq_len,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                bottleneck=bottleneck,
                learning_rate=0.001,
                train_ratio=0.99,
                batch_size=batch_size,
                device=device_type,
                model_name=model_name,
                grad_clip=1.0,
                num_workers=2 if device_type == "cuda" else 0,
                lr_scheduler="cosine",
                lr_warmup_epochs=0.05,
                cudnn_benchmark=False,
            )

            train_ds, val_ds = prepare_data(text, config)

            model = Autoencoder(layer_sizes, name=config.model_name).to(device)
            model = _compile_model(model, device)
            optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, fused=device_type == 'cuda')
            total_batches = int(max_symbols / config.batch_size / config.seq_len) + 1
            scheduler = build_scheduler(optimizer, config, total_batches)
            criterion = nn.MSELoss()
            logger = CSVLogger(csv_path)

            # Resume or start
            start_symbols = _train_setup(config, model, optimizer, csv_path, model_path, device)

            remaining = max(0, max_symbols - start_symbols)
            if remaining == 0:
                print(f"  Already complete ({start_symbols:,} symbols). Skipping.")
                _log_sweep_row(SWEEP_LOG, [
                    seq_len, n_hidden, input_dim, hidden_dim, bottleneck,
                    n_params, batch_size, start_symbols, start_symbols // seq_len if seq_len else 0,
                    epochs_needed, '', '', 'skipped', 0
                ])
                continue

            import time as time_mod
            t_start = time_mod.time()

            try:
                final_symbols = run_training(
                    start_symbols=start_symbols,
                    max_symbols=max_symbols,
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    train_dataset=train_ds,
                    val_dataset=val_ds,
                    logger=logger,
                    model_path=model_path,
                    batch_size=config.batch_size,
                    symbols_per_sample=config.seq_len,
                    grad_clip=config.grad_clip,
                    num_workers=config.num_workers,
                    scheduler=scheduler,
                )
                status = "done"
                # Read final losses from CSV
                final_train, final_val = '', ''
                if os.path.isfile(csv_path):
                    with open(csv_path, 'r') as f:
                        lines = f.readlines()
                        if len(lines) >= 2:
                            parts = lines[-1].strip().split(',')
                            if len(parts) >= 3:
                                final_train, final_val = parts[1], parts[2]
            except KeyboardInterrupt:
                final_symbols = get_last_symbols(csv_path)
                status = "interrupted"
                final_train, final_val = '', ''
                print(f"\n  Interrupted at {final_symbols:,} symbols. Skipping to next model.")
                _cuda_safe_cleanup()

            duration = time_mod.time() - t_start

            # Summary line
            print(f"  Result: {status} | {final_symbols:,} symbols | {duration:.0f}s")
            if final_train:
                print(f"  Final loss: train={final_train}, val={final_val}")

            _log_sweep_row(SWEEP_LOG, [
                seq_len, n_hidden, input_dim, hidden_dim, bottleneck,
                n_params, batch_size, final_symbols, final_symbols // seq_len if seq_len else 0,
                epochs_needed, final_train, final_val, status, int(duration)
            ])

            print()
            _cuda_safe_cleanup()

    print(f"\n{'='*60}")
    print(f"Sweep complete. Summary: {SWEEP_LOG}")
    print(f"{'='*60}")


def main():
    config = PrimaryConfig()
    if config.device == "cuda" and not torch.cuda.is_available():
        config.device = "cpu"
    device = torch.device(config.device)
    print(f"Using device: {device}")
    print(f"Encoding: unicode21")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = config.cudnn_benchmark

    if "--experiment" in sys.argv:
        run_experiments()
        return

    text = load_text()
    train_ds, val_ds = prepare_data(text, config)
    layer_sizes = _default_layer_sizes(config)

    model = Autoencoder(layer_sizes, name=config.model_name).to(device)
    model = _compile_model(model, device)

    model_path, csv_path = _save_paths(layer_sizes, config.model_name)

    print(f"Model path: {model_path}")
    print(f"CSV path: {csv_path}")

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, fused=config.device == 'cuda')
    criterion = nn.MSELoss()
    logger = CSVLogger(csv_path)

    start_symbols = _train_setup(config, model, optimizer, csv_path, model_path, device)

    total_epochs = 30
    total_symbols_per_epoch = len(train_ds) * config.seq_len
    target_symbols = total_epochs * total_symbols_per_epoch

    scheduler = build_scheduler(optimizer, config, total_epochs)

    print("Commands: <text to reconstruct>, 'resume N', 'export', 'quit'")
    while True:
        user_input = input("> ")
        if user_input.lower() in ('quit', 'exit'):
            _cuda_safe_cleanup()
            break
        if user_input.lower().startswith('resume'):
            parts = user_input.split()
            if len(parts) == 2 and parts[1].isdigit():
                extra_epochs = int(parts[1])
                extra_symbols = extra_epochs * total_symbols_per_epoch
                new_max = start_symbols + extra_symbols
                print(f"Training for {extra_epochs} more epochs ({extra_symbols} symbols)...")
                try:
                    start_symbols = run_training(
                        start_symbols, new_max, model, optimizer, criterion,
                        train_ds, val_ds, logger, model_path,
                        config.batch_size, config.seq_len,
                        grad_clip=config.grad_clip,
                        num_workers=config.num_workers,
                        scheduler=scheduler,
                    )
                except KeyboardInterrupt:
                    print("\nTraining interrupted. Checkpoint saved.")
                print("Done.\n")
            else:
                print("Usage: resume <epochs>")
            continue
        if user_input.lower() == 'export':
            export_latent_vectors(model, text, config, device,
                                  output_path="data/latent/latent_vectors.pt")
            continue
        if not user_input:
            print("Empty input.")
            continue
        reconstructed = reconstruct_text(model, user_input, config, device)
        print("Reconstructed:", reconstructed, "\n")


if __name__ == "__main__":
    main()
