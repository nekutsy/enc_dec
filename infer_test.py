"""Interactive inference — scan trained models, pick one, test reconstruction.

Commands:
  <#>               Load model by number
  load <#>          Same
  val <#> <#> ...   Quick validation (no model needed)
  random, r         Random sample from dataset
  full [pos]        First 20 windows from dataset (sequential)
  <text>            Reconstruct text (auto-pads to seq_len)
  q, quit           Exit
"""

import torch, sys, os, random, glob, re, csv
sys.path.insert(0, os.path.dirname(__file__))

from configs import UNICODE_BITS
from model import Autoencoder
from data import _build_full_bits, load_text, vec2seq, chars_to_bits, prepare_data
import numpy as np
from trainers import _validate
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_key(path):
    """Parse model sizes from filename like '2688_7644_7644_128_sweep_n2.pth'.
    
    Filenames store only the encoder half (up to bottleneck).
    Reconstruct full architecture by mirroring decoder.
    """
    base = os.path.basename(path).replace('.pth', '').replace('_best', '')
    parts = base.split('_')
    sizes = []
    for p in parts:
        if p.isdigit():
            sizes.append(int(p))
        elif p.startswith('sweep') or p.startswith('batch'):
            break
    if len(sizes) < 3:
        return []
    # Encoder half → full mirror architecture
    decoder = list(reversed(sizes[:-1]))
    return sizes + decoder


def _scan_models(sessions_dir="sessions"):
    """Walk sessions/ for .pth files, return list of (path, sizes, n_params, folder)."""
    models = []
    for root, dirs, files in os.walk(sessions_dir):
        for f in files:
            if f.endswith('.pth') and '_best' not in f and 'training_losses' not in f:
                full = os.path.join(root, f)
                sizes = _parse_key(full)
                if len(sizes) < 3:
                    continue
                n_params = sum(
                    sizes[i] * sizes[i+1] + sizes[i+1] + 2 * sizes[i+1]
                    for i in range(len(sizes) - 1)
                )
                folder = os.path.relpath(root, sessions_dir)
                models.append((full, sizes, n_params, folder))
    return sorted(models, key=lambda m: -m[2])


def _load_model_from_path(path, sizes):
    """Load a trained model given file path and layer sizes."""
    model = Autoencoder(sizes).to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    if any(k.startswith('_orig_mod.') for k in state.keys()):
        state = {k[len('_orig_mod.'):]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def _compute_val_loss(model, text, seq_len):
    """Quick validation pass to evaluate model quality."""
    from configs import PrimaryConfig
    config = PrimaryConfig(seq_len=seq_len)
    train_ds, val_ds = prepare_data(text, config)
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            num_workers=2 if device.type == 'cuda' else 0,
                            pin_memory=(device.type == 'cuda'))
    criterion = nn.MSELoss()
    return _validate(model, val_loader, criterion, device)


def _reconstruct(model, chunk, sl):
    """Reconstruct a single window of size sl. Pads with \\0 if needed."""
    padded = chunk + '\0' * (sl - len(chunk))
    codes = np.array([ord(ch) if ch != '\0' else 0 for ch in padded], dtype=np.uint32)
    bits_np = chars_to_bits(codes).ravel()
    inp_t = torch.from_numpy(bits_np).float().unsqueeze(0).to(device)
    with torch.inference_mode():
        out = model(inp_t).squeeze(0).cpu().numpy()
    rec = vec2seq(out)
    errors = sum(1 for a, b in zip(padded, rec) if a != b)
    bit_err = np.abs(bits_np - out).sum()
    return rec, errors, bit_err


def main():
    print(f"Device: {device}\n")

    # ── Scan models ──
    print("Scanning for trained models...")
    models = _scan_models()
    if not models:
        print("No .pth files found in sessions/")
        return

    # ── Display ──
    print(f"\n{'#':>3}  {'n_params':>12}  {'seq_len':>8}  {'n_hidden':>8}  {'folder':>25}  file")
    print("-" * 100)
    for i, (path, sizes, n_params, folder) in enumerate(models):
        mid = len(sizes) // 2
        seq_len = sizes[0] // UNICODE_BITS
        n_hidden = mid - 1
        fname = os.path.basename(path)
        print(f"{i:>3}  {n_params:>12,}  {seq_len:>8}  {n_hidden:>8}  {folder:>25}  {fname[:60]}")

    # ── State ──
    print(f"\nCommands: <#> load model | 'val <#> <#> ...' | 'random' | 'full' | 'q' quit")
    text = load_text()
    full_bits = _build_full_bits(text)
    n_chars = full_bits.numel() // UNICODE_BITS

    loaded_model = None
    loaded_sl = 0
    loaded_n_params = 0
    rng = random.Random()

    while True:
        cmd = input('> ').strip()
        if not cmd:
            continue

        # ── Quit ──
        if cmd.lower() in ('q', 'quit', 'exit'):
            break

        # ── Load model ──
        if cmd.lower().startswith('load') or cmd.isdigit():
            idx_str = cmd.split()[1] if cmd.lower().startswith('load') else cmd
            try:
                idx = int(idx_str)
            except ValueError:
                print(f"Unknown command: {cmd}")
                continue
            if idx < 0 or idx >= len(models):
                print(f"#{idx} out of range (0–{len(models)-1})")
                continue
            path, sizes, n_params, folder = models[idx]
            sl = sizes[0] // UNICODE_BITS
            print(f"Loaded #{idx}: s{sl}, {n_params:,} params")
            loaded_model = _load_model_from_path(path, sizes)
            loaded_sl = sl
            loaded_n_params = n_params
            continue

        # ── Validation (no model needed) ──
        if cmd.lower().startswith('val'):
            parts = cmd.split()
            if len(parts) < 2:
                print("Usage: val 0 1 3")
                continue
            for p in parts[1:]:
                try:
                    idx = int(p)
                except ValueError:
                    print(f"  Invalid #: {p}")
                    continue
                if idx < 0 or idx >= len(models):
                    print(f"  #{idx} out of range")
                    continue
                path, sizes, n_params, folder = models[idx]
                sl = sizes[0] // UNICODE_BITS
                print(f"  #{idx} (s{sl}, {n_params//1e6:.0f}M)...", end=' ', flush=True)
                model = _load_model_from_path(path, sizes)
                val = _compute_val_loss(model, text, sl)
                print(f"val={val:.6f}")
            continue

        # ── Everything below needs a loaded model ──
        if loaded_model is None:
            print("No model loaded. Use '<#>' or 'load <#>' first.")
            continue

        model = loaded_model
        sl = loaded_sl

        # ── Random sample ──
        if cmd.lower() in ('r', 'random'):
            pos = rng.randint(0, max(0, n_chars - sl - 1))
            chunk = text[pos:pos + sl]
            print(f"@{pos}: {chunk!r}")
            rec, errors, bit_err = _reconstruct(model, chunk, sl)
            print(rec[:sl])
            if errors:
                print(f"errors: {errors}/{sl} chars | {bit_err:.1f} bits")
            continue

        # ── Full sequential scan ──
        if cmd.lower().startswith('full'):
            parts = cmd.split()
            start_pos = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
            if start_pos >= n_chars:
                print(f"Position {start_pos} beyond text end ({n_chars})")
                continue
            n_windows = 20
            total_err_c, total_err_b, total_c = 0, 0, 0
            for w in range(min(n_windows, n_chars - start_pos)):
                chunk = text[start_pos + w:start_pos + w + sl]
                rec, errors, bit_err = _reconstruct(model, chunk, sl)
                total_err_c += errors
                total_err_b += bit_err
                total_c += sl
                indicator = f" {errors}" if errors else ""
                print(f"@{start_pos+w}: {chunk[:40].strip()!r} → {rec[:40].strip()!r}{indicator}")
            print(f"total: {total_err_c}/{total_c} char errors | {total_err_b:.1f} bit errors")
            continue

        # ── Text reconstruction ──
        inp = cmd
        if len(inp) > sl:
            total_err_c, total_err_b, total_c = 0, 0, 0
            for start in range(0, len(inp), sl):
                chunk = inp[start:start + sl]
                rec, errors, bit_err = _reconstruct(model, chunk, sl)
                total_err_c += errors
                total_err_b += bit_err
                total_c += sl
                print(rec.rstrip('\0'))
            if total_err_c:
                print(f"errors: {total_err_c}/{total_c} chars | {total_err_b:.1f} bits")
        else:
            rec, errors, bit_err = _reconstruct(model, inp, sl)
            print(rec[:len(inp)])
            if errors:
                print(f"errors: {errors}/{sl} chars | {bit_err:.1f} bits")

    print("Done.")


if __name__ == "__main__":
    main()
