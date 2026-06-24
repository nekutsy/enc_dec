"""Interactive inference REPL — thin CLI wrapper around inference subsystem.

Usage:
  python cli/infer.py          # CPU
  python cli/infer.py --gpu    # GPU
"""

import argparse
import sys
import os
import random

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from encoding.unicode21 import UNICODE_BITS
from data import load_text, _build_full_bits, prepare_data
from inference.scan import scan_models, parse_key, load_model
from inference.api import ModelInference
from torch.utils.data import DataLoader


# ── Chain parser ────────────────────────────────────────────

CHAIN_VERBS = {'enc', 'dec', 'z', 'latent', 'r', 'random', 'full'}
_ARGS_VERBS = {'enc', 'full'}
_STRUCT_VERBS = {'dec', 'z', 'latent'}


def _parse_chain(cmd_str: str) -> list[tuple[str, str]] | None:
    """Parse chain: 'dec enc random' → [('dec', ''), ('enc', 'random')]."""
    tokens = cmd_str.split()
    if not tokens or tokens[0].lower() not in CHAIN_VERBS:
        return None
    sub = []
    i = 0
    while i < len(tokens):
        verb = tokens[i].lower()
        i += 1
        args_parts: list[str] = []
        if verb == 'dec':
            while i < len(tokens):
                try:
                    float(tokens[i])
                    args_parts.append(tokens[i])
                    i += 1
                except ValueError:
                    break
        elif verb in _ARGS_VERBS:
            while (i < len(tokens)
                   and tokens[i].lower() not in _ARGS_VERBS
                   and tokens[i].lower() not in _STRUCT_VERBS):
                args_parts.append(tokens[i])
                i += 1
        sub.append((verb, ' '.join(args_parts)))
    return sub


def _random_chunk(text: str, sl: int, n_chars: int, rng: random.Random
                  ) -> tuple[int, str]:
    pos = rng.randint(0, max(0, n_chars - sl - 1))
    return pos, text[pos:pos + sl]


# ── REPL ────────────────────────────────────────────────────

def main(device_override: torch.device | None = None):
    device = device_override or torch.device('cpu')
    print(f"Device: {device}")

    # ── Scan models ──
    print("Scanning for trained models...")
    models = scan_models()
    if not models:
        print("No .pth files found in sessions/")
        return

    # ── Display ──
    print(f"\n{'#':>3}  {'n_params':>12}  {'seq_len':>8}  "
          f"{'n_hidden':>8}  {'folder':>25}  file")
    print("-" * 100)
    for i, (path, sizes, n_params, folder) in enumerate(models):
        mid = len(sizes) // 2
        seq_len = sizes[0] // UNICODE_BITS
        n_hidden = mid - 1
        fname = os.path.basename(path)
        print(f"{i:>3}  {n_params:>12,}  {seq_len:>8}  "
              f"{n_hidden:>8}  {folder:>25}  {fname[:60]}")

    # ── Help ──
    print(f"\nEnc: enc <text|random|@pos> | Dec: dec | Show latent: z")
    print(f"Direct: <text> | Random: r | Full: full [pos] | Quit: q")
    print(f"Chains: dec enc random | enc random dec | ...")

    # ── Data ──
    text = load_text()
    full_bits = _build_full_bits(text)
    n_chars = full_bits.numel() // UNICODE_BITS

    # ── Session state ──
    inf: ModelInference | None = None
    loaded_sl = 0
    last_latent: np.ndarray | None = None
    last_latent_sl = 0
    rng = random.Random()

    # ── Handlers ──
    def _resolve_input(input_text: str) -> tuple[str | None, str]:
        lt = input_text.lower()
        if lt in ('random', 'r'):
            pos, chunk = _random_chunk(text, loaded_sl, n_chars, rng)
            return f"@random {pos}", chunk
        if input_text.startswith('@') and input_text[1:].isdigit():
            pos = int(input_text[1:])
            chunk = text[pos:pos + loaded_sl]
            return f"@pos {pos}", chunk
        return None, input_text

    def _print_latent(latent: np.ndarray):
        print(f"latent [{len(latent)}]:")
        for i in range(0, len(latent), 16):
            row = latent[i:i + 16]
            print('  ' + ' '.join(f'{v:+.4f}' for v in row))
        print(f"  range: [{latent.min():+.4f}, {latent.max():+.4f}]  "
              f"mean={latent.mean():+.4f}  std={latent.std():.4f}")

    def _cmd_enc(args_str: str):
        nonlocal last_latent, last_latent_sl
        if not args_str:
            print("Usage: enc <text|random|@pos>")
            return
        note, chunk = _resolve_input(args_str)
        if note:
            print(f"{note}: {chunk!r}" if chunk else note)
        assert inf is not None
        last_latent = inf.encode(chunk)
        last_latent_sl = loaded_sl
        _print_latent(last_latent)

    def _cmd_dec(args_str: str = ''):
        nonlocal last_latent, last_latent_sl
        if args_str:
            try:
                vals = [float(x) for x in args_str.split()]
                last_latent = np.array(vals, dtype=np.float32)
                last_latent_sl = loaded_sl
                print(f"latent set from input [{len(last_latent)} values]")
            except ValueError:
                print("Invalid values — expected: dec 1.2 -3.4 5.6")
                return
        if last_latent is None:
            print("No latent stored. Use 'enc <text>' first, or 'dec <values>'.")
            return
        assert inf is not None
        rec = inf.decode(last_latent)
        print(rec.rstrip('\0')[:last_latent_sl])

    def _cmd_z(_args_str: str = ''):
        if last_latent is None:
            print("No latent stored.")
            return
        _print_latent(last_latent)

    def _cmd_random(_args_str: str = ''):
        nonlocal last_latent, last_latent_sl
        pos, chunk = _random_chunk(text, loaded_sl, n_chars, rng)
        print(f"@{pos}: {chunk!r}")
        assert inf is not None
        rec, errors, bit_err = inf.reconstruct(chunk)
        print(rec[:loaded_sl])
        if errors:
            print(f"errors: {errors}/{loaded_sl} chars | {bit_err:.1f} bits")
        last_latent = inf.encode(chunk)
        last_latent_sl = loaded_sl

    def _cmd_full(args_str: str):
        nonlocal last_latent, last_latent_sl
        parts = args_str.split() if args_str else []
        start_pos = int(parts[0]) if parts and parts[0].isdigit() else 0
        if start_pos >= n_chars:
            print(f"Position {start_pos} beyond text end ({n_chars})")
            return
        n_windows = 20
        total_err_c, total_err_b, total_c = 0, 0, 0.0
        assert inf is not None
        for w in range(min(n_windows, n_chars - start_pos)):
            chunk = text[start_pos + w:start_pos + w + loaded_sl]
            rec, errors, bit_err = inf.reconstruct(chunk)
            total_err_c += errors
            total_err_b += bit_err
            total_c += loaded_sl
            indicator = f" {errors}" if errors else ""
            print(f"@{start_pos + w}: {chunk[:40].strip()!r} "
                  f"→ {rec[:40].strip()!r}{indicator}")
            if w == 0:
                last_latent = inf.encode(chunk)
                last_latent_sl = loaded_sl
        print(f"total: {total_err_c}/{total_c} char errors | "
              f"{total_err_b:.1f} bit errors")

    def _cmd_text(inp: str):
        nonlocal last_latent, last_latent_sl
        assert inf is not None
        sl = loaded_sl
        if len(inp) > sl:
            rec, errors, bit_err = inf.reconstruct_long(inp)
            print(rec)
            if errors:
                total_c = ((len(inp) + sl - 1) // sl) * sl
                print(f"errors: {errors}/{total_c} chars | {bit_err:.1f} bits")
        else:
            rec, errors, bit_err = inf.reconstruct(inp)
            print(rec.rstrip('\0'))
            if errors:
                print(f"errors: {errors}/{sl} chars | {bit_err:.1f} bits")
            last_latent = inf.encode(inp)
            last_latent_sl = sl

    CHAIN_DISPATCH = {
        'enc': _cmd_enc, 'dec': _cmd_dec,
        'z': _cmd_z, 'latent': _cmd_z,
        'r': _cmd_random, 'random': _cmd_random,
        'full': _cmd_full,
    }

    # ── REPL loop ──
    while True:
        cmd = input('> ').strip()
        if not cmd:
            continue

        # Quit
        if cmd.lower() in ('q', 'quit', 'exit'):
            break

        # Load model
        if cmd.lower().startswith('load') or cmd.isdigit():
            idx_str = (cmd.split()[1] if cmd.lower().startswith('load')
                       else cmd)
            try:
                idx = int(idx_str)
            except ValueError:
                print(f"Unknown command: {cmd}")
                continue
            if idx < 0 or idx >= len(models):
                print(f"#{idx} out of range (0–{len(models) - 1})")
                continue
            path, sizes, n_params, folder = models[idx]
            sl = sizes[0] // UNICODE_BITS
            print(f"Loaded #{idx}: s{sl}, {n_params:,} params")
            loaded_model = load_model(path, sizes, str(device))
            inf = ModelInference(loaded_model, sl, device)
            loaded_sl = sl
            continue

        # Validation (no model needed)
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
                print(f"  #{idx} (s{sl}, {n_params // 10**6:.0f}M)...",
                      end=' ', flush=True)
                model = load_model(path, sizes, str(device))
                _, val_ds = prepare_data(text, sl)
                val_loader = DataLoader(
                    val_ds, batch_size=256, shuffle=False,
                    num_workers=2 if device.type == 'cuda' else 0,
                    pin_memory=(device.type == 'cuda'),
                )
                val_inf = ModelInference(model, sl, device)
                val = val_inf.validate(val_loader)
                print(f"val={val:.6f}")
            continue

        # Command chain
        chain = _parse_chain(cmd)
        if chain is not None:
            if inf is None:
                print("No model loaded. Use '<#>' or 'load <#>' first.")
                continue
            _PRODUCER_ORDER = {'enc': 0, 'full': 0, 'r': 0, 'random': 0,
                               'dec': 1, 'z': 1, 'latent': 1}
            ordered = sorted(chain,
                             key=lambda x: _PRODUCER_ORDER.get(x[0], 2))
            for verb, args in ordered:
                handler = CHAIN_DISPATCH.get(verb)
                if handler:
                    handler(args)
            continue

        # Needs a loaded model below
        if inf is None:
            print("No model loaded. Use '<#>' or 'load <#>' first.")
            continue

        # Single command dispatch
        if cmd.lower().startswith('enc'):
            _cmd_enc(cmd[3:].strip() if cmd.lower().startswith('enc ') else '')
            continue
        if cmd.lower().startswith('dec'):
            _cmd_dec(cmd[3:].strip() if cmd.lower().startswith('dec ') else '')
            continue
        if cmd.lower() in ('z', 'latent'):
            _cmd_z()
            continue
        if cmd.lower() in ('r', 'random'):
            _cmd_random()
            continue
        if cmd.lower().startswith('full'):
            _cmd_full(' '.join(cmd.split()[1:]))
            continue

        # Text reconstruction
        _cmd_text(cmd)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Interactive model inference')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--cpu', action='store_true', help='Force CPU (default)')
    args = parser.parse_args()

    if args.gpu and torch.cuda.is_available():
        main(torch.device('cuda'))
    else:
        if args.gpu:
            print("CUDA not available, falling back to CPU")
        main(torch.device('cpu'))
