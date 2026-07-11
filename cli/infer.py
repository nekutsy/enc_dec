"""Interactive inference REPL — thin CLI wrapper around inference subsystem.

Usage:
  python cli/infer.py          # CPU
  python cli/infer.py --gpu    # GPU
"""

import argparse
import sys
import os
import random
import readline  # noqa — enables line editing / history

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from encoding.unicode21 import UNICODE_BITS
from data import load_text, _build_full_bits, prepare_data
from inference.scan import scan_models, ModelInfo, load_model
from inference.api import ModelInference
from torch.utils.data import DataLoader
from utils import gpu_health_check


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


def _fmt_params(n: int) -> str:
    """Format n_params like '448.1M' or '12.3K'."""
    if n >= 1_000_000:
        return f'{n / 1_000_000:.1f}M'
    if n >= 1_000:
        return f'{n / 1_000:.1f}K'
    return str(n)


def _fmt_loss(val: float | None) -> str:
    """Format loss for display. None → '—'."""
    if val is None:
        return '—'
    return f'{val:.4f}'


def _print_model_table(models: list[ModelInfo]):
    """Print scan results table with losses and run_ids."""
    print(f"{'#':>3}  {'params':>7}  {'layers':>6}  "
          f"{'train':>8}  {'val':>8}  {'run_id':>12}  model")
    print("-" * 100)
    for i, mi in enumerate(models):
        print(f"{i:>3}  {_fmt_params(mi.n_params):>7}  "
              f"{mi.n_hidden_str:>6}  "
              f"{_fmt_loss(mi.train_loss):>8}  "
              f"{_fmt_loss(mi.val_loss):>8}  "
              f"{mi.run_id[:12]:>12}  {mi.label}")


# ── REPL ────────────────────────────────────────────────────

def main(device_override: torch.device | None = None):
    device = device_override or torch.device('cpu')
    print(f"Device: {device}")

    models = scan_models()
    if not models:
        print("No .pth files found in sessions/")
        return

    _print_model_table(models)

    print(f"\nCommands: enc <text|random|@pos> | dec | z | r | full | val | info | q")
    print(f"Chains: dec enc random | enc random dec | ...")

    # ── Data ──
    texts = load_text()
    full_bits, _ = _build_full_bits(texts, seq_len=128)
    n_chars = full_bits.numel() // UNICODE_BITS
    text = ''.join(texts)

    # ── Session state ──
    inf: ModelInference | None = None
    loaded_info: ModelInfo | None = None
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

    def _cmd_info(_args_str: str = ''):
        if loaded_info is None:
            print("No model loaded.")
            return
        mi = loaded_info
        print(f"  #{mi.run_id[:12]}  {mi.label}")
        print(f"  params: {_fmt_params(mi.n_params)}  "
              f"layers: {mi.n_hidden_str}  "
              f"bottleneck: {mi.sizes[len(mi.sizes) // 2]}")
        if mi.train_loss is not None:
            print(f"  train={mi.train_loss:.6f}  val={mi.val_loss:.6f}"
                  if mi.val_loss is not None else f"  train={mi.train_loss:.6f}")

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

    def _load_model_by_idx(idx: int):
        nonlocal inf, loaded_sl, loaded_info
        info = models[idx]
        try:
            model = load_model(info.path, info.sizes, str(device))
        except torch.cuda.OutOfMemoryError:
            print(f"  ⚠ OOM loading #{idx} ({_fmt_params(info.n_params)} params)")
            if device.type == 'cuda' and info.n_params > 100_000_000:
                print(f"  Try loading on CPU instead (restart without --gpu)")
            return
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"  ⚠ OOM loading #{idx} ({_fmt_params(info.n_params)} params)")
                return
            raise
        inf = ModelInference(model, info.sizes[0] // UNICODE_BITS, device)
        loaded_sl = info.sizes[0] // UNICODE_BITS
        loaded_info = info
        loss_str = ''
        if info.train_loss is not None:
            loss_str = f'  train={info.train_loss:.4f}'
            if info.val_loss is not None:
                loss_str += f'  val={info.val_loss:.4f}'
        print(f"Loaded #{idx} [{info.run_id[:12]}] "
              f"s{loaded_sl} {_fmt_params(info.n_params)} "
              f"n={info.n_hidden_str}{loss_str}")

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

        if cmd.lower() in ('q', 'quit', 'exit'):
            break

        # info — show loaded model details
        if cmd.lower() == 'info':
            _cmd_info()
            continue

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
            _load_model_by_idx(idx)
            continue

        # Validation (no model loaded needed)
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
                info = models[idx]
                sl = info.sizes[0] // UNICODE_BITS
                m_params_label = f"s{sl} {_fmt_params(info.n_params)} [{info.run_id[:12]}]"
                print(f"  #{idx} ({m_params_label})...", end=' ', flush=True)
                try:
                    model = load_model(info.path, info.sizes, str(device))
                except torch.cuda.OutOfMemoryError:
                    print("OOM")
                    continue
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        print("OOM")
                        continue
                    raise
                try:
                    _, val_ds = prepare_data(texts, sl)
                    val_loader = DataLoader(
                        val_ds, batch_size=256, shuffle=False,
                        num_workers=2 if device.type == 'cuda' else 0,
                        pin_memory=(device.type == 'cuda'),
                    )
                    val_inf = ModelInference(model, sl, device)
                    val = val_inf.validate(val_loader)
                    print(f"val={val:.6f}")
                finally:
                    del model, val_inf
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
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


def run(argv: list[str] | None = None):
    """Parse args and call main. For use as library function."""
    import sys as _sys
    if argv is None:
        argv = _sys.argv[1:]
    parser = argparse.ArgumentParser(description='Interactive model inference')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--cpu', action='store_true', help='Force CPU (default)')
    args = parser.parse_args(argv)

    if args.gpu:
        if not gpu_health_check():
            print("GPU not available, falling back to CPU")
            main(torch.device('cpu'))
        else:
            main(torch.device('cuda'))
    else:
        main(torch.device('cpu'))


if __name__ == "__main__":
    run()
