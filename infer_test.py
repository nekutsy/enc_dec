"""Interactive inference — scan trained models, pick one, test reconstruction.

Commands:
  <#>               Load model by number
  load <#>          Same
  val <#> <#> ...   Quick validation (no model needed)
  random, r         Random sample → reconstruct
  full [pos]        First 20 windows from dataset (sequential)
  enc <text>        Encode text → latent vector (also enc @pos)
  dec               Decode stored latent → text
  z, latent         Show full stored latent vector
  <text>            Reconstruct text (auto-pads to seq_len)
  q, quit           Exit

Chaining:
  Commands can be chained, e.g.:
    dec enc random     — encode random sample, then decode it
    enc random dec     — same (order reversed — reads left-to-right naturally)
    dec enc @400000    — encode position 400k, then decode

Options:
  --gpu             Use GPU (default: CPU for interactive, auto for val)
  --cpu             Force CPU (default for interactive)
"""

import argparse, torch, sys, os, random, glob, re, csv
sys.path.insert(0, os.path.dirname(__file__))

from configs import UNICODE_BITS
from model import Autoencoder
from data import _build_full_bits, load_text, vec2seq, chars_to_bits, prepare_data
import numpy as np
from training import _validate
import torch.nn as nn
import torch.nn.functional as F

# Module-level device — overridden by main()
device = torch.device("cpu")

CHAIN_VERBS = {'enc', 'dec', 'z', 'latent', 'r', 'random', 'full'}


def _parse_key(path):
    """Parse model sizes from filename like '2688_7644_7644_128_sweep_n2.pth'.
    
    Encoder half is a contiguous block of pure-numeric segments at the start.
    Stops at the first non-digit segment (suffix like 'gain0_1', 'nbF_nlF', etc).
    Reconstructs full architecture by mirroring decoder.
    """
    base = os.path.basename(path).replace('.pth', '').replace('_best', '')
    parts = base.split('_')
    sizes = []
    for p in parts:
        if p.isdigit():
            sizes.append(int(p))
        else:
            break  # stop at first non-digit segment (model name suffix)
    if len(sizes) < 3:
        return []
    # Encoder half → full mirror architecture
    decoder = list(reversed(sizes[:-1]))
    return sizes + decoder


def _scan_models(sessions_dir="sessions"):
    """Walk sessions/ for .pth files, return list of (path, sizes, n_params, folder).
    
    Picks _best.pth over .pth when both exist in the same folder.
    Deduplicates by (folder, clean_base).
    """
    candidates = {}  # (folder, clean_base) → (path, is_best)
    for root, dirs, files in os.walk(sessions_dir):
        for f in files:
            if not f.endswith('.pth') or 'training_losses' in f:
                continue
            full = os.path.join(root, f)
            base = f.replace('.pth', '')
            is_best = base.endswith('_best')
            clean_base = base[:-5] if is_best else base
            folder = os.path.relpath(root, sessions_dir)
            key = (folder, clean_base)
            # Prefer _best version within same folder
            if key not in candidates or is_best:
                candidates[key] = (full, is_best)
    
    models = []
    for (folder, clean_base), (full, is_best) in candidates.items():
        sizes = _parse_key(full)
        if len(sizes) < 3:
            continue
        n_params = sum(
            sizes[i] * sizes[i+1] + sizes[i+1] + 2 * sizes[i+1]
            for i in range(len(sizes) - 1)
        )
        models.append((full, sizes, n_params, folder))
    return sorted(models, key=lambda m: -m[2])


def _parse_norm_from_name(name):
    """Parse norm_bottleneck/norm_last from filename suffix.
    
    Looks for patterns like 'nbT_nlT', 'nbF_nlF', etc.
    Returns (norm_bottleneck, norm_last) defaulting to current best (False, False).
    """
    import re
    m = re.search(r'nb([TF])_nl([TF])', name)
    if m:
        return m.group(1) == 'T', m.group(2) == 'T'
    return False, False  # current best default


def _load_model_from_path(path, sizes):
    """Load a trained model given file path, sizes, and optional norm flags."""
    nb, nl = _parse_norm_from_name(path)
    model = Autoencoder(sizes, norm_bottleneck=nb, norm_last=nl).to(device)
    state = torch.load(path, map_location=device, weights_only=True)
    if any(k.startswith('_orig_mod.') for k in state.keys()):
        state = {k[len('_orig_mod.'):]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def _compute_val_loss(model, text, seq_len):
    """Quick validation pass to evaluate model quality."""
    train_ds, val_ds = prepare_data(text, seq_len)
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            num_workers=2 if device.type == 'cuda' else 0,
                            pin_memory=(device.type == 'cuda'))
    criterion = nn.BCEWithLogitsLoss()
    return _validate(model, val_loader, criterion, device)


def _bits_from_chunk(chunk, sl):
    """Pad to sl and convert to bit tensor [1, sl*21]."""
    padded = chunk + '\0' * (sl - len(chunk))
    codes = np.array([ord(ch) if ch != '\0' else 0 for ch in padded], dtype=np.uint32)
    bits_np = chars_to_bits(codes).ravel()
    return torch.from_numpy(bits_np).float().unsqueeze(0).to(device), padded


def _reconstruct(model, chunk, sl):
    """Reconstruct a single window of size sl. Pads with \\0 if needed."""
    inp_t, padded = _bits_from_chunk(chunk, sl)
    with torch.inference_mode():
        out_logits = model(inp_t).squeeze(0).cpu().numpy()
    out_bits = torch.sigmoid(torch.from_numpy(out_logits)).numpy()
    rec = vec2seq(out_bits)
    errors = sum(1 for a, b in zip(padded, rec) if a != b)
    codes = np.array([ord(ch) if ch != '\0' else 0 for ch in padded], dtype=np.uint32)
    bit_err = np.abs(chars_to_bits(codes).ravel() - out_bits).sum()
    return rec, errors, bit_err


def _encode_text(model, chunk, sl):
    """Encode text chunk → latent vector (numpy)."""
    inp_t, _ = _bits_from_chunk(chunk, sl)
    with torch.inference_mode():
        latent = model.encode(inp_t).squeeze(0).cpu().numpy()
    return latent


def _decode_latent(model, latent):
    """Decode latent vector → reconstructed text."""
    z = torch.from_numpy(latent).float().unsqueeze(0).to(device)
    with torch.inference_mode():
        out_logits = model.decode(z).squeeze(0).cpu().numpy()
    out_bits = torch.sigmoid(torch.from_numpy(out_logits)).numpy()
    return vec2seq(out_bits)


# Verbs that accept text arguments (random/r/@pos are valid text inputs)
_ARGS_VERBS = {'enc', 'full'}
# Verbs that never make sense as text input → chain separators
_STRUCT_VERBS = {'dec', 'z', 'latent'}


def _parse_chain(cmd_str):
    """Parse chain: 'dec enc random' → [('dec', ''), ('enc', 'random')].

    Verbs that take args ('enc', 'full') consume tokens until a structural
    verb ('dec', 'z', 'latent') or another args-verb ('enc', 'full') is hit.
    Verbs that don't take args ('dec', 'z', 'r', 'random') consume nothing.
    """
    tokens = cmd_str.split()
    if not tokens or tokens[0].lower() not in CHAIN_VERBS:
        return None
    sub = []
    i = 0
    while i < len(tokens):
        verb = tokens[i].lower()
        i += 1
        args_parts = []
        if verb in _ARGS_VERBS:
            while i < len(tokens) and tokens[i].lower() not in _ARGS_VERBS and tokens[i].lower() not in _STRUCT_VERBS:
                args_parts.append(tokens[i])
                i += 1
        sub.append((verb, ' '.join(args_parts)))
    return sub


def _random_chunk(text, sl, n_chars, rng):
    """Pick a random window of sl chars from text."""
    pos = rng.randint(0, max(0, n_chars - sl - 1))
    return pos, text[pos:pos + sl]


def main(device_override=None):
    global device
    if device_override is not None:
        device = device_override
    else:
        device = torch.device("cpu")  # default: CPU for interactive use
    print(f"Device: {device}")

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
    print(f"\nCommands: <#> load | val <#> ... | random | full | enc <text> | dec | z | q")
    print(f"Chains: dec enc random | enc random dec | enc @1000 dec | ...")
    text = load_text()
    full_bits = _build_full_bits(text)
    n_chars = full_bits.numel() // UNICODE_BITS

    loaded_model = None
    loaded_sl = 0
    loaded_n_params = 0
    last_latent = None
    last_latent_sl = 0
    rng = random.Random()

    # ── Handler callables ──
    def _resolve_input(input_text):
        """Resolve 'random', '@pos', or verbatim text → (display_note, chunk)."""
        lt = input_text.lower()
        if lt in ('random', 'r'):
            pos, chunk = _random_chunk(text, sl, n_chars, rng)
            return f"@random {pos}", chunk
        if input_text.startswith('@') and input_text[1:].isdigit():
            pos = int(input_text[1:])
            chunk = text[pos:pos + sl]
            return f"@pos {pos}", chunk
        return None, input_text

    def _cmd_enc(args_str):
        nonlocal last_latent, last_latent_sl
        if not args_str:
            print("Usage: enc <text|random|@pos>")
            return
        note, chunk = _resolve_input(args_str)
        if note:
            if chunk:
                print(f"{note}: {chunk[:60]!r}")
            else:
                print(note)
        last_latent = _encode_text(model, chunk, sl)
        last_latent_sl = sl
        vals = ', '.join(f'{v:+.4f}' for v in last_latent[:16])
        more = f' ... +{len(last_latent) - 16} more' if len(last_latent) > 16 else ''
        print(f"latent [{len(last_latent)}]: [{vals}{more}]")
        print(f"  range: [{last_latent.min():+.4f}, {last_latent.max():+.4f}]  "
              f"mean={last_latent.mean():+.4f}  std={last_latent.std():.4f}")

    def _cmd_dec():
        if last_latent is None:
            print("No latent stored. Use 'enc <text>' first.")
            return
        rec = _decode_latent(model, last_latent)
        cleaned = rec.rstrip('\0')[:last_latent_sl]
        print(cleaned)

    def _cmd_z():
        if last_latent is None:
            print("No latent stored. Use 'enc <text>' first.")
            return
        print(f"latent [{len(last_latent)}]:")
        for i in range(0, len(last_latent), 16):
            row = last_latent[i:i+16]
            print('  ' + ' '.join(f'{v:+.4f}' for v in row))

    def _cmd_random():
        nonlocal last_latent, last_latent_sl
        pos, chunk = _random_chunk(text, sl, n_chars, rng)
        print(f"@{pos}: {chunk!r}")
        rec, errors, bit_err = _reconstruct(model, chunk, sl)
        print(rec[:sl])
        if errors:
            print(f"errors: {errors}/{sl} chars | {bit_err:.1f} bits")
        last_latent = _encode_text(model, chunk, sl)
        last_latent_sl = sl

    def _cmd_full(args_str):
        nonlocal last_latent, last_latent_sl
        parts = args_str.split() if args_str else []
        start_pos = int(parts[0]) if parts and parts[0].isdigit() else 0
        if start_pos >= n_chars:
            print(f"Position {start_pos} beyond text end ({n_chars})")
            return
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
            if w == 0:
                last_latent = _encode_text(model, chunk, sl)
                last_latent_sl = sl
        print(f"total: {total_err_c}/{total_c} char errors | {total_err_b:.1f} bit errors")

    def _cmd_text(inp):
        nonlocal last_latent, last_latent_sl
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
            last_latent = _encode_text(model, inp, sl)
            last_latent_sl = sl

    CHAIN_DISPATCH = {
        'enc': _cmd_enc,
        'dec': _cmd_dec,
        'z': _cmd_z, 'latent': _cmd_z,
        'r': _cmd_random, 'random': _cmd_random,
        'full': _cmd_full,
    }

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

        # ── Command chain (e.g. 'dec enc random' or 'enc random dec') ──
        chain = _parse_chain(cmd)
        if chain is not None:
            if loaded_model is None:
                print("No model loaded. Use '<#>' or 'load <#>' first.")
                continue
            # Sort producers (enc, random, full) before consumers (dec, z)
            # so both 'dec enc random' and 'enc random dec' work identically.
            _PRODUCER_ORDER = {'enc': 0, 'full': 0, 'r': 0, 'random': 0,
                              'dec': 1, 'z': 1, 'latent': 1}
            ordered = sorted(chain, key=lambda x: _PRODUCER_ORDER.get(x[0], 2))
            for verb, args in ordered:
                handler = CHAIN_DISPATCH.get(verb)
                if handler:
                    handler(args)
            continue

        # ── Everything below needs a loaded model ──
        if loaded_model is None:
            print("No model loaded. Use '<#>' or 'load <#>' first.")
            continue

        model = loaded_model
        sl = loaded_sl

        # ── Single command dispatch ──
        if cmd.lower().startswith('enc'):
            args_str = cmd[3:].strip() if cmd.lower().startswith('enc ') else ''
            _cmd_enc(args_str)
            continue

        if cmd.lower() == 'dec':
            _cmd_dec()
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

        # ── Text reconstruction ──
        _cmd_text(cmd)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Interactive model inference')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--cpu', action='store_true', help='Force CPU (default)')
    args = parser.parse_args()
    if args.gpu and torch.cuda.is_available():
        main(torch.device("cuda"))
    else:
        if args.gpu:
            print("CUDA not available, falling back to CPU")
        main(torch.device("cpu"))
