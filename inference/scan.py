"""Model scanner — find trained .pth files, parse architecture from filenames.

Pure domain logic. Filesystem I/O is the only side effect.
"""

import os
import re

from encoding.unicode21 import UNICODE_BITS


def parse_key(path: str) -> list[int]:
    """Parse model layer sizes from filename.

    Filename format: '2688_7644_7644_128_sweep_n2_best.pth'
    Encoder half → numeric segments → mirror for decoder → full architecture.

    Returns [] if filename doesn't contain a valid architecture.
    """
    base = os.path.basename(path).replace('.pth', '').replace('_best', '')
    parts = base.split('_')
    sizes: list[int] = []
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


def parse_norm_from_name(path: str) -> tuple[bool, bool]:
    """Parse norm_bottleneck/norm_last from filename suffix.

    Looks for patterns like 'nbT_nlT', 'nbF_nlF', etc.
    Returns (norm_bottleneck, norm_last), defaulting to (False, False).
    """
    m = re.search(r'nb([TF])_nl([TF])', path)
    if m:
        return m.group(1) == 'T', m.group(2) == 'T'
    return False, False


def count_params(sizes: list[int]) -> int:
    """Count Linear + BatchNorm1d parameters for the given layer sizes."""
    n = 0
    for i in range(len(sizes) - 1):
        n += sizes[i] * sizes[i + 1] + sizes[i + 1]
        n += 2 * sizes[i + 1]
    return n


def scan_models(sessions_dir: str = 'sessions') -> list[
        tuple[str, list[int], int, str]]:
    """Walk sessions/ for .pth files.

    Returns list of (path, sizes, n_params, folder), sorted by n_params desc.
    Prefers _best.pth over .pth when both exist in the same folder.
    Deduplicates by (folder, clean_base).
    """
    candidates: dict[tuple[str, str], tuple[str, bool]] = {}
    # (folder, clean_base) → (path, is_best)

    for root, _, files in os.walk(sessions_dir):
        for f in files:
            if not f.endswith('.pth') or 'training_losses' in f:
                continue
            full = os.path.join(root, f)
            base = f.replace('.pth', '')
            is_best = base.endswith('_best')
            clean_base = base[:-5] if is_best else base
            folder = os.path.relpath(root, sessions_dir)
            key = (folder, clean_base)
            if key not in candidates or is_best:
                candidates[key] = (full, is_best)

    models: list[tuple[str, list[int], int, str]] = []
    for (folder, _), (full, _) in candidates.items():
        sizes = parse_key(full)
        if len(sizes) < 3:
            continue
        n_params = count_params(sizes)
        models.append((full, sizes, n_params, folder))

    return sorted(models, key=lambda m: -m[2])


def load_model(path: str, sizes: list[int],
               device: str | None = None):
    """Load a trained Autoencoder from a checkpoint file.

    Handles compiled-model state_dict (_orig_mod prefix unwrapping).
    Parses norm flags from filename.
    """
    import torch
    from model import Autoencoder

    dev = torch.device(device) if device else torch.device('cpu')
    nb, nl = parse_norm_from_name(path)
    model = Autoencoder(sizes, norm_bottleneck=nb, norm_last=nl).to(dev)
    state = torch.load(path, map_location=dev, weights_only=True)
    if any(k.startswith('_orig_mod.') for k in state.keys()):
        state = {k[len('_orig_mod.'):]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model
