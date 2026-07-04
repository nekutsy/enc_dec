"""Model scanner — find trained .pth files, parse architecture from filenames.

Pure domain logic. Filesystem I/O is the only side effect.
"""

import os
import re
import json

from encoding.unicode21 import UNICODE_BITS


def parse_key(path: str) -> list[int]:
    """Parse model layer sizes from meta.json. Falls back to filename."""
    # New format: dir/model.meta.json
    model_dir = os.path.dirname(path)
    meta_path = os.path.join(model_dir, 'model.meta.json')
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
            sizes = meta.get('layer_sizes', [])
            if sizes:
                return sizes
    # Fallback: old-format filename
    base = os.path.basename(path).replace('.pth', '').replace('_best', '')
    parts = base.split('_')
    sizes: list[int] = []
    for p in parts:
        if p.isdigit():
            sizes.append(int(p))
        else:
            break
    if len(sizes) < 3:
        return []
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
    """Walk sessions/ for model.pth files.

    New format: {experiment}/{model_name}/model.pth  (+ model.meta.json)
    Old format: {experiment}/*_best.pth  (fallback)

    Returns list of (path, sizes, n_params, folder), sorted by n_params desc.
    """
    models: list[tuple[str, list[int], int, str]] = []

    # New format: dir-per-model
    for root, dirs, files in os.walk(sessions_dir):
        pth_file = None
        if 'model.pth' in files:
            pth_file = 'model.pth'
        elif 'best.pth' in files:
            pth_file = 'best.pth'
        if pth_file is not None:
            rel = os.path.relpath(root, sessions_dir)
            parts = rel.split(os.sep)
            folder = parts[0] if len(parts) > 1 else rel  # top-level experiment
            full = os.path.join(root, pth_file)
            sizes = parse_key(full)
            if len(sizes) >= 3:
                n_params = count_params(sizes)
                models.append((full, sizes, n_params, folder))
            # Don't recurse deeper — model dirs are leaves
            dirs.clear()
            continue

        # Old format: flat .pth files
        for f in sorted(files, reverse=True):
            if not f.endswith('.pth') or 'training_losses' in f:
                continue
            is_best = f.endswith('_best.pth')
            if not is_best:
                continue  # only best checkpoints
            full = os.path.join(root, f)
            sizes = parse_key(full)
            if len(sizes) < 3:
                continue
            n_params = count_params(sizes)
            folder = os.path.relpath(root, sessions_dir)
            models.append((full, sizes, n_params, folder))

    return sorted(models, key=lambda m: -m[2])


def load_model(path: str, sizes: list[int],
               device: str | None = None):
    """Load a trained Autoencoder from a checkpoint file.

    Handles compiled-model state_dict (_orig_mod prefix unwrapping).
    Parses norm flags from filename.
    """
    import torch, json
    from model import Autoencoder

    dev = torch.device(device) if device else torch.device('cpu')

    # Read model config from meta.json
    model_dir = os.path.dirname(path)
    meta_path = os.path.join(model_dir, 'model.meta.json')
    kwargs = {}
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        for k in ('activation', 'normalization', 'init_gain', 'dropout',
                  'norm_bottleneck', 'norm_last'):
            if k in meta:
                kwargs[k] = meta[k]
    else:
        nb, nl = parse_norm_from_name(path)
        kwargs = {'norm_bottleneck': nb, 'norm_last': nl}

    model = Autoencoder(sizes, **kwargs).to(dev)
    state = torch.load(path, map_location=dev, weights_only=True)
    if any(k.startswith('_orig_mod.') for k in state.keys()):
        state = {k[len('_orig_mod.'):]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model
