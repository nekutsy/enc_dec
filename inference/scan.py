"""Model scanner — find trained models via Registry + filesystem fallback.

For new registry runs: reads from sessions/runs/{id}/model.pth or best.pth.
For legacy models: walks sessions/ for .pth files.

Pure domain logic. Filesystem I/O is the only side effect.
"""

import json
import os
import re

from encoding.unicode21 import UNICODE_BITS


def scan_models(sessions_dir: str = 'sessions') -> list[
        tuple[str, list[int], int, str]]:
    """Scan for trained models. Registry format preferred, legacy fallback.

    New format: sessions/runs/{run_id}/model.pth + meta.json
    Legacy format: sessions/{experiment}/{model_name}/*.pth

    Returns list of (path, sizes, n_params, label), sorted by n_params desc.
    """
    models: list[tuple[str, list[int], int, str]] = []

    # ── New format: registry runs ──
    runs_dir = os.path.join(sessions_dir, 'runs')
    if os.path.isdir(runs_dir):
        for run_id in sorted(os.listdir(runs_dir)):
            run_path = os.path.join(runs_dir, run_id)
            if not os.path.isdir(run_path):
                continue

            # Prefer best.pth, fallback to model.pth
            pth_file = None
            for name in ('best.pth', 'model.pth'):
                full = os.path.join(run_path, name)
                if os.path.isfile(full):
                    pth_file = full
                    break
            if pth_file is None:
                continue

            sizes, label = _parse_new_format(pth_file, run_path)
            if len(sizes) < 3:
                continue

            n_params = count_params(sizes)
            models.append((pth_file, sizes, n_params, label))

    # ── Legacy format ──
    _scan_legacy(sessions_dir, models)

    return sorted(models, key=lambda m: -m[2])


def _parse_new_format(pth_file: str, run_path: str) -> tuple[list[int], str]:
    """Parse sizes and label from meta.json in a registry run directory."""
    meta_path = os.path.join(run_path, 'meta.json')
    sizes: list[int] = []
    label = os.path.basename(run_path)

    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
                sizes = meta.get('layer_sizes', [])
                exp = meta.get('experiment', '')
                model_cfg = meta.get('model_config', {})
                shape = model_cfg.get('shape', 'rect')[:4]
                seq_len = model_cfg.get('seq_len', '?')
                if exp:
                    label = f"{exp}/{os.path.basename(run_path)[:8]}"
        except (json.JSONDecodeError, KeyError):
            pass

    if not sizes:
        sizes = _parse_sizes_from_filename(pth_file)

    return sizes, label


def _scan_legacy(sessions_dir: str, models: list):
    """Walk for old-format .pth files. Append to models list."""
    for root, dirs, files in os.walk(sessions_dir):
        # Skip new-format dirs
        if 'runs' in root.split(os.sep) or 'experiments' in root.split(os.sep):
            dirs.clear()
            continue

        # New format: dir-per-model with model.pth / best.pth
        pth_file = None
        if 'model.pth' in files:
            pth_file = 'model.pth'
        elif 'best.pth' in files:
            pth_file = 'best.pth'
        if pth_file is not None:
            rel = os.path.relpath(root, sessions_dir)
            parts = rel.split(os.sep)
            top_exp = parts[0] if len(parts) > 1 else rel
            full = os.path.join(root, pth_file)
            sizes = parse_key(full)
            if len(sizes) >= 3:
                n_params = count_params(sizes)
                model_name = _read_model_name(full)
                label = (f"{top_exp}/{model_name}" if model_name else top_exp)
                models.append((full, sizes, n_params, label))
            dirs.clear()
            continue

        # Old format: flat .pth files
        for f in sorted(files, reverse=True):
            if not f.endswith('.pth') or 'training_losses' in f:
                continue
            if not f.endswith('_best.pth'):
                continue
            full = os.path.join(root, f)
            sizes = parse_key(full)
            if len(sizes) < 3:
                continue
            n_params = count_params(sizes)
            model_name = _read_model_name(full)
            top_exp = os.path.relpath(root, sessions_dir)
            label = (f"{top_exp}/{model_name}" if model_name else top_exp)
            models.append((full, sizes, n_params, label))


def _parse_sizes_from_filename(path: str) -> list[int]:
    """Parse layer sizes from filename (legacy fallback)."""
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
    return sizes + list(reversed(sizes[:-1]))


def parse_key(path: str) -> list[int]:
    """Parse model layer sizes from meta.json. Falls back to filename."""
    model_dir = os.path.dirname(path)
    meta_path = os.path.join(model_dir, 'meta.json')
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
            sizes = meta.get('layer_sizes', [])
            if sizes:
                return sizes

    # Fallback: model.meta.json (old format)
    meta_path = os.path.join(model_dir, 'model.meta.json')
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
            sizes = meta.get('layer_sizes', [])
            if sizes:
                return sizes

    return _parse_sizes_from_filename(path)


def _read_model_name(path: str) -> str | None:
    """Read model_name from meta.json."""
    model_dir = os.path.dirname(path)
    for meta_name in ('meta.json', 'model.meta.json'):
        meta_path = os.path.join(model_dir, meta_name)
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
                return meta.get('model_name')
    return None


def parse_norm_from_name(path: str) -> tuple[bool, bool]:
    """Parse norm_bottleneck/norm_last from filename suffix."""
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


def load_model(path: str, sizes: list[int], device: str | None = None):
    """Load a trained Autoencoder from a checkpoint file."""
    import torch
    from model import Autoencoder

    dev = torch.device(device) if device else torch.device('cpu')

    # Read model config from meta.json
    model_dir = os.path.dirname(path)
    kwargs = {}
    for meta_name in ('meta.json', 'model.meta.json'):
        meta_path = os.path.join(model_dir, meta_name)
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            # New format: model_config is nested
            model_cfg = meta.get('model_config', meta)
            for k in ('activation', 'normalization', 'init_gain', 'dropout',
                      'norm_bottleneck', 'norm_last'):
                if k in model_cfg:
                    kwargs[k] = model_cfg[k]
            break
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
