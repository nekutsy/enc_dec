"""Model scanner — find trained models via Registry + filesystem fallback.

For new registry runs: reads from sessions/runs/{id}/model.pth or best.pth.
For legacy models: walks sessions/ for .pth files.
"""

import csv
import json
import os
import re
from dataclasses import dataclass

from encoding.unicode21 import UNICODE_BITS
from model.architecture import count_params


@dataclass
class ModelInfo:
    """Scanned model metadata for display and loading."""
    path: str
    sizes: list[int]
    n_params: int
    label: str
    run_id: str = ''
    enc_n: int | None = None
    dec_n: int | None = None
    train_loss: float | None = None
    val_loss: float | None = None

    @property
    def n_hidden_str(self) -> str:
        """Human-readable layer count: '6' for symmetric, '3/5' for asymmetric."""
        enc = self.enc_n
        dec = self.dec_n
        if enc is not None and dec is not None:
            if enc == dec:
                return str(enc)
            return f'{enc}/{dec}'
        mid = len(self.sizes) // 2
        return str(mid)


def scan_models(sessions_dir: str = 'sessions') -> list[ModelInfo]:
    """Scan for trained models. Registry format preferred, legacy fallback.

    Returns list of ModelInfo, sorted by n_params desc.
    """
    models: list[ModelInfo] = []

    runs_dir = os.path.join(sessions_dir, 'runs')
    if os.path.isdir(runs_dir):
        for run_name in sorted(os.listdir(runs_dir)):
            run_path = os.path.join(runs_dir, run_name)
            if not os.path.isdir(run_path) or os.path.islink(run_path):
                continue

            pth_file = None
            for name in ('best.pth', 'model.pth'):
                full = os.path.join(run_path, name)
                if os.path.isfile(full):
                    pth_file = full
                    break
            if pth_file is None:
                continue

            info = _parse_new_format(pth_file, run_path)
            if info is None:
                continue
            models.append(info)

    _scan_legacy(sessions_dir, models)
    return sorted(models, key=lambda m: -m.n_params)


def _parse_new_format(pth_file: str, run_path: str) -> ModelInfo | None:
    """Parse ModelInfo from meta.json + result.json in a registry run directory."""
    meta_path = os.path.join(run_path, 'meta.json')
    sizes: list[int] = []
    label = os.path.basename(run_path)
    run_id = ''
    enc_n = dec_n = None

    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
                sizes = meta.get('layer_sizes', [])
                run_id = meta.get('run_id', '')
                enc_n = meta.get('enc_n')
                dec_n = meta.get('dec_n')
                exp = meta.get('experiment', '')
                model_name = meta.get('model_name', '')
                if model_name and exp:
                    label = f'{exp}/{model_name}'
                elif model_name:
                    label = model_name
                elif exp:
                    rid = run_id[:6] if run_id else ''
                    label = f'{exp}/{rid}'
        except (json.JSONDecodeError, KeyError):
            pass

    if not sizes:
        sizes = _parse_sizes_from_filename(pth_file)
    if len(sizes) < 3:
        return None

    n_params = count_params(sizes)
    train_loss, val_loss = _read_result(run_path)
    return ModelInfo(
        path=pth_file, sizes=sizes, n_params=n_params, label=label,
        run_id=run_id, enc_n=enc_n, dec_n=dec_n,
        train_loss=train_loss, val_loss=val_loss,
    )


def _read_result(run_path: str) -> tuple[float | None, float | None]:
    """Read train_loss and val_loss from result.json or log.csv."""
    result_path = os.path.join(run_path, 'result.json')
    if os.path.isfile(result_path):
        try:
            with open(result_path) as f:
                result = json.load(f)
                return result.get('final_train_loss'), result.get('final_val_loss')
        except (json.JSONDecodeError, KeyError):
            pass
    return _read_loss_from_csv(run_path)


def _read_loss_from_csv(run_path: str) -> tuple[float | None, float | None]:
    """Read last train_loss and val_loss from the log CSV."""
    for name in sorted(os.listdir(run_path)):
        if not name.endswith('.csv'):
            continue
        csv_path = os.path.join(run_path, name)
        try:
            with open(csv_path) as f:
                reader = csv.reader(f)
                header = next(reader, [])
                last_row = None
                for row in reader:
                    last_row = row
                if last_row and header:
                    try:
                        ti = header.index('train_loss')
                        tl = float(last_row[ti]) if ti < len(last_row) else None
                        vl = None
                        if 'val_loss' in header:
                            vi = header.index('val_loss')
                            if vi < len(last_row) and last_row[vi]:
                                vl = float(last_row[vi])
                        return tl, vl
                    except (ValueError, IndexError):
                        pass
        except Exception:
            pass
    return None, None


def _scan_legacy(sessions_dir: str, models: list[ModelInfo]):
    """Walk for old-format .pth files. Append to models list."""
    for root, dirs, files in os.walk(sessions_dir):
        if 'runs' in root.split(os.sep) or 'experiments' in root.split(os.sep):
            dirs.clear()
            continue

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
                models.append(ModelInfo(
                    path=full, sizes=sizes, n_params=n_params, label=label,
                ))
            dirs.clear()
            continue

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
            models.append(ModelInfo(
                path=full, sizes=sizes, n_params=n_params, label=label,
            ))


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


def load_model(path: str, sizes: list[int], device: str | None = None):
    """Load a trained Autoencoder from a checkpoint file.

    Weights are loaded to CPU first, then the model is moved to the target device
    to avoid double GPU allocation (model init on GPU + state dict on GPU).
    """
    import torch
    from model import Autoencoder

    dev = torch.device(device) if device else torch.device('cpu')

    model_dir = os.path.dirname(path)
    meta = {}
    kwargs = {}
    for meta_name in ('meta.json', 'model.meta.json'):
        meta_path = os.path.join(model_dir, meta_name)
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            model_cfg = meta.get('model_config', meta)
            for k in ('activation', 'normalization', 'init_gain', 'dropout',
                      'norm_bottleneck', 'norm_last', 'residual', 'residual_norm'):
                if k in model_cfg:
                    kwargs[k] = model_cfg[k]
            break
    else:
        nb, nl = parse_norm_from_name(path)
        kwargs = {'norm_bottleneck': nb, 'norm_last': nl}

    enc_n = meta.get('enc_n') or kwargs.get('enc_n')
    if enc_n is not None:
        kwargs['enc_n'] = enc_n

    model = Autoencoder(sizes, **kwargs)
    state = torch.load(path, map_location='cpu', weights_only=True)
    if any(k.startswith('_orig_mod.') for k in state.keys()):
        state = {k[len('_orig_mod.'):]: v for k, v in state.items()}
    model.load_state_dict(state)
    del state
    model = model.to(dev)
    model.eval()
    return model
