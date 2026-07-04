"""Data loading, Unicode-21 encoding, and dataset utilities."""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from configs import UNICODE_BITS
from encoding.unicode21 import (
    chars_to_bits,
    bits_to_chars as _bits_to_chars,
    seq_to_vec,
    vec_to_seq,
    split_into_chunks,
    pack_bits_uint8,
    unpack_uint8_to_float,
)

FULL_BITS_CACHE = "data/cache/full_bits.u8"   # uint8 packed — 8 bits/byte (~0.13 GB)
OLD_FLOAT_CACHE = "data/cache/full_bits.pt"   # legacy float32 (auto-migrated)


def load_text(data_dir="data/dataset", verbose=False):
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    texts = []
    if not txt_files:
        print(f"No .txt files found in {data_dir}, using dummy text.")
        return "Это тестовый текст для автоэнкодера. " * 50
    total_chars = 0
    for path in txt_files:
        with open(path, "r", encoding="utf8") as f:
            content = f.read()
            texts.append(content)
            total_chars += len(content)
    if verbose:
        print(f"Found {len(txt_files)} .txt file(s) in {data_dir}:")
        for path in txt_files:
            filename = os.path.basename(path)
            with open(path, "r", encoding="utf8") as f:
                print(f"  - {filename}: {len(f.read())} characters")
    else:
        print(f"Loaded {len(txt_files)} text files — {total_chars:,} characters total")
    return "".join(texts)


# Re-exported from encoding.unicode21 for backward compatibility
_pack_bits_uint8 = pack_bits_uint8
_unpack_uint8_to_float = unpack_uint8_to_float


def _build_full_bits(text):
    """Build/cache packed uint8 on disk; return float32 tensor for sliding windows.

    Disk: uint8 packed (~0.13 GB for 50M chars × 21 bits).
    RAM: float32 (~4.2 GB) — required for torch.as_strided.
    Auto-migrates legacy float32 cache.
    """
    os.makedirs("data/cache", exist_ok=True)
    codes = np.array([ord(ch) for ch in text], dtype=np.uint32)
    total_bits = len(codes) * UNICODE_BITS

    # Existing uint8 cache
    if os.path.exists(FULL_BITS_CACHE):
        packed = torch.load(FULL_BITS_CACHE, map_location='cpu', weights_only=True)
        print(f"  Loaded uint8 cache: {packed.numel() / 1e6:.1f} MB")
        return _unpack_uint8_to_float(packed, total_bits)

    # Legacy float32 cache → migrate
    if os.path.exists(OLD_FLOAT_CACHE):
        print("  Migrating old float32 cache → uint8...")
        packed = torch.from_numpy(_pack_bits_uint8(codes))
        torch.save(packed, FULL_BITS_CACHE)
        os.remove(OLD_FLOAT_CACHE)
        print(f"  Migrated: {packed.numel() / 1e6:.1f} MB uint8")
        return _unpack_uint8_to_float(packed, total_bits)

    # Fresh build
    packed = torch.from_numpy(_pack_bits_uint8(codes))
    torch.save(packed, FULL_BITS_CACHE)
    n_gb = packed.numel() / 1e9
    print(f"  Built uint8 cache: {n_gb:.2f} GB (unpacked: {total_bits * 4 / 1e9:.2f} GB float32)")
    return _unpack_uint8_to_float(packed, total_bits)


class SlidingWindowDataset(Dataset):
    """Sliding window over character bits — stride=1 character.

    Each window captures seq_len consecutive characters. With stride=1,
    each character appears in seq_len different windows → no data starvation
    for long seq_lens.
    """

    def __init__(self, full_bits, seq_len, indices=None):
        self.seq_len = seq_len
        self.window_bits = seq_len * UNICODE_BITS
        n_total = full_bits.numel() // UNICODE_BITS - seq_len + 1
        # as_strided view: (n_windows, window_bits) with stride (UNICODE_BITS, 1)
        self._windows = torch.as_strided(
            full_bits,
            size=(n_total, self.window_bits),
            stride=(UNICODE_BITS, 1),
        )
        self._indices = (
            indices if indices is not None
            else torch.arange(n_total)
        )

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        # empty_like + copy_ is cheaper than clone() for strided views
        w = torch.empty(self.window_bits, dtype=torch.float32, device='cpu')
        w.copy_(self._windows[self._indices[idx]])
        return w, w


class NoisyDataset(Dataset):
    """Wraps a SlidingWindowDataset — adds uint21-level Gaussian noise to inputs.

    For each character (21-bit group) in the input window:
      1. Convert 21 bits → uint21 value
      2. With probability noise_prob, add N(0, noise_std²)
      3. Round to nearest integer, clamp to [0, 2²¹−1]
      4. Convert back to 21 bits

    Target (y) stays clean — model must denoise.
    """

    def __init__(self, base_dataset, noise_prob=0.05, noise_std=3.0, seed=None):
        self.base = base_dataset
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed)
        self._powers = 2 ** torch.arange(UNICODE_BITS, dtype=torch.float32)
        self._max_val = (1 << UNICODE_BITS) - 1

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        noisy_x = self._add_noise(x, self.base.seq_len)
        return noisy_x, y

    def _add_noise(self, bits, seq_len):
        chars = bits.view(seq_len, UNICODE_BITS)
        uints = (chars @ self._powers).long()

        r = self._rng.random(seq_len)
        noise_mask = torch.from_numpy(r < self.noise_prob).to(torch.long)
        noise_vals = torch.from_numpy(
            self._rng.normal(0, self.noise_std, seq_len)).float()

        noisy = uints.float() + noise_mask.float() * noise_vals
        noisy = torch.round(noisy).long().clamp(0, self._max_val)

        bit_positions = torch.arange(UNICODE_BITS)
        noisy_bits = ((noisy.unsqueeze(-1) >> bit_positions) & 1).float().flatten()
        return noisy_bits


# Re-exported from encoding.unicode21 for backward compatibility
chars_to_bits = chars_to_bits  # already imported above
vec2seq = vec_to_seq
seq2vec = seq_to_vec
split_into_chunks = split_into_chunks


def prepare_data(text: str, seq_len: int, train_ratio: float = 0.99):
    """Build sliding-window dataset and return (train_ds, val_ds).

    Uses shared uint8-packed cache + per-seq_len as_strided view.
    Returns SlidingWindowDataset objects with non-overlapping indices.
    """
    full_bits = _build_full_bits(text)
    dataset = SlidingWindowDataset(full_bits, seq_len)
    n = len(dataset)
    indices = torch.randperm(n)
    train_size = int(n * train_ratio)
    train_ds = SlidingWindowDataset(full_bits, seq_len, indices=indices[:train_size])
    val_ds = SlidingWindowDataset(full_bits, seq_len, indices=indices[train_size:])
    return train_ds, val_ds


def export_latent_vectors(model, text, config, device, output_path="data/latent/latent_vectors.pt"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()
    full_bits = _build_full_bits(text)
    dataset = SlidingWindowDataset(full_bits, config.seq_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    latents = []
    with torch.inference_mode():
        for batch in loader:
            x_batch, _ = batch
            x_batch = x_batch.to(device)
            z = model.encode(x_batch)
            latents.append(z.cpu())
    latents = torch.cat(latents, dim=0)
    torch.save(latents, output_path)
    print(f"Exported latent vectors: {latents.shape} -> {output_path}")


def load_latent_vectors(path="data/latent/latent_vectors.pt"):
    return torch.load(path, map_location='cpu', weights_only=True)
