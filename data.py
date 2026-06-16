"""Data loading, Unicode-21 encoding, and dataset utilities."""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from configs import UNICODE_BITS

FULL_BITS_CACHE = "data/cache/full_bits.pt"


def load_text(data_dir="data/dataset"):
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    texts = []
    if not txt_files:
        print(f"No .txt files found in {data_dir}, using dummy text.")
        return "Это тестовый текст для автоэнкодера. " * 50
    print(f"Found {len(txt_files)} .txt file(s) in {data_dir}:")
    for path in txt_files:
        filename = os.path.basename(path)
        with open(path, "r", encoding="utf8") as f:
            content = f.read()
            texts.append(content)
            print(f"  - {filename}: {len(content)} characters")
    return "".join(texts)


def _build_full_bits(text):
    """Build and cache a single float32 tensor of ALL character bits — 1D, ~2GB.

    This is the single source of truth. Sliding-window datasets create
    torch.as_strided views into this tensor without copying.
    """
    os.makedirs("data/cache", exist_ok=True)
    if os.path.exists(FULL_BITS_CACHE):
        return torch.load(FULL_BITS_CACHE, weights_only=True)

    codes = np.array([ord(ch) for ch in text], dtype=np.uint32)
    bits = np.zeros((len(codes), UNICODE_BITS), dtype=np.float32)
    for i in range(UNICODE_BITS):
        bits[:, i] = (codes >> (UNICODE_BITS - 1 - i)) & 1
    full_flat = torch.from_numpy(bits.ravel())
    torch.save(full_flat, FULL_BITS_CACHE)
    print(f"  Built full_bits cache: {full_flat.shape} ({full_flat.numel() * 4 / 1e9:.2f} GB)")
    return full_flat


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
        # as_strided view: (n_windows, window_bits) with stride (21, 1)
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
        w = self._windows[self._indices[idx]].clone()  # clone → contiguous for DataLoader
        return w, w  # (input, target) for autoencoder


def chars_to_bits(codes: np.ndarray) -> np.ndarray:
    """Vectorized: (N,) uint32 codepoints → (N, UNICODE_BITS) float32 bits."""
    bits = np.zeros((len(codes), UNICODE_BITS), dtype=np.float32)
    for i in range(UNICODE_BITS):
        bits[:, i] = (codes >> (UNICODE_BITS - 1 - i)) & 1
    return bits


def seq2vec(seq: str, max_bits: int):
    max_symbols = max_bits // UNICODE_BITS
    codes = np.array([ord(ch) for ch in seq[:max_symbols]], dtype=np.uint32)
    used = len(codes)
    bits = np.zeros(max_bits, dtype=np.float32)
    if used > 0:
        bits[:used * UNICODE_BITS] = chars_to_bits(codes).ravel()
    return bits.tolist(), used


def vec2seq(vec):
    arr = np.asarray(vec, dtype=np.float32).reshape(-1, UNICODE_BITS)
    powers = 2 ** np.arange(UNICODE_BITS - 1, -1, -1)
    codepoints = ((arr > 0.5) @ powers).astype(int)
    valid_codes = codepoints[codepoints > 0]
    return ''.join(chr(c) for c in valid_codes)


def split_into_chunks(text: str, max_bits: int):
    chunks = []
    i = 0
    max_symbols = max_bits // UNICODE_BITS
    while i < len(text):
        chunk_text = text[i:i + max_symbols]
        bits, used = seq2vec(chunk_text, max_bits)
        chunks.append((chunk_text, bits))
        i += used
    return chunks


def prepare_data(text: str, config):
    """Build sliding-window dataset and return (train_ds, val_ds) — lazy.

    Uses shared full_bits cache (~2 GB) + per-seq_len as_strided view.
    Returns SlidingWindowDataset objects with non-overlapping indices.
    Each __getitem__ clones a single window → DataLoader handles batching.
    """
    full_bits = _build_full_bits(text)
    dataset = SlidingWindowDataset(full_bits, config.seq_len)
    n = len(dataset)
    indices = torch.randperm(n)
    train_size = int(n * config.train_ratio)
    train_ds = SlidingWindowDataset(full_bits, config.seq_len, indices=indices[:train_size])
    val_ds = SlidingWindowDataset(full_bits, config.seq_len, indices=indices[train_size:])
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
            x_batch, _ = batch  # sliding window returns (input, target)
            x_batch = x_batch.to(device)
            z = model.encode(x_batch)
            latents.append(z.cpu())
    latents = torch.cat(latents, dim=0)
    torch.save(latents, output_path)
    print(f"Exported latent vectors: {latents.shape} -> {output_path}")


def load_latent_vectors(path="data/latent/latent_vectors.pt"):
    return torch.load(path, map_location='cpu', weights_only=True)
