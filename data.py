"""Data loading, Unicode-21 encoding, and dataset utilities."""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from configs import UNICODE_BITS

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


def _pack_bits_uint8(codes: np.ndarray) -> np.ndarray:
    """Pack Unicode codepoint bits into uint8: 8 bits per byte, little-endian.

    codes: (N,) uint32 → packed uint8 with ceil(N*UNICODE_BITS/8) bytes.
    """
    total_bits = len(codes) * UNICODE_BITS
    bits = np.zeros(total_bits, dtype=np.uint8)
    for i in range(UNICODE_BITS):
        bits[i::UNICODE_BITS] = (codes >> (UNICODE_BITS - 1 - i)) & 1
    n_padded = ((total_bits + 7) // 8) * 8
    padded = np.zeros(n_padded, dtype=np.uint8)
    padded[:total_bits] = bits
    return np.packbits(padded.reshape(-1, 8), axis=1, bitorder='little').ravel()


def _unpack_uint8_to_float(packed, total_bits: int) -> torch.Tensor:
    """Unpack uint8 → float32 tensor of exact 0.0/1.0 values."""
    if isinstance(packed, torch.Tensor):
        packed = packed.cpu().numpy()
    packed = np.asarray(packed, dtype=np.uint8)
    if len(packed) == 0:
        return torch.zeros(total_bits, dtype=torch.float32)
    unpacked = np.unpackbits(packed, bitorder='little')
    return torch.from_numpy(unpacked[:total_bits].astype(np.float32))


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

    Uses shared uint8-packed cache + per-seq_len as_strided view.
    Returns SlidingWindowDataset objects with non-overlapping indices.
    Each __getitem__ materializes a single window → DataLoader handles batching.
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
            x_batch, _ = batch
            x_batch = x_batch.to(device)
            z = model.encode(x_batch)
            latents.append(z.cpu())
    latents = torch.cat(latents, dim=0)
    torch.save(latents, output_path)
    print(f"Exported latent vectors: {latents.shape} -> {output_path}")


def load_latent_vectors(path="data/latent/latent_vectors.pt"):
    return torch.load(path, map_location='cpu', weights_only=True)
