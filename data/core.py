"""Data loading, Unicode-21 encoding, and dataset utilities."""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from configs import UNICODE_BITS
from encoding.unicode21 import pack_bits_uint8, unpack_uint8_to_float


def load_text(data_dir="data/dataset", verbose=False):
    """Scan directory recursively, return one string per .txt file.

    Use **/*.txt with recursive=True to include subdirectories.
    Files are sorted by path for deterministic ordering.
    Each file is loaded separately — caller (prepare_data) handles
    null-padding between files to prevent sliding windows from crossing
    file boundaries.
    """
    txt_files = sorted(glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True))
    if not txt_files:
        print(f"No .txt files found in {data_dir}, using dummy text.")
        return ["Это тестовый текст для автоэнкодера. " * 50]
    texts: list[str] = []
    total_chars = 0
    for path in txt_files:
        try:
            with open(path, "r", encoding="utf8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(path, "rb") as f:
                raw = f.read()
            for enc in ["windows-1251", "koi8-r", "cp866"]:
                try:
                    content = raw.decode(enc)
                    if verbose:
                        print(f"  [fallback: {enc}] {os.path.relpath(path, data_dir)}")
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
            else:
                raise RuntimeError(f"Cannot decode {path}")
        texts.append(content)
        total_chars += len(content)
    if verbose:
        print(f"Found {len(txt_files)} .txt file(s) in {data_dir}:")
        for path, content in zip(txt_files, texts):
            print(f"  - {os.path.relpath(path, data_dir)}: {len(content)} characters")
    else:
        print(f"Loaded {len(txt_files)} text files — {total_chars:,} characters total")
    return texts





def _build_full_bits(texts: list[str], seq_len: int, cache_name: str = "full_bits"):
    """Build/cache packed uint8 on disk; return float32 tensor + per-file window counts.

    Files are joined with (seq_len - 1) \\0 padding to prevent sliding windows
    from crossing file boundaries. Trailing padding covers last file's tail windows.

    Disk: uint8 packed (~0.13 GB for 50M chars × 21 bits).
    RAM: float32 (~4.2 GB) — required for torch.as_strided.
    """
    os.makedirs("data/cache", exist_ok=True)
    cache_path = f"data/cache/{cache_name}_s{seq_len}.u8"

    # Build padded text: file1 + \0*(seq_len-1) + file2 + ... + \0*(seq_len-1)
    null_pad = '\0' * (seq_len - 1)
    padded_text = null_pad.join(texts) + null_pad
    codes = np.array([ord(ch) for ch in padded_text], dtype=np.uint32)
    total_bits = len(codes) * UNICODE_BITS

    # ── Load from disk cache if exists ──
    if os.path.exists(cache_path):
        packed = torch.load(cache_path, map_location='cpu', weights_only=True)
        expected_bytes = (total_bits + 7) // 8
        actual_bytes = len(np.asarray(packed)) * (packed.itemsize if hasattr(packed, 'itemsize') else 1)
        if abs(actual_bytes - expected_bytes) <= 1:
            print(f"  Loaded uint8 cache: {packed.numel() / 1e6:.1f} MB")
            full_bits = unpack_uint8_to_float(packed, total_bits)
            return full_bits, _compute_file_offsets(texts, seq_len)
        # Cache mismatch (different corpus size) — rebuild
        print("  Cache mismatch — rebuilding...")

    # ── Fresh build ──
    packed = torch.from_numpy(pack_bits_uint8(codes))
    torch.save(packed, cache_path)
    n_gb = packed.numel() / 1e9
    print(f"  Built uint8 cache: {n_gb:.2f} GB (unpacked: {total_bits * 4 / 1e9:.2f} GB float32)")
    full_bits = unpack_uint8_to_float(packed, total_bits)
    return full_bits, _compute_file_offsets(texts, seq_len)


def _compute_file_offsets(texts: list[str], seq_len: int) -> np.ndarray:
    """Return (N, 2) array of [start_window, end_window) per file.

    Each file of length L contributes L windows (L positions with ≥1 real char).
    The (seq_len - 1) zeros between files prevent boundary crossing;
    all-zero windows in the gap are excluded.
    """
    offsets = []
    cursor = 0  # current character position in padded text
    for text in texts:
        L = len(text)
        offsets.append([cursor, cursor + L])
        cursor += L + (seq_len - 1)  # file chars + padding
    return np.array(offsets, dtype=np.int64)


class SlidingWindowDataset(Dataset):
    """Sliding window over character bits — stride=1 character.

    Each window captures seq_len consecutive characters. With stride=1,
    each character appears in seq_len different windows.

    When file_offsets is provided, only windows with ≥1 real (non-padding)
    character are included — null-padding gaps are skipped.
    """

    def __init__(self, full_bits, seq_len, indices=None, file_offsets=None):
        self.seq_len = seq_len
        self.window_bits = seq_len * UNICODE_BITS
        n_total = full_bits.numel() // UNICODE_BITS - seq_len + 1
        # as_strided view: (n_total, window_bits) with stride (UNICODE_BITS, 1)
        self._windows = torch.as_strided(
            full_bits,
            size=(n_total, self.window_bits),
            stride=(UNICODE_BITS, 1),
        )

        # When file_offsets is set: only include windows with ≥1 real char
        if file_offsets is not None:
            valid_start_positions = _build_valid_indices(file_offsets, seq_len)
            if indices is not None:
                # Subset: map global indices to valid positions
                self._indices = indices
                self._all_valid = valid_start_positions
            else:
                self._indices = valid_start_positions
        else:
            self._indices = (
                indices if indices is not None
                else torch.arange(n_total)
            )

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        pos = self._indices[idx]
        # empty_like + copy_ is cheaper than clone() for strided views
        w = torch.empty(self.window_bits, dtype=torch.float32, device='cpu')
        w.copy_(self._windows[pos])
        return w, w


def _build_valid_indices(file_offsets: np.ndarray, seq_len: int) -> torch.Tensor:
    """Build tensor of valid window start positions from file offsets.

    Each file of length L contributes L windows: positions [start, start+L-1].
    Trailing (seq_len-1) pad after each file ensures no boundary crossing.
    """
    segments = []
    for start, end in file_offsets:
        L = end - start
        if L > 0:
            segments.append(torch.arange(start, start + L, dtype=torch.int64))
    if not segments:
        return torch.zeros(0, dtype=torch.int64)
    return torch.cat(segments)


class NoisyDataset(Dataset):
    """Wraps a SlidingWindowDataset — adds uint21-level Gaussian noise to inputs.

    For each character (21-bit group) in the input window:
      1. Convert 21 bits → uint21 value
      2. With probability noise_prob, add N(0, noise_std²)
      3. Round to nearest integer, clamp to [0, 2²¹−1]
      4. Convert back to 21 bits

    Per-batch sampling: noise_prob and noise_std are sampled uniformly
    from [min, max] ranges on each __getitem__ call.

    Target (y) stays clean — model must denoise.
    """

    def __init__(self, base_dataset, noise_prob_min=0.0, noise_prob_max=0.0,
                 noise_std_min=3.0, noise_std_max=3.0, seed=None):
        self.base = base_dataset
        self.noise_prob_min = noise_prob_min
        self.noise_prob_max = noise_prob_max
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max
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

        noise_prob = float(self._rng.uniform(self.noise_prob_min, self.noise_prob_max))
        noise_std = float(self._rng.uniform(self.noise_std_min, self.noise_std_max))

        if noise_prob <= 0.0:
            return bits

        r = self._rng.random(seq_len)
        noise_mask = torch.from_numpy(r < noise_prob).to(torch.long)
        noise_vals = torch.from_numpy(
            self._rng.normal(0, noise_std, seq_len)).float()

        noisy = uints.float() + noise_mask.float() * noise_vals
        noisy = torch.round(noisy).long().clamp(0, self._max_val)

        bit_positions = torch.arange(UNICODE_BITS)
        noisy_bits = ((noisy.unsqueeze(-1) >> bit_positions) & 1).float().flatten()
        return noisy_bits





def prepare_data(texts: list[str], seq_len: int, train_ratio: float = 0.99):
    """Build sliding-window dataset and return (train_ds, val_ds).

    texts: list of strings — one per input file.
    Files are padded with (seq_len-1) \0 chars to prevent boundary-crossing
    windows. Only windows with ≥1 real character are included.
    Returns SlidingWindowDataset objects with non-overlapping indices.
    """
    full_bits, file_offsets = _build_full_bits(texts, seq_len)
    dataset = SlidingWindowDataset(full_bits, seq_len, file_offsets=file_offsets)
    n = len(dataset)
    indices = torch.randperm(n)
    train_size = int(n * train_ratio)
    train_ds = SlidingWindowDataset(full_bits, seq_len, file_offsets=file_offsets,
                                    indices=indices[:train_size])
    val_ds = SlidingWindowDataset(full_bits, seq_len, file_offsets=file_offsets,
                                  indices=indices[train_size:])
    return train_ds, val_ds


def export_latent_vectors(model, texts, config, device, output_path="data/latent/latent_vectors.pt"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()
    full_bits, file_offsets = _build_full_bits(texts, config.seq_len)
    dataset = SlidingWindowDataset(full_bits, config.seq_len, file_offsets=file_offsets)
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
