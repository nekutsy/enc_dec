"""Unicode-21 bit encoding — pure domain logic.

Notation
  codepoint = ord(char)            →  uint32 in 0..0x10FFFF
  codes     = ndarray[int32]       →  (N,) array of codepoints
  bits      = ndarray[float32]     →  (N, 21) or (N*21,) bit vectors
  packed    = ndarray[uint8]       →  compressed uint8 for disk storage
  chars     = str                  →  decoded text string

Dependencies: numpy only. No PyTorch, no filesystem.
"""

import numpy as np

UNICODE_BITS = 21                     # bits per codepoint
CODEPOINT_MASK = (1 << UNICODE_BITS) - 1


# ── Core: codepoints ↔ bits ─────────────────────────────────

def chars_to_bits(codes: np.ndarray) -> np.ndarray:
    """(N,) uint32 codepoints → (N, UNICODE_BITS) float32 bits, MSB first.

    >>> chars_to_bits(np.array([65], dtype=np.uint32)).astype(int)
    array([[0, 0, ..., 0, 0, 1, 0, 0, 0, 0, 0, 1]])  # 'A' = 0x41

    Always returns float32 for direct use in ML pipelines.
    Vectorised: no Python loops over characters.
    """
    N = len(codes)
    bits = np.zeros((N, UNICODE_BITS), dtype=np.float32)
    for i in range(UNICODE_BITS):
        bits[:, i] = (codes >> (UNICODE_BITS - 1 - i)) & 1
    return bits


def bits_to_chars(bits: np.ndarray) -> str:
    """(N, UNICODE_BITS) float32 bits → decoded string.

    Codepoints > 0.5 threshold → 1. Non-zero codepoints are decoded.
    Zero codepoints are stripped (padding markers).

    >>> bits_to_chars(chars_to_bits(np.array([65, 66], dtype=np.uint32)))
    'AB'
    """
    if bits.ndim == 1:
        bits = bits.reshape(-1, UNICODE_BITS)
    powers = 2 ** np.arange(UNICODE_BITS - 1, -1, -1)
    codepoints = ((bits > 0.5) @ powers).astype(int)
    valid = codepoints[codepoints > 0]
    return ''.join(chr(c) for c in valid)


# ── Pack/Unpack for disk storage ─────────────────────────────

def pack_bits_uint8(codes: np.ndarray) -> np.ndarray:
    """(N,) uint32 codepoints → packed uint8 for disk.

    Uses lil-endian packbits. ceil(N * 21 / 8) bytes.
    """
    total_bits = len(codes) * UNICODE_BITS
    bits = np.zeros(total_bits, dtype=np.uint8)
    for i in range(UNICODE_BITS):
        bits[i::UNICODE_BITS] = (codes >> (UNICODE_BITS - 1 - i)) & 1
    n_padded = ((total_bits + 7) // 8) * 8
    padded = np.zeros(n_padded, dtype=np.uint8)
    padded[:total_bits] = bits
    return np.packbits(padded.reshape(-1, 8), axis=1, bitorder='little').ravel()


def unpack_uint8_to_float(packed, total_bits: int):
    """Packed uint8 → float32 tensor (0.0/1.0). Accepts numpy or torch.

    Returns torch.Tensor for direct use in Dataset.
    """
    import torch
    if isinstance(packed, torch.Tensor):
        packed = packed.cpu().numpy()
    packed = np.asarray(packed, dtype=np.uint8)
    if len(packed) == 0:
        return torch.zeros(total_bits, dtype=torch.float32)
    unpacked = np.unpackbits(packed, bitorder='little')
    return torch.from_numpy(unpacked[:total_bits].astype(np.float32))


# ── High-level helpers ───────────────────────────────────────

def seq_to_vec(seq: str, max_bits: int) -> tuple[list[float], int]:
    """Encode a string into a fixed-length float32 bit vector.

    Returns (bits_list, chars_used). Truncates to max_bits capacity.
    """
    max_chars = max_bits // UNICODE_BITS
    codes = np.array([ord(ch) for ch in seq[:max_chars]], dtype=np.uint32)
    used = len(codes)
    bits = np.zeros(max_bits, dtype=np.float32)
    if used > 0:
        bits[:used * UNICODE_BITS] = chars_to_bits(codes).ravel()
    return bits.tolist(), used


def vec_to_seq(vec) -> str:
    """Fixed-size float32 bit vector → decoded string.

    Accepts both (N*UNICODE_BITS,) flat and (N, UNICODE_BITS) 2D arrays.
    Padded zero-codepoints are stripped.
    """
    arr = np.asarray(vec, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] == UNICODE_BITS:
        return bits_to_chars(arr)
    n_chars = len(arr) // UNICODE_BITS
    return bits_to_chars(arr[:n_chars * UNICODE_BITS])


def split_into_chunks(text: str, max_bits: int) -> list[tuple[str, list[float]]]:
    """Split text into chunks of ≤max_bits bits each. Returns (text, bits_list)."""
    chunks: list[tuple[str, list[float]]] = []
    i = 0
    max_chars = max_bits // UNICODE_BITS
    while i < len(text):
        chunk_text = text[i:i + max_chars]
        bits, used = seq_to_vec(chunk_text, max_bits)
        chunks.append((chunk_text, bits))
        i += used
    return chunks
