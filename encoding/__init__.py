"""Unicode-21 bit encoding — pure domain logic, no PyTorch."""

from encoding.unicode21 import (
    UNICODE_BITS,
    chars_to_bits,
    bits_to_chars,
    seq_to_vec,
    vec_to_seq,
    pack_bits_uint8,
    unpack_uint8_to_float,
)
