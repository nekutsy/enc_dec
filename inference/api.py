"""Inference API — encode, decode, reconstruct — clean domain layer.

Uses ModelLike protocol — decoupled from concrete Autoencoder class.
"""

import torch
import torch.nn as nn
import numpy as np

from encoding.unicode21 import (
    UNICODE_BITS, chars_to_bits, bits_to_chars as _bits_to_chars,
)
from core.types import ModelLike


class ModelInference:
    """High-level inference API for autoencoder models.

    Usage:
        inf = ModelInference(model, seq_len=128, device='cuda')
        latent = inf.encode("Hello world")         # → np.ndarray
        text = inf.decode(latent)                   # → str
        rec, char_err, bit_err = inf.reconstruct("Hello")  # → (str, int, float)
    """

    def __init__(self, model: ModelLike, seq_len: int,
                 device: str | torch.device = 'cpu'):
        self.model = model
        self.seq_len = seq_len
        self.device = torch.device(device)
        self._input_dim = seq_len * UNICODE_BITS

    # ── Core operations ─────────────────────────────────

    def encode(self, text: str) -> np.ndarray:
        """Encode text → latent vector (numpy). Pads to seq_len with \\0."""
        inp, _ = self._text_to_tensor(text)
        with torch.inference_mode():
            latent = self.model.encode(inp).squeeze(0).cpu().numpy()
        return latent

    def decode(self, latent: np.ndarray) -> str:
        """Decode latent vector → reconstructed text."""
        z = torch.from_numpy(latent).float().unsqueeze(0).to(self.device)
        with torch.inference_mode():
            out_logits = self.model.decode(z).squeeze(0).cpu().numpy()
        out_bits = 1.0 / (1.0 + np.exp(-out_logits))
        return _bits_to_chars(out_bits)

    def reconstruct(self, text: str) -> tuple[str, int, float]:
        """Reconstruct text through encode→decode.

        Returns (reconstructed_text, char_errors, bit_errors).
        Pads text to seq_len with \\0.
        """
        inp, padded = self._text_to_tensor(text)
        with torch.inference_mode():
            out_logits = self.model(inp).squeeze(0).cpu().numpy()
        out_bits = 1.0 / (1.0 + np.exp(-out_logits))
        rec = _bits_to_chars(out_bits)
        errors = sum(1 for a, b in zip(padded, rec) if a != b)
        codes = np.array([ord(ch) if ch != '\0' else 0 for ch in padded],
                         dtype=np.uint32)
        bit_err = float(np.abs(chars_to_bits(codes).ravel() - out_bits).sum())
        return rec, errors, bit_err

    # ── Batch reconstruction ─────────────────────────────

    def reconstruct_long(self, text: str) -> tuple[str, int, float]:
        """Reconstruct text longer than seq_len — splits into windows.

        Returns (reconstructed_text, char_errors, bit_errors).
        """
        sl = self.seq_len
        total_err_c, total_err_b = 0, 0.0
        parts: list[str] = []
        for start in range(0, len(text), sl):
            chunk = text[start:start + sl]
            rec, errors, bit_err = self.reconstruct(chunk)
            parts.append(rec.rstrip('\0'))
            total_err_c += errors
            total_err_b += bit_err
        return ''.join(parts), total_err_c, total_err_b

    # ── Batch validation ────────────────────────────────

    def validate(self, val_loader, device: str | torch.device | None = None
                 ) -> float:
        """Run validation pass and return average BCE loss.

        val_loader must yield (x_batch, y_batch) tuples.
        """
        dev = torch.device(device) if device else self.device
        criterion = nn.BCEWithLogitsLoss()
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        with torch.inference_mode():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(dev)
                y_batch = y_batch.to(dev)
                out = self.model(x_batch)
                loss = criterion(out, y_batch)
                n = x_batch.size(0)
                total_loss += loss.item() * n
                total_samples += n
        return total_loss / total_samples if total_samples > 0 else 0.0

    # ── Internal ────────────────────────────────────────

    def _text_to_tensor(self, text: str) -> tuple[torch.Tensor, str]:
        """Convert text → padded (1, seq_len*21) float tensor + padded string."""
        sl = self.seq_len
        padded = text + '\0' * (sl - len(text))
        codes = np.array(
            [ord(ch) if ch != '\0' else 0 for ch in padded],
            dtype=np.uint32,
        )
        bits = chars_to_bits(codes).ravel()
        tensor = torch.from_numpy(bits).float().unsqueeze(0).to(self.device)
        return tensor, padded
