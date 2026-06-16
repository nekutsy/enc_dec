"""Symmetric autoencoder with configurable layer sizes."""

import torch.nn as nn


class Autoencoder(nn.Module):
    """Symmetric autoencoder — encoder compresses to bottleneck, decoder reconstructs.

    Layer sizes are split at the midpoint (bottleneck).  First half forms the
    encoder, second half the decoder.  Linear → BatchNorm1d → SiLU after each
    layer except the last.  Weights are orthogonally initialised with gain=0.5.
    """

    def __init__(self, layer_sizes: list[int], name: str = "autoencoder"):
        super().__init__()
        self.name = name
        self.layer_sizes = layer_sizes
        self._bottleneck = layer_sizes[len(layer_sizes) // 2]
        b_idx = len(layer_sizes) // 2

        enc_layers = []
        for i in range(b_idx):
            enc_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < b_idx - 1:
                enc_layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
                enc_layers.append(nn.SiLU())

        dec_layers = []
        for i in range(b_idx, len(layer_sizes) - 1):
            dec_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                dec_layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
                dec_layers.append(nn.SiLU())

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.constant_(m.bias, 0.0)

    @property
    def bottleneck(self) -> int:
        return self._bottleneck

    def extra_repr(self) -> str:
        enc = "→".join(str(s) for s in self.layer_sizes[:len(self.layer_sizes) // 2 + 1])
        dec = "→".join(str(s) for s in self.layer_sizes[len(self.layer_sizes) // 2:])
        return f"name={self.name}, enc=({enc}), bottleneck={self._bottleneck}, dec=({dec})"

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
