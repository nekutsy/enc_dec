"""Symmetric autoencoder with configurable activation and normalization."""

import torch.nn as nn


_ACTIVATIONS = {
    'silu': nn.SiLU,
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'leaky_relu': lambda: nn.LeakyReLU(0.01),
}

_NORMS = {
    'batchnorm': nn.BatchNorm1d,
    'layernorm': nn.LayerNorm,
    'none': lambda dim: nn.Identity(),
}


def _build_seq(layer_sizes, activation, norm, start_idx, end_idx):
    """Build sequential block: Linear → Norm → Activation (except last layer)."""
    layers = []
    for i in range(start_idx, end_idx):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < end_idx - 1:
            if norm != 'none':
                layers.append(_NORMS[norm](layer_sizes[i + 1]))
            if activation != 'none':
                layers.append(_ACTIVATIONS[activation]())
    return layers


class Autoencoder(nn.Module):
    """Symmetric autoencoder — encoder compresses to bottleneck, decoder reconstructs.

    Layer sizes are split at the midpoint (bottleneck). First half forms the
    encoder, second half the decoder. Supports configurable activation and
    normalization.
    """

    def __init__(self, layer_sizes: list[int], name: str = "autoencoder",
                 activation: str = "silu", normalization: str = "batchnorm"):
        super().__init__()
        self.name = name
        self.layer_sizes = layer_sizes
        self._bottleneck = layer_sizes[len(layer_sizes) // 2]
        self.activation = activation
        self.normalization = normalization
        b_idx = len(layer_sizes) // 2

        enc_layers = _build_seq(layer_sizes, activation, normalization, 0, b_idx)
        dec_layers = _build_seq(layer_sizes, activation, normalization, b_idx, len(layer_sizes) - 1)

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
        return f"name={self.name}, act={self.activation}, norm={self.normalization}, enc=({enc}), bottleneck={self._bottleneck}, dec=({dec})"

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
