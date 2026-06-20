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


def _build_seq(layer_sizes, activation, norm, start_idx, end_idx,
               norm_before_last=True, activation_before_last=True, dropout=0.0):
    """Build sequential block: Linear → Norm → Activation → Dropout.

    Last layer gets norm/activation/dropout only if the corresponding flag is True.
    """
    layers = []
    for i in range(start_idx, end_idx):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        is_final_layer = (i == end_idx - 1)
        apply_norm = (not is_final_layer or norm_before_last) and norm != 'none'
        apply_act = (not is_final_layer or activation_before_last) and activation != 'none'
        if apply_norm:
            layers.append(_NORMS[norm](layer_sizes[i + 1]))
        if apply_act:
            layers.append(_ACTIVATIONS[activation]())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    return layers


class Autoencoder(nn.Module):
    """Symmetric autoencoder — encoder compresses to bottleneck, decoder reconstructs.

    Layer sizes are split at the midpoint (bottleneck). First half forms the
    encoder, second half the decoder. Supports configurable activation and
    normalization.
    """

    def __init__(self, layer_sizes: list[int], name: str = "autoencoder",
                 activation: str = "silu", normalization: str = "batchnorm",
                 init_gain: float = 0.5,
                 norm_bottleneck: bool = True, norm_last: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.name = name
        self.layer_sizes = layer_sizes
        self._bottleneck = layer_sizes[len(layer_sizes) // 2]
        self.activation = activation
        self.normalization = normalization
        self.init_gain = init_gain
        self.dropout = dropout
        b_idx = len(layer_sizes) // 2

        enc_layers = _build_seq(
            layer_sizes, activation, normalization, 0, b_idx,
            norm_before_last=norm_bottleneck, activation_before_last=False,
            dropout=dropout,
        )
        dec_layers = _build_seq(
            layer_sizes, activation, normalization, b_idx, len(layer_sizes) - 1,
            norm_before_last=norm_last, activation_before_last=False,
            dropout=dropout,
        )

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=init_gain)
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
