"""Symmetric autoencoder with configurable activation and normalization."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMS Normalization — scale-only, no mean-centering, no bias.

    RMSNorm(x) = x * g / sqrt(mean(x²) + eps)

    Preserves directional structure better than LayerNorm for autoencoders:
    - No mean subtraction → linear bias term is not wasted
    - Learns per-feature gain γ
    - Stable at high dimensions → no ε-float-underflow risk
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x * self.weight / rms


_ACTIVATIONS = {
    'silu': nn.SiLU,
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'leaky_relu': lambda: nn.LeakyReLU(0.01),
}

_NORMS = {
    'batchnorm': nn.BatchNorm1d,
    'layernorm': nn.LayerNorm,
    'rmsnorm': RMSNorm,
    'none': lambda dim: nn.Identity(),
}


def _build_seq(layer_sizes, activation, norm, start_idx, end_idx,
               norm_before_last=True, activation_before_last=True, dropout=0.0,
               residual=False, residual_norm='post'):
    """Build sequential block: Linear → Norm → Activation → Dropout.

    Last layer gets norm/activation/dropout only if the corresponding flag is True.
    When residual=True and input_dim == output_dim, wraps the block in _Residual.
    residual_norm='post': x + (Linear → Norm → Act)(x)  [current default]
    residual_norm='pre':  x + (Linear → Act)(Norm(x))
    """
    layers = []
    for i in range(start_idx, end_idx):
        in_dim = layer_sizes[i]
        out_dim = layer_sizes[i + 1]
        is_final_layer = (i == end_idx - 1)
        apply_norm = (not is_final_layer or norm_before_last) and norm != 'none'
        apply_act = (not is_final_layer or activation_before_last) and activation != 'none'
        use_residual = residual and in_dim == out_dim
        pre_norm = use_residual and residual_norm == 'pre'

        block_layers = []

        # Pre-Norm: normalize input before Linear
        if pre_norm and apply_norm:
            block_layers.append(_NORMS[norm](in_dim))

        block_layers.append(nn.Linear(in_dim, out_dim))

        # Post-Norm: normalize output after Linear (default)
        if not pre_norm and apply_norm:
            block_layers.append(_NORMS[norm](out_dim))

        if apply_act:
            block_layers.append(_ACTIVATIONS[activation]())
        if dropout > 0:
            block_layers.append(nn.Dropout(dropout))
        block = nn.Sequential(*block_layers)
        if use_residual:
            block = _Residual(block)
        layers.append(block)
    return layers


class _Residual(nn.Module):
    """Classic residual: f(x) + x. Wraps a sub-module block."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class Autoencoder(nn.Module):
    """Autoencoder — encoder compresses to bottleneck, decoder reconstructs.

    Uses enc_n to locate the bottleneck (default: midpoint for backward compat).
    Supports configurable activation, normalization, and optional residual.

    VAE mode (vae=True): encoder output → fc_mu / fc_logvar → reparameterize → decoder.
    forward() returns (reconstruction, mu, logvar) in VAE mode.
    encode() returns mu deterministically; sample() draws from N(0,I).
    """

    def __init__(self, layer_sizes: list[int], name: str = "autoencoder",
                 activation: str = "silu", normalization: str = "batchnorm",
                 init_gain: float = 1.0,
                 norm_bottleneck: bool = False, norm_last: bool = False,
                 dropout: float = 0.0, residual: bool = False,
                 residual_norm: str = 'post', enc_n: int | None = None,
                 vae: bool = False, vae_beta: float = 1.0):
        super().__init__()
        self.name = name
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.normalization = normalization
        self.init_gain = init_gain
        self.dropout = dropout
        self.residual = residual
        self.residual_norm = residual_norm if residual else 'none'
        self.vae = vae
        self.vae_beta = vae_beta

        # Bottleneck position: enc_n if given, else midpoint (backward compat)
        if enc_n is not None:
            b_idx = 1 + enc_n  # input + enc_n hidden layers → bottleneck
        else:
            b_idx = len(layer_sizes) // 2
        self._bottleneck = layer_sizes[b_idx]

        enc_layers = _build_seq(
            layer_sizes, activation, normalization, 0, b_idx,
            norm_before_last=norm_bottleneck, activation_before_last=False,
            dropout=dropout, residual=residual, residual_norm=residual_norm,
        )
        dec_layers = _build_seq(
            layer_sizes, activation, normalization, b_idx, len(layer_sizes) - 1,
            norm_before_last=norm_last, activation_before_last=False,
            dropout=dropout, residual=residual, residual_norm=residual_norm,
        )

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        # VAE head: parallel μ and log σ² from encoder output
        self.fc_mu: nn.Linear | None = None
        self.fc_logvar: nn.Linear | None = None
        if vae:
            b_dim = self._bottleneck
            self.fc_mu = nn.Linear(b_dim, b_dim)
            self.fc_logvar = nn.Linear(b_dim, b_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=init_gain)
                nn.init.constant_(m.bias, 0.0)

    @property
    def bottleneck(self) -> int:
        return self._bottleneck

    @staticmethod
    def reparameterize(mu: 'torch.Tensor', logvar: 'torch.Tensor') -> 'torch.Tensor':
        """Reparameterization trick: z = μ + ε·exp(0.5·log σ²), ε ~ N(0,I)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_loss(mu: 'torch.Tensor', logvar: 'torch.Tensor') -> 'torch.Tensor':
        """Closed-form KL divergence D_KL(q(z|x) || N(0,I)), summed over latent dims.

        Returns mean per sample (not sum over batch) — scale-invariant.
        """
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def extra_repr(self) -> str:
        b_idx = self.layer_sizes.index(self._bottleneck)
        enc = "→".join(str(s) for s in self.layer_sizes[:b_idx + 1])
        dec = "→".join(str(s) for s in self.layer_sizes[b_idx:])
        extras = []
        if self.vae:
            extras.append(f'vae(β={self.vae_beta})')
        if self.residual:
            tag = '+res-pre' if self.residual_norm == 'pre' else '+res-post'
            extras.append(tag)
        return f"name={self.name}, act={self.activation}, norm={self.normalization}, enc=({enc}), bottleneck={self._bottleneck}, dec=({dec})" + (', ' + ', '.join(extras) if extras else '')

    def forward(self, x):
        h = self.encoder(x)
        if self.vae:
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            z = self.reparameterize(mu, logvar)
            recon = self.decoder(z)
            return recon, mu, logvar
        return self.decoder(h)

    def encode(self, x):
        h = self.encoder(x)
        if self.vae:
            return self.fc_mu(h)  # deterministic: return μ
        return h

    def decode(self, z):
        return self.decoder(z)

    def sample(self, n: int = 1, device=None):
        """Generate samples from VAE prior N(0,I). Returns logits tensor (n, D)."""
        if not self.vae:
            raise RuntimeError('sample() requires vae=True')
        dev = device or next(self.parameters()).device
        z = torch.randn(n, self._bottleneck, device=dev)
        return self.decode(z)
