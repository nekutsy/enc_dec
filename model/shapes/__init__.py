"""Shape Protocol — pluggable architecture shapes.

Each shape is a plugin: make_sizes, solve_width, solve_depth.
New shapes register with @register and are immediately available everywhere.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from experiment.config import SweepConfig


# ── Helpers ──────────────────────────────────────────────────

def count_params(layer_sizes: list[int]) -> int:
    """Count Linear + BatchNorm1d parameters for the given layer sizes."""
    n = 0
    for i in range(len(layer_sizes) - 1):
        n += layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
        n += 2 * layer_sizes[i + 1]
    return n


def _binary_search_b(make_sizes_fn, target_params: int, input_dim: int,
                     lo: float = 0.1, hi: float = 20.0, iters: int = 30
                     ) -> tuple[float, int, int]:
    """Generic binary search for width parameter b.

    Searches ∈ [lo, hi] → returns (b, hidden_dim, n_params).
    """
    def _p(x):
        h = max(1, int(round(input_dim * x)))
        return count_params(make_sizes_fn(h))

    for _ in range(iters):
        mid = (lo + hi) / 2
        if _p(mid) < target_params:
            lo = mid
        else:
            hi = mid

    p_lo, p_hi = _p(lo), _p(hi)
    b = (round(lo, 6) if abs(p_lo - target_params) <= abs(p_hi - target_params)
         else round(hi, 6))
    h = max(1, int(round(input_dim * b)))
    return b, h, _p(b)


def _binary_search_n(make_sizes_fn, target_params: int, hidden_dim: int,
                     lo: int = 1, hi: int = 20, iters: int = 30
                     ) -> tuple[int, int, int]:
    """Generic binary search for depth n. Returns (n, hidden_dim, n_params)."""
    def _p(n_val):
        return count_params(make_sizes_fn(int(n_val)))

    for _ in range(iters):
        mid = int((lo + hi) // 2)
        if _p(mid) < target_params:
            lo = mid + 1
        else:
            hi = mid

    p_lo, p_hi = _p(lo), _p(hi)
    n = lo if abs(p_lo - target_params) <= abs(p_hi - target_params) else hi
    n = max(1, min(n, hi))
    return n, hidden_dim, _p(n)


def _format_result(sizes: list[int], b_val: float | None, enc_n: int,
                   dec_n: int, hidden_dim: int, input_dim: int) -> dict:
    """Standard result dict for architecture resolution."""
    return {
        'sizes': sizes,
        'b': round(hidden_dim / input_dim, 6) if b_val is None else b_val,
        'enc_n': enc_n, 'dec_n': dec_n,
        'n': enc_n if enc_n == dec_n else None,
        'hidden_dim': hidden_dim,
        'n_params': count_params(sizes),
    }


def _get_ns(mc, fixed: dict) -> tuple[int, int]:
    """Extract (enc_n, dec_n) from model config + sweep fixed dict."""
    enc_n = mc.enc_n if mc.enc_n is not None else fixed.get('enc_n', fixed.get('n', None))
    if enc_n is None:
        raise ValueError("enc_n or n must be specified in ModelConfig or sweep.fixed")
    dec_n = mc.dec_n if mc.dec_n is not None else fixed.get('dec_n', enc_n)
    return int(enc_n), int(dec_n)


# ── Vary categories ─────────────────────────────────────────

MODEL_LEVEL_VARY = {'normalization', 'activation', 'dropout',
                    'norm_bottleneck', 'norm_last', 'bottleneck',
                    'trapezoid_alpha', 'residual', 'residual_norm'}
TRAIN_LEVEL_VARY = {'lr', 'scheduler', 'grad_clip', 'optimizer',
                    'weight_decay', 'batch_size', 'num_workers',
                    'noise_prob_min', 'noise_prob_max', 'noise_std_min', 'noise_std_max',
                    'noise_strategy', 'noise_stride', 'vae_beta'}
N_VARY = {'n', 'enc_n', 'dec_n'}


# ── Shape base class ────────────────────────────────────────

class Shape(ABC):
    """Pluggable architecture shape.

    Subclass, implement make_sizes + solve_width (+ optionally solve_depth),
    register with @register().
    """
    name: str

    @abstractmethod
    def make_sizes(self, input_dim: int, hidden_dim: int, bottleneck: int,
                   enc_n: int, dec_n: int, **kwargs) -> list[int]:
        """Build layer sizes list from parameters."""
        ...

    @abstractmethod
    def solve_width(self, target_params: int, input_dim: int, bottleneck: int,
                    enc_n: int, dec_n: int, **kwargs) -> tuple[float, int, int]:
        """Binary search for width. Returns (b, hidden_dim, n_params)."""
        ...

    def solve_depth(self, b_val: float, target_params: int, input_dim: int,
                    bottleneck: int) -> tuple[int, int, int] | None:
        """Binary search for depth. Returns (n, hidden_dim, n_params) or None."""
        return None

    def resolve(self, input_dim: int, bottleneck: int, enc_n: int, dec_n: int,
                budget: int | None, solve_mode: str | None, b_val: float | None,
                **kwargs) -> dict:
        """Full resolution: config params → architecture result dict.

        Single entry point — shape encapsulates all of build + solve + format.
        """
        if solve_mode == 'b':
            if budget is None:
                raise ValueError(f"Shape '{self.name}': solve=b requires budget")
            b_val, hidden_dim, _ = self.solve_width(
                budget, input_dim, bottleneck, enc_n, dec_n, **kwargs)
        elif solve_mode == 'n':
            if b_val is None:
                raise ValueError(f"Shape '{self.name}': solve=n requires fixed b")
            result = self.solve_depth(b_val, budget or 0, input_dim, bottleneck)
            if result is None:
                raise NotImplementedError(f"solve=n not supported for '{self.name}' shape")
            n, hidden_dim, _ = result
            enc_n = dec_n = n
        elif b_val is not None:
            hidden_dim = max(1, int(round(input_dim * b_val)))
        else:
            raise ValueError(f"Shape '{self.name}': need b, or solve+fixed")

        sizes = self.make_sizes(input_dim, hidden_dim, bottleneck,
                                enc_n, dec_n, **kwargs)
        return _format_result(sizes, b_val, enc_n, dec_n, hidden_dim, input_dim)


# ── Registry ─────────────────────────────────────────────────

_registry: dict[str, Shape] = {}

def register(shape_cls: type) -> type:
    """Decorator: register a Shape subclass as a shape plugin."""
    instance = shape_cls()
    _registry[instance.name] = instance
    return shape_cls

def get_shape(name: str) -> Shape:
    """Look up a shape by name."""
    if name not in _registry:
        raise ValueError(f"Unknown shape: '{name}'. Available: {list(_registry.keys())}")
    return _registry[name]

def list_shapes() -> list[str]:
    return list(_registry.keys())
