"""Architecture math — thin facade over pluggable shape registry.

For backward compatibility, all old function names are re-exported.
New code should use `model/shapes/` directly or `get_shape(name).resolve(...)`.

To add a new shape: create `model/shapes/new_shape.py` with a `@register` class.
That's it — no changes needed anywhere else.
"""

from configs import UNICODE_BITS
from experiment.config import SweepConfig

# Import shape plugins to trigger @register
from model.shapes.rectangular import RectangularShape  # noqa: F401
from model.shapes.pyramid import PyramidShape  # noqa: F401
from model.shapes.interleaved import InterleavedShape  # noqa: F401
from model.shapes.trapezoid import TrapezoidShape  # noqa: F401

from model.shapes import (
    Shape,
    Shape as _Shape,  # noqa: F811
    register,
    get_shape,
    list_shapes,
    count_params,
    MODEL_LEVEL_VARY,
    TRAIN_LEVEL_VARY,
    N_VARY,
    _get_ns,
    _format_result,
)

# ── Backward-compatible make_* re-exports ──────────────────

def make_rectangular(input_dim: int, hidden_dim: int, bottleneck: int,
                     enc_n: int, dec_n: int | None = None) -> list[int]:
    """[input] → [hidden]×enc_n → [bottleneck] → [hidden]×dec_n → [input]"""
    if dec_n is None:
        dec_n = enc_n
    return get_shape('rectangular').make_sizes(
        input_dim, hidden_dim, bottleneck, enc_n, dec_n)


def make_pyramid(input_dim: int, bottleneck: int,
                 enc_n: int, dec_n: int | None = None,
                 d: float | None = None) -> list[int]:
    """Build pyramid: D→h1→…→h_en→B→h_dn→…→h1→D"""
    if dec_n is None:
        dec_n = enc_n
    return get_shape('pyramid').make_sizes(
        input_dim, 0, bottleneck, enc_n, dec_n, d=d)


def make_trapezoid(input_dim: int, hidden_dim: int, bottleneck: int,
                   enc_n: int, dec_n: int | None = None,
                   alpha: float = 0.1) -> list[int]:
    """Trapezoid: widths linearly interpolate from hidden_dim×(1+α) to hidden_dim×(1−α)."""
    if dec_n is None:
        dec_n = enc_n
    return get_shape('trapezoid').make_sizes(
        input_dim, hidden_dim, bottleneck, enc_n, dec_n, alpha=alpha)


def make_interleaved(input_dim: int, hidden_dim: int, bottleneck: int,
                     enc_n: int, dec_n: int | None = None) -> list[int]:
    """Wide (hidden_dim) layers alternate with pyramid layers."""
    if dec_n is None:
        dec_n = enc_n
    return get_shape('interleaved').make_sizes(
        input_dim, hidden_dim, bottleneck, enc_n, dec_n)


# ── Backward-compatible solver re-exports ──────────────────

def solve_b_for_rect(target_params: int, input_dim: int, bottleneck: int,
                     enc_n: int, dec_n: int) -> tuple[float, int, int]:
    return get_shape('rectangular').solve_width(
        target_params, input_dim, bottleneck, enc_n, dec_n)


def solve_b_for_trapezoid(target_params: int, input_dim: int, bottleneck: int,
                          enc_n: int, dec_n: int, alpha: float
                          ) -> tuple[float, int, int]:
    return get_shape('trapezoid').solve_width(
        target_params, input_dim, bottleneck, enc_n, dec_n, alpha=alpha)


def solve_b_for_interleaved(target_params: int, input_dim: int, bottleneck: int,
                            enc_n: int, dec_n: int
                            ) -> tuple[float, int, int]:
    return get_shape('interleaved').solve_width(
        target_params, input_dim, bottleneck, enc_n, dec_n)


def solve_d_for_n(target_params: int, D: int, B: int, enc_n: int, dec_n: int,
                  max_d: float | None = None
                  ) -> tuple[float, int, int]:
    """Binary search d ∈ [0.1, max_d] for pyramid architecture."""
    return get_shape('pyramid').solve_width(
        target_params, D, B, enc_n, dec_n)


def solve_b_for_n(n_hidden: int, target_params: int, input_dim: int,
                  bottleneck: int) -> tuple[float, int, int]:
    """Legacy wrapper: symmetric n."""
    return solve_b_for_rect(target_params, input_dim, bottleneck, n_hidden, n_hidden)


def solve_b_for_n_trapezoid(n_hidden: int, target_params: int, input_dim: int,
                              bottleneck: int, alpha: float
                              ) -> tuple[float, int, int]:
    return solve_b_for_trapezoid(target_params, input_dim, bottleneck,
                                 n_hidden, n_hidden, alpha)


def solve_b_for_n_interleaved(n: int, target_params: int, input_dim: int,
                               bottleneck: int
                               ) -> tuple[float, int, int]:
    return solve_b_for_interleaved(target_params, input_dim, bottleneck, n, n)


def solve_n_for_b(b_val: float, target_params: int, input_dim: int,
                  bottleneck: int, max_n: int = 20
                  ) -> tuple[int, int, int]:
    shape = get_shape('rectangular')
    return shape.solve_depth(b_val, target_params, input_dim, bottleneck) or (1, 1, 0)


def solve_n_for_b_interleaved(b_val: float, target_params: int, input_dim: int,
                               bottleneck: int, max_n: int = 20
                               ) -> tuple[int, int, int]:
    shape = get_shape('interleaved')
    return shape.solve_depth(b_val, target_params, input_dim, bottleneck) or (1, 1, 0)


def solve_d_for_n_sym(n: int, target_params: int, D: int, B: int,
                      max_d: float | None = None
                      ) -> tuple[float, int, int]:
    """Legacy wrapper: symmetric pyramid."""
    return solve_d_for_n(target_params, D, B, n, n, max_d=max_d)


# ── resolve_architecture — unified entry point ──────────────

def resolve_architecture(vary_value, vary_name: str,
                         sweep_config: SweepConfig) -> dict:
    """Resolve full architecture for a sweep candidate.

    Dispatches to shape.resolve() — shape plugins handle all logic.
    """
    mc = sweep_config.model
    sc = sweep_config.sweep

    # Handle legacy vary names that were renamed to range fields
    _LEGACY_VARY = {'noise_prob', 'noise_std'}

    # Apply vary to the right config object
    if vary_name in MODEL_LEVEL_VARY:
        setattr(mc, vary_name, vary_value)
    elif vary_name in TRAIN_LEVEL_VARY:
        setattr(sweep_config.training, vary_name, vary_value)
    elif vary_name in _LEGACY_VARY:
        if vary_name == 'noise_prob':
            sweep_config.training.noise_prob_min = vary_value
            sweep_config.training.noise_prob_max = vary_value
        elif vary_name == 'noise_std':
            sweep_config.training.noise_std_min = vary_value
            sweep_config.training.noise_std_max = vary_value

    # If vary was model/train-level, fall back to n or b from fixed
    if vary_name in MODEL_LEVEL_VARY | TRAIN_LEVEL_VARY | _LEGACY_VARY:
        fixed = dict(sc.fixed)
        for n_key in ['n', 'enc_n', 'dec_n']:
            if n_key in fixed:
                vary_name, vary_value = n_key, fixed[n_key]
                break
        else:
            if 'b' in fixed:
                vary_name, vary_value = 'b', fixed['b']
            else:
                raise ValueError(
                    f'vary={vary_name} needs either n=/enc_n=/dec_n= or b= in fixed')

    input_dim = mc.seq_len * UNICODE_BITS
    bottleneck = mc.bottleneck if mc.bottleneck is not None else mc.seq_len
    shape = getattr(mc, 'shape', 'rectangular')
    fixed = dict(sc.fixed)

    # Place vary_value into fixed — handle n-like params
    if vary_name in N_VARY:
        fixed[vary_name] = vary_value
        if vary_name == 'n':
            fixed.setdefault('enc_n', vary_value)
            fixed.setdefault('dec_n', vary_value)
    elif vary_name == 'b':
        fixed['b'] = vary_value

    # total_n: auto-set complementary enc_n/dec_n
    if sc.total_n is not None:
        if 'enc_n' in fixed and 'dec_n' not in fixed:
            fixed['dec_n'] = sc.total_n - int(fixed['enc_n'])
        elif 'dec_n' in fixed and 'enc_n' not in fixed:
            fixed['enc_n'] = sc.total_n - int(fixed['dec_n'])

    enc_n, dec_n = _get_ns(mc, fixed)
    b_val = fixed.get('b', None)

    # Use kwargs specific to each shape
    kwargs = {}
    if shape == 'trapezoid':
        kwargs['alpha'] = getattr(mc, 'trapezoid_alpha', 0.1)

    return get_shape(shape).resolve(
        input_dim, bottleneck, enc_n, dec_n,
        budget=sc.budget, solve_mode=sc.solve, b_val=b_val,
        **kwargs,
    )
