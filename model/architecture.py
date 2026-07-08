"""Architecture math — layer size builders + budget-constrained solvers.

Pure domain logic. No PyTorch (except count_params which uses simple arithmetic).
Depends on: encoding (for UNICODE_BITS), experiment.config (for SweepConfig).
"""

from configs import UNICODE_BITS
from experiment.config import SweepConfig


def count_params(layer_sizes: list[int]) -> int:
    """Count Linear + BatchNorm1d parameters for the given layer sizes."""
    n = 0
    for i in range(len(layer_sizes) - 1):
        n += layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
        n += 2 * layer_sizes[i + 1]
    return n


# ── Shape builders ──────────────────────────────────────────

def make_rectangular(input_dim: int, hidden_dim: int, bottleneck: int,
                     n_hidden: int) -> list[int]:
    """[input] → [hidden]×n → [bottleneck] → [hidden]×n → [input]"""
    return (
        [input_dim]
        + [hidden_dim] * n_hidden
        + [bottleneck]
        + [hidden_dim] * n_hidden
        + [input_dim]
    )


def make_pyramid(input_dim: int, bottleneck: int, n_hidden: int,
                 d: float) -> list[int]:
    """Build pyramid: D→h1→h2→…→hn→B→hn→…→h2→h1→D"""
    B, n = bottleneck, n_hidden
    enc: list[int] = [input_dim]
    for i in range(1, n + 1):
        enc.append(int(B + d * (n - i + 1) / n))
    enc.append(B)
    dec: list[int] = []
    for i in range(n - 1, -1, -1):
        dec.append(int(B + d * (i + 1) / n))
    dec.append(input_dim)
    return enc + dec


def make_trapezoid(input_dim: int, hidden_dim: int, bottleneck: int,
                   n_hidden: int, alpha: float) -> list[int]:
    """Trapezoid: layer widths linearly interpolate from hd×(1+α) to hd×(1−α).

    n_hidden layers per side, each at width:
      hd × (1 + α − 2α·(i−1)/(n−1))   for i ∈ [1, n_hidden]

    n_hidden=4, α=0.1:
      input → hd×1.1 → hd×1.033 → hd×0.967 → hd×0.9 → bottleneck
           → hd×0.9 → hd×0.967 → hd×1.033 → hd×1.1 → input
    """
    assert n_hidden >= 1
    if n_hidden == 1:
        return [input_dim, hidden_dim, bottleneck, hidden_dim, input_dim]

    def _width(i):
        frac = (i - 1) / (n_hidden - 1)
        return int(round(hidden_dim * (1 + alpha - 2 * alpha * frac)))

    enc: list[int] = [input_dim]
    for i in range(1, n_hidden + 1):
        enc.append(_width(i))
    enc.append(bottleneck)

    dec: list[int] = []
    for i in range(n_hidden, 0, -1):
        dec.append(_width(i))
    dec.append(input_dim)

    return enc + dec


def make_interleaved(input_dim: int, hidden_dim: int, bottleneck: int,
                     n: int) -> list[int]:
    """Wide (hidden_dim) layers alternate with pyramid layers.

    Pyramid layers linearly interpolate from input_dim to bottleneck.
    n = number of wide layers per side.

    n=3, bottleneck=input_dim*0.25:
      input → hidden → pyr(0.75) → hidden → pyr(0.5) → hidden → bottleneck
           → hidden → pyr(0.5) → hidden → pyr(0.75) → hidden → input
    """
    assert n >= 1, f"n must be >= 1, got {n}"
    if n == 1:
        return [input_dim, hidden_dim, bottleneck, hidden_dim, input_dim]

    step = (input_dim - bottleneck) / n

    enc_pyr = [int(input_dim - i * step) for i in range(1, n)]
    dec_pyr = list(reversed(enc_pyr))

    enc: list[int] = [input_dim]
    for pyr in enc_pyr:
        enc.append(hidden_dim)
        enc.append(pyr)
    enc.append(hidden_dim)
    enc.append(bottleneck)

    dec: list[int] = [hidden_dim]
    for pyr in dec_pyr:
        dec.append(pyr)
        dec.append(hidden_dim)
    dec.append(input_dim)

    return enc + dec


# ── Budget-constrained solvers ──────────────────────────────


def _binary_search_b(make_sizes_fn, target_params: int, input_dim: int,
                     lo: float = 0.1, hi: float = 20.0, iters: int = 30
                     ) -> tuple[float, int, int]:
    """Generic binary search for width parameter.

    Searches ∈ [lo, hi] for the value that minimises |n_params − target_params|.
    make_sizes_fn(param) → list[int] layer sizes.

    Returns (param, hidden_dim, n_params).
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
    param = (round(lo, 6) if abs(p_lo - target_params) <= abs(p_hi - target_params)
             else round(hi, 6))
    h = max(1, int(round(input_dim * param)))
    return param, h, _p(param)


def _binary_search_n(make_sizes_fn, target_params: int, hidden_dim: int,
                     lo: int = 1, hi: int = 20, iters: int = 30
                     ) -> tuple[int, int, int]:
    """Generic binary search for depth parameter.

    Searches n ∈ [lo, hi] for the value that minimises |n_params − target_params|.
    make_sizes_fn(n) → list[int] layer sizes (hidden_dim already baked in).

    Returns (n, hidden_dim, n_params).
    """
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


def solve_b_for_n(n_hidden: int, target_params: int, input_dim: int,
                  bottleneck: int) -> tuple[float, int, int]:
    return _binary_search_b(
        lambda h: make_rectangular(input_dim, h, bottleneck, n_hidden),
        target_params, input_dim)


def solve_b_for_n_trapezoid(n_hidden: int, target_params: int, input_dim: int,
                              bottleneck: int, alpha: float
                              ) -> tuple[float, int, int]:
    return _binary_search_b(
        lambda h: make_trapezoid(input_dim, h, bottleneck, n_hidden, alpha),
        target_params, input_dim)


def solve_b_for_n_interleaved(n: int, target_params: int, input_dim: int,
                               bottleneck: int
                               ) -> tuple[float, int, int]:
    return _binary_search_b(
        lambda h: make_interleaved(input_dim, h, bottleneck, n),
        target_params, input_dim)


def solve_n_for_b(b_val: float, target_params: int, input_dim: int,
                  bottleneck: int, max_n: int = 20
                  ) -> tuple[int, int, int]:
    h = max(1, int(round(input_dim * b_val)))
    return _binary_search_n(
        lambda n: make_rectangular(input_dim, h, bottleneck, n),
        target_params, h, lo=1, hi=max_n)


def solve_n_for_b_interleaved(b_val: float, target_params: int, input_dim: int,
                               bottleneck: int, max_n: int = 20
                               ) -> tuple[int, int, int]:
    h = max(1, int(round(input_dim * b_val)))
    return _binary_search_n(
        lambda n: make_interleaved(input_dim, h, bottleneck, n),
        target_params, h, lo=1, hi=max_n)


def solve_d_for_n(n: int, target_params: int, D: int, B: int,
                  max_d: float | None = None
                  ) -> tuple[float, int, int]:
    """Binary search d ∈ [0.1, max_d] for pyramid architecture.

    Returns (d, h_start, n_params).
    """
    if max_d is None:
        max_d = D * 10

    def _p(d_val):
        return count_params(make_pyramid(D, B, n, d_val))

    lo, hi = 0.1, max_d
    for _ in range(50):
        mid = (lo + hi) / 2
        if _p(mid) < target_params:
            lo = mid
        else:
            hi = mid

    p_lo, p_hi = _p(lo), _p(hi)
    d = lo if abs(p_lo - target_params) <= abs(p_hi - target_params) else hi
    d = round(d, 6)
    h_start = B + d
    return d, h_start, _p(d)


# ── Vary categories ─────────────────────────────────────────

MODEL_LEVEL_VARY = {'normalization', 'activation', 'dropout',
                    'norm_bottleneck', 'norm_last', 'bottleneck',
                    'trapezoid_alpha', 'residual', 'residual_norm'}
TRAIN_LEVEL_VARY = {'lr', 'scheduler', 'grad_clip', 'optimizer',
                    'weight_decay', 'batch_size', 'num_workers',
                    'noise_prob', 'noise_std'}


def resolve_architecture(vary_value, vary_name: str,
                         sweep_config: SweepConfig) -> dict:
    """Resolve full architecture for a sweep candidate.

    Given a SweepConfig + (vary_name, vary_value), returns:
      {'sizes': [D, ..., B, ..., D], 'b': ..., 'n': ...,
       'hidden_dim': ..., 'n_params': ...}
    """
    mc = sweep_config.model
    sc = sweep_config.sweep

    # Apply vary to the right config object
    if vary_name in MODEL_LEVEL_VARY:
        setattr(mc, vary_name, vary_value)
    elif vary_name in TRAIN_LEVEL_VARY:
        setattr(sweep_config.training, vary_name, vary_value)

    # If vary was model/train-level, fall back to n or b from fixed
    if vary_name in MODEL_LEVEL_VARY | TRAIN_LEVEL_VARY:
        fixed = dict(sc.fixed)
        if 'n' in fixed:
            vary_name, vary_value = 'n', fixed['n']
        elif 'b' in fixed:
            vary_name, vary_value = 'b', fixed['b']
        else:
            raise ValueError(
                f'vary={vary_name} needs either n= or b= in fixed')

    input_dim = mc.seq_len * UNICODE_BITS
    bottleneck = mc.bottleneck if mc.bottleneck is not None else mc.seq_len
    shape = getattr(mc, 'shape', 'rectangular')

    fixed = dict(sc.fixed)
    fixed[vary_name] = vary_value
    n = fixed.get('n', None)
    b_val = fixed.get('b', None)
    budget = sc.budget

    if shape == 'trapezoid':
        return _resolve_trapezoid(mc, sc, input_dim, bottleneck, n, b_val, budget)
    if shape == 'pyramid':
        return _resolve_pyramid(mc, sc, input_dim, bottleneck, n, b_val, budget)
    if shape == 'interleaved':
        return _resolve_interleaved(mc, sc, input_dim, bottleneck, n, b_val, budget)
    return _resolve_rectangular(sc, input_dim, bottleneck, n, b_val, budget)


def _resolve_trapezoid(mc, sc, input_dim, bottleneck, n, b_val, budget) -> dict:
    alpha = getattr(mc, 'trapezoid_alpha', 0.1)
    if sc.solve == 'b':
        assert n is not None, "need fixed n when solve=b"
        b_val, hidden_dim, _ = solve_b_for_n_trapezoid(
            n, budget, input_dim, bottleneck, alpha)
    elif sc.solve == 'n':
        raise NotImplementedError("solve=n not supported for trapezoid shape")
    elif n is not None and b_val is not None:
        hidden_dim = max(1, int(round(input_dim * b_val)))
    else:
        raise ValueError("Trapezoid shape needs solve=b with n, or n+b fixed")

    sizes = make_trapezoid(input_dim, hidden_dim, bottleneck, n, alpha)
    return _format_result(sizes, b_val, n, hidden_dim, input_dim)


def _resolve_pyramid(mc, sc, input_dim, bottleneck, n, b_val, budget) -> dict:
    if sc.solve == 'b':
        assert n is not None, "need fixed n when solve=b"
        d, h_start, _ = solve_d_for_n(n, budget, input_dim, bottleneck)
        sizes = make_pyramid(input_dim, bottleneck, n, d)
        return _format_result(sizes, round(h_start / input_dim, 6), n, h_start, input_dim)
    if sc.solve == 'n':
        raise NotImplementedError("solve=n not supported for pyramid shape")
    if n is not None and b_val is not None:
        h_start = int(input_dim * b_val)
        d = h_start - bottleneck
        sizes = make_pyramid(input_dim, bottleneck, n, d)
        return _format_result(sizes, b_val, n, h_start, input_dim)
    raise ValueError("Pyramid shape needs solve=b with n, or n+b fixed")


def _resolve_interleaved(mc, sc, input_dim, bottleneck, n, b_val, budget) -> dict:
    if sc.solve == 'b':
        assert n is not None, "need fixed n when solve=b"
        b_val, hidden_dim, _ = solve_b_for_n_interleaved(
            n, budget, input_dim, bottleneck)
    elif sc.solve == 'n':
        assert b_val is not None, "need fixed b when solve=n"
        n, hidden_dim, _ = solve_n_for_b_interleaved(
            b_val, budget, input_dim, bottleneck)
    elif n is not None and b_val is not None:
        hidden_dim = max(1, int(round(input_dim * b_val)))
    elif n is not None:
        raise ValueError(f"n={n} given but no solve/b — cannot determine hidden_dim")
    elif b_val is not None:
        hidden_dim = max(1, int(round(input_dim * b_val)))
        if budget is not None:
            n, _, _ = solve_n_for_b_interleaved(b_val, budget, input_dim, bottleneck)
        else:
            n = 1
    else:
        raise ValueError("Interleaved shape needs n+b, or solve+fixed")

    sizes = make_interleaved(input_dim, hidden_dim, bottleneck, n)
    return _format_result(sizes, b_val, n, hidden_dim, input_dim)


def _resolve_rectangular(sc, input_dim, bottleneck, n, b_val, budget) -> dict:
    if sc.solve == 'b':
        assert n is not None, "need fixed n when solve=b"
        b_val, hidden_dim, _ = solve_b_for_n(n, budget, input_dim, bottleneck)
    elif sc.solve == 'n':
        assert b_val is not None, "need fixed b when solve=n"
        n, hidden_dim, _ = solve_n_for_b(b_val, budget, input_dim, bottleneck)
    elif n is not None and b_val is not None:
        hidden_dim = max(1, int(round(input_dim * b_val)))
    elif n is not None:
        raise ValueError(f"n={n} given but no solve/b — cannot determine hidden_dim")
    elif b_val is not None:
        hidden_dim = max(1, int(round(input_dim * b_val)))
        if budget is not None:
            n, _, _ = solve_n_for_b(b_val, budget, input_dim, bottleneck)
        else:
            n = 1
    else:
        raise ValueError("Cannot determine architecture: need n+b, or solve+fixed")

    sizes = make_rectangular(input_dim, hidden_dim, bottleneck, n)
    return _format_result(sizes, b_val, n, hidden_dim, input_dim)


def _format_result(sizes, b_val, n, hidden_dim, input_dim) -> dict:
    return {
        'sizes': sizes,
        'b': round(hidden_dim / input_dim, 6) if b_val is None else b_val,
        'n': n, 'hidden_dim': hidden_dim,
        'n_params': count_params(sizes),
    }
