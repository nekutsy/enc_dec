"""Architecture math — layer size builders + budget-constrained solvers.

Pure domain logic. No PyTorch (except count_params which uses simple arithmetic).
Depends on: encoding (for UNICODE_BITS), experiment.config (for SweepConfig).

Asymmetric support (v2): enc_n and dec_n can differ. Shape builders accept separate
counts. Solvers target total params across both sides.
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


# ── n-extraction helpers ────────────────────────────────────

def _get_ns(mc, fixed: dict) -> tuple[int, int]:
    """Extract (enc_n, dec_n) from model config + sweep fixed dict.

    Precedence: mc.enc_n > fixed['enc_n'] > fixed['n'] > mc.dec_n > fixed['dec_n'] > enc_n.
    """
    enc_n = mc.enc_n if mc.enc_n is not None else fixed.get('enc_n', fixed.get('n', None))
    if enc_n is None:
        raise ValueError("enc_n or n must be specified in ModelConfig or sweep.fixed")
    dec_n = mc.dec_n if mc.dec_n is not None else fixed.get('dec_n', enc_n)
    return int(enc_n), int(dec_n)


# ── Shape builders ──────────────────────────────────────────

def make_rectangular(input_dim: int, hidden_dim: int, bottleneck: int,
                     enc_n: int, dec_n: int | None = None) -> list[int]:
    """[input] → [hidden]×enc_n → [bottleneck] → [hidden]×dec_n → [input]"""
    if dec_n is None:
        dec_n = enc_n
    return (
        [input_dim]
        + [hidden_dim] * enc_n
        + [bottleneck]
        + [hidden_dim] * dec_n
        + [input_dim]
    )


def make_pyramid(input_dim: int, bottleneck: int,
                 enc_n: int, dec_n: int | None = None,
                 d: float | None = None) -> list[int]:
    """Build pyramid: D→h1→…→h_en→B→h_dn→…→h1→D

    d: width increment. Automatically computed from hidden_dim if None
       (d = (hidden_dim - B) * enc_n, but we need hidden_dim context).
       Caller should provide d explicitly; this is for manual layer building.
    """
    if dec_n is None:
        dec_n = enc_n

    def _build_side(n_layers, d_val, reverse: bool):
        result = []
        for i in range(1, n_layers + 1):
            idx = (n_layers - i + 1) if reverse else i
            result.append(int(B + d_val * idx / n_layers))
        return result

    B = bottleneck
    enc: list[int] = [input_dim]
    enc += _build_side(enc_n, d, reverse=False)
    enc.append(B)

    dec: list[int] = []
    dec += _build_side(dec_n, d, reverse=True)
    dec.append(input_dim)

    return enc + dec


def make_trapezoid(input_dim: int, hidden_dim: int, bottleneck: int,
                   enc_n: int, dec_n: int | None = None, alpha: float = 0.1
                   ) -> list[int]:
    """Trapezoid: layer widths linearly interpolate from hd×(1+α) to hd×(1−α).

    enc_n, dec_n layers per side, each at width:
      hd × (1 + α − 2α·(i−1)/(n−1))   for i ∈ [1, n]

    alpha=0.1: input → hd×1.1 → ... → hd×0.9 → bottleneck
    """
    if dec_n is None:
        dec_n = enc_n

    def _build_side(n_layers, reverse: bool):
        if n_layers == 1:
            return [hidden_dim]
        result = []
        for i in range(1, n_layers + 1):
            if reverse:
                idx = n_layers - i + 1
            else:
                idx = i
            frac = (idx - 1) / (n_layers - 1)
            w = int(round(hidden_dim * (1 + alpha - 2 * alpha * frac)))
            result.append(w)
        return result

    enc: list[int] = [input_dim]
    enc += _build_side(enc_n, reverse=False)
    enc.append(bottleneck)

    dec: list[int] = []
    dec += _build_side(dec_n, reverse=True)
    dec.append(input_dim)

    return enc + dec


def make_interleaved(input_dim: int, hidden_dim: int, bottleneck: int,
                     enc_n: int, dec_n: int | None = None) -> list[int]:
    """Wide (hidden_dim) layers alternate with pyramid layers.

    enc_n = number of wide layers per encoder side.
    dec_n = number of wide layers per decoder side.

    enc_n=3, bottleneck=input_dim*0.25:
      input → hidden → pyr(0.75) → hidden → pyr(0.5) → hidden → bottleneck
    """
    if dec_n is None:
        dec_n = enc_n

    def _build_side(n, reverse: bool):
        step = (input_dim - bottleneck) / n
        if n == 1:
            return [hidden_dim]

        pyr_values = [int(input_dim - i * step) for i in range(1, n)]
        if reverse:
            pyr_values = list(reversed(pyr_values))

        result = []
        if reverse:
            result.append(hidden_dim)
        for pyr in pyr_values:
            if reverse:
                result.append(pyr)
                result.append(hidden_dim)
            else:
                result.append(hidden_dim)
                result.append(pyr)
        if not reverse:
            result.append(hidden_dim)
        return result

    enc: list[int] = [input_dim]
    enc += _build_side(enc_n, reverse=False)
    enc.append(bottleneck)

    dec: list[int] = []
    dec += _build_side(dec_n, reverse=True)
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


def solve_b_for_rect(target_params: int, input_dim: int, bottleneck: int,
                     enc_n: int, dec_n: int) -> tuple[float, int, int]:
    return _binary_search_b(
        lambda h: make_rectangular(input_dim, h, bottleneck, enc_n, dec_n),
        target_params, input_dim)


def solve_b_for_trapezoid(target_params: int, input_dim: int, bottleneck: int,
                          enc_n: int, dec_n: int, alpha: float
                          ) -> tuple[float, int, int]:
    return _binary_search_b(
        lambda h: make_trapezoid(input_dim, h, bottleneck, enc_n, dec_n, alpha),
        target_params, input_dim)


def solve_b_for_interleaved(target_params: int, input_dim: int, bottleneck: int,
                            enc_n: int, dec_n: int
                            ) -> tuple[float, int, int]:
    return _binary_search_b(
        lambda h: make_interleaved(input_dim, h, bottleneck, enc_n, dec_n),
        target_params, input_dim)


def solve_d_for_n(target_params: int, D: int, B: int, enc_n: int, dec_n: int,
                  max_d: float | None = None
                  ) -> tuple[float, int, int]:
    """Binary search d ∈ [0.1, max_d] for pyramid architecture.

    Returns (d, h_start, n_params).
    """
    if max_d is None:
        max_d = D * 10

    def _p(d_val):
        return count_params(make_pyramid(D, B, enc_n, dec_n, d_val))

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


# ── Backward-compatible wrappers (single-n API) ─────────────

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
    h = max(1, int(round(input_dim * b_val)))
    return _binary_search_n(
        lambda n: make_rectangular(input_dim, h, bottleneck, n, n),
        target_params, h, lo=1, hi=max_n)


def solve_n_for_b_interleaved(b_val: float, target_params: int, input_dim: int,
                               bottleneck: int, max_n: int = 20
                               ) -> tuple[int, int, int]:
    h = max(1, int(round(input_dim * b_val)))
    return _binary_search_n(
        lambda n: make_interleaved(input_dim, h, bottleneck, n, n),
        target_params, h, lo=1, hi=max_n)


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


# ── Legacy pyramid solver ───────────────────────────────────

def solve_d_for_n_sym(n: int, target_params: int, D: int, B: int,
                      max_d: float | None = None
                      ) -> tuple[float, int, int]:
    """Legacy wrapper: symmetric pyramid."""
    return solve_d_for_n(target_params, D, B, n, n, max_d=max_d)


# ── Vary categories ─────────────────────────────────────────

MODEL_LEVEL_VARY = {'normalization', 'activation', 'dropout',
                    'norm_bottleneck', 'norm_last', 'bottleneck',
                    'trapezoid_alpha', 'residual', 'residual_norm'}
TRAIN_LEVEL_VARY = {'lr', 'scheduler', 'grad_clip', 'optimizer',
                    'weight_decay', 'batch_size', 'num_workers',
                    'noise_prob', 'noise_std'}

# n-like params: sweep-varying them changes layer counts
N_VARY = {'n', 'enc_n', 'dec_n'}


def resolve_architecture(vary_value, vary_name: str,
                         sweep_config: SweepConfig) -> dict:
    """Resolve full architecture for a sweep candidate.

    Given a SweepConfig + (vary_name, vary_value), returns:
      {'sizes': [D, ..., B, ..., D], 'b': ..., 'enc_n': ..., 'dec_n': ...,
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
        # If plain 'n' was set, use it as fallback for enc_n/dec_n
        if vary_name == 'n':
            fixed.setdefault('enc_n', vary_value)
            fixed.setdefault('dec_n', vary_value)
    elif vary_name == 'b':
        fixed['b'] = vary_value

    # total_n: auto-set complementary enc_n/dec_n from fixed sum
    if sc.total_n is not None:
        if 'enc_n' in fixed and 'dec_n' not in fixed:
            fixed['dec_n'] = sc.total_n - int(fixed['enc_n'])
        elif 'dec_n' in fixed and 'enc_n' not in fixed:
            fixed['enc_n'] = sc.total_n - int(fixed['dec_n'])

    # Dispatch per shape
    if shape == 'trapezoid':
        return _resolve_trapezoid(mc, sc, input_dim, bottleneck, fixed)
    if shape == 'pyramid':
        return _resolve_pyramid(mc, sc, input_dim, bottleneck, fixed)
    if shape == 'interleaved':
        return _resolve_interleaved(mc, sc, input_dim, bottleneck, fixed)
    return _resolve_rectangular(mc, sc, input_dim, bottleneck, fixed)


def _resolve_trapezoid(mc, sc, input_dim, bottleneck, fixed) -> dict:
    alpha = getattr(mc, 'trapezoid_alpha', 0.1)
    enc_n, dec_n = _get_ns(mc, fixed)
    b_val = fixed.get('b', None)
    budget = sc.budget

    if sc.solve == 'b':
        b_val, hidden_dim, _ = solve_b_for_trapezoid(
            budget, input_dim, bottleneck, enc_n, dec_n, alpha)
    elif sc.solve == 'n':
        raise NotImplementedError("solve=n not supported for trapezoid shape")
    elif b_val is not None:
        hidden_dim = max(1, int(round(input_dim * b_val)))
    else:
        raise ValueError("Trapezoid shape needs solve=b, or b fixed")

    sizes = make_trapezoid(input_dim, hidden_dim, bottleneck, enc_n, dec_n, alpha)
    return _format_result(sizes, b_val, enc_n, dec_n, hidden_dim, input_dim)


def _resolve_pyramid(mc, sc, input_dim, bottleneck, fixed) -> dict:
    enc_n, dec_n = _get_ns(mc, fixed)
    b_val = fixed.get('b', None)
    budget = sc.budget

    if sc.solve == 'b':
        d, h_start, _ = solve_d_for_n(budget, input_dim, bottleneck, enc_n, dec_n)
        sizes = make_pyramid(input_dim, bottleneck, enc_n, dec_n, d)
        return _format_result(sizes, round(h_start / input_dim, 6),
                              enc_n, dec_n, h_start, input_dim)
    if sc.solve == 'n':
        raise NotImplementedError("solve=n not supported for pyramid shape")
    if b_val is not None:
        h_start = int(input_dim * b_val)
        d = h_start - bottleneck
        sizes = make_pyramid(input_dim, bottleneck, enc_n, dec_n, d)
        return _format_result(sizes, b_val, enc_n, dec_n, h_start, input_dim)
    raise ValueError("Pyramid shape needs solve=b, or b fixed")


def _resolve_interleaved(mc, sc, input_dim, bottleneck, fixed) -> dict:
    enc_n, dec_n = _get_ns(mc, fixed)
    b_val = fixed.get('b', None)
    budget = sc.budget

    if sc.solve == 'b':
        b_val, hidden_dim, _ = solve_b_for_interleaved(
            budget, input_dim, bottleneck, enc_n, dec_n)
    elif sc.solve == 'n':
        assert b_val is not None, "need fixed b when solve=n"
        # symmetric for solve=n
        n, hidden_dim, _ = solve_n_for_b_interleaved(
            b_val, budget, input_dim, bottleneck)
        enc_n, dec_n = n, n
    elif b_val is not None:
        hidden_dim = max(1, int(round(input_dim * b_val)))
    else:
        raise ValueError("Interleaved shape needs b, or solve+fixed")

    sizes = make_interleaved(input_dim, hidden_dim, bottleneck, enc_n, dec_n)
    return _format_result(sizes, b_val, enc_n, dec_n, hidden_dim, input_dim)


def _resolve_rectangular(mc, sc, input_dim, bottleneck, fixed) -> dict:
    enc_n, dec_n = _get_ns(mc, fixed)
    b_val = fixed.get('b', None)
    budget = sc.budget

    if sc.solve == 'b':
        b_val, hidden_dim, _ = solve_b_for_rect(
            budget, input_dim, bottleneck, enc_n, dec_n)
    elif sc.solve == 'n':
        assert b_val is not None, "need fixed b when solve=n"
        # symmetric for solve=n
        n, hidden_dim, _ = solve_n_for_b(b_val, budget, input_dim, bottleneck)
        enc_n, dec_n = n, n
    elif b_val is not None:
        hidden_dim = max(1, int(round(input_dim * b_val)))
    else:
        raise ValueError("Cannot determine architecture: need b, or solve+fixed")

    sizes = make_rectangular(input_dim, hidden_dim, bottleneck, enc_n, dec_n)
    return _format_result(sizes, b_val, enc_n, dec_n, hidden_dim, input_dim)


# ── Legacy wrappers for callers that pass n/b directly ──────

def _resolve_rectangular_legacy(sc, input_dim, bottleneck, n, b_val, budget) -> dict:
    """Drop-in for old _resolve_rectangular signature."""
    if sc.solve == 'b':
        assert n is not None, "need fixed n when solve=b"
        b_val, hidden_dim, _ = solve_b_for_rect(
            budget, input_dim, bottleneck, n, n)
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

    sizes = make_rectangular(input_dim, hidden_dim, bottleneck, n, n)
    return _format_result(sizes, b_val, n, n, hidden_dim, input_dim)


def _format_result(sizes, b_val, enc_n, dec_n, hidden_dim, input_dim) -> dict:
    return {
        'sizes': sizes,
        'b': round(hidden_dim / input_dim, 6) if b_val is None else b_val,
        'enc_n': enc_n, 'dec_n': dec_n,
        'n': enc_n if enc_n == dec_n else None,  # legacy compat
        'hidden_dim': hidden_dim,
        'n_params': count_params(sizes),
    }
