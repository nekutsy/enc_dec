"""Architecture math — layer size builders + budget-constrained solvers.

Pure domain logic. No PyTorch (except count_params which uses simple arithmetic).
Depends on: encoding (for UNICODE_BITS), sweep_config (for SweepConfig).
"""

from configs import UNICODE_BITS
from sweep_config import SweepConfig


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


# ── Budget-constrained solvers ──────────────────────────────

def solve_b_for_n(n_hidden: int, target_params: int, input_dim: int,
                  bottleneck: int) -> tuple[float, int, int]:
    """Binary search b ∈ [0.1, 20] for rectangular architecture.

    Returns (b, hidden_dim, n_params).
    """
    def _p(b_val):
        h = max(1, int(round(input_dim * b_val)))
        return count_params(make_rectangular(input_dim, h, bottleneck, n_hidden))

    lo, hi = 0.1, 20.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if _p(mid) < target_params:
            lo = mid
        else:
            hi = mid

    p_lo, p_hi = _p(lo), _p(hi)
    b_val = (round(lo, 6) if abs(p_lo - target_params) <= abs(p_hi - target_params)
             else round(hi, 6))
    h = max(1, int(round(input_dim * b_val)))
    return b_val, h, _p(b_val)


def solve_n_for_b(b_val: float, target_params: int, input_dim: int,
                  bottleneck: int, max_n: int = 20
                  ) -> tuple[int, int, int]:
    """Binary search n ∈ [1, max_n] for rectangular architecture.

    Returns (n, hidden_dim, n_params).
    """
    def _p(n):
        h = max(1, int(round(input_dim * b_val)))
        return count_params(make_rectangular(input_dim, h, bottleneck, int(n)))

    lo, hi = 1, max_n
    for _ in range(30):
        mid = int((lo + hi) // 2)
        if _p(mid) < target_params:
            lo = mid + 1
        else:
            hi = mid

    p_lo, p_hi = _p(lo), _p(hi)
    n = lo if abs(p_lo - target_params) <= abs(p_hi - target_params) else hi
    n = max(1, min(n, max_n))
    h = max(1, int(round(input_dim * b_val)))
    return n, h, _p(n)


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
                    'norm_bottleneck', 'norm_last'}
TRAIN_LEVEL_VARY = {'lr', 'scheduler', 'grad_clip', 'optimizer',
                    'weight_decay', 'batch_size', 'num_workers'}


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

    seq_len = mc.seq_len
    input_dim = seq_len * UNICODE_BITS
    bottleneck = mc.bottleneck if mc.bottleneck is not None else seq_len
    shape = getattr(mc, 'shape', 'rectangular')

    fixed = dict(sc.fixed)
    fixed[vary_name] = vary_value
    n = fixed.get('n', None)
    b_val = fixed.get('b', None)
    budget = sc.budget

    # ── Pyramid shape ──
    if shape == 'pyramid':
        if sc.solve == 'b':
            assert n is not None, "need fixed n when solve=b"
            d, h_start, n_params = solve_d_for_n(
                n, budget, input_dim, bottleneck)
            sizes = make_pyramid(input_dim, bottleneck, n, d)
            return {
                'sizes': sizes, 'b': round(h_start / input_dim, 6),
                'n': n, 'hidden_dim': h_start, 'n_params': n_params,
            }
        elif sc.solve == 'n':
            raise NotImplementedError(
                "solve=n not supported for pyramid shape")
        elif n is not None and b_val is not None:
            h_start = int(input_dim * b_val)
            d = h_start - bottleneck
            sizes = make_pyramid(input_dim, bottleneck, n, d)
            return {
                'sizes': sizes, 'b': b_val, 'n': n,
                'hidden_dim': h_start, 'n_params': count_params(sizes),
            }
        else:
            raise ValueError(
                "Pyramid shape needs solve=b with n, or n+b fixed")

    # ── Rectangular shape ──
    if sc.solve == 'b':
        assert n is not None, "need fixed n when solve=b"
        b_val, hidden_dim, n_params = solve_b_for_n(
            n, budget, input_dim, bottleneck)
    elif sc.solve == 'n':
        assert b_val is not None, "need fixed b when solve=n"
        n, hidden_dim, n_params = solve_n_for_b(
            b_val, budget, input_dim, bottleneck)
    elif n is not None and b_val is not None:
        hidden_dim = max(1, int(round(input_dim * b_val)))
        n_params = count_params(
            make_rectangular(input_dim, hidden_dim, bottleneck, n))
    elif n is not None:
        raise ValueError(
            f"n={n} given but no solve/b — cannot determine hidden_dim")
    elif b_val is not None:
        hidden_dim = max(1, int(round(input_dim * b_val)))
        if budget is not None:
            n, _, _ = solve_n_for_b(
                b_val, budget, input_dim, bottleneck)
        else:
            n = 1
        n_params = count_params(
            make_rectangular(input_dim, hidden_dim, bottleneck, n))
    else:
        raise ValueError(
            "Cannot determine architecture: need n+b, or solve+fixed")

    sizes = make_rectangular(input_dim, hidden_dim, bottleneck, n)
    n_params = count_params(sizes)
    return {
        'sizes': sizes,
        'b': round(hidden_dim / input_dim, 6) if b_val is None else b_val,
        'n': n, 'hidden_dim': hidden_dim, 'n_params': n_params,
    }
