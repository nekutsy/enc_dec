"""Pyramid shape: layer widths linearly increase from bottleneck."""

from model.shapes import Shape, register, count_params, _format_result


@register
class PyramidShape(Shape):
    name = 'pyramid'

    def make_sizes(self, input_dim: int, hidden_dim: int, bottleneck: int,
                   enc_n: int, dec_n: int, d: float | None = None, **kwargs) -> list[int]:
        if d is None:
            d = hidden_dim - bottleneck

        def _build_side(n_layers, d_val, reverse: bool):
            result = []
            for i in range(1, n_layers + 1):
                idx = (n_layers - i + 1) if reverse else i
                result.append(int(bottleneck + d_val * idx / n_layers))
            return result

        enc: list[int] = [input_dim]
        enc += _build_side(enc_n, d, reverse=False)
        enc.append(bottleneck)

        dec: list[int] = []
        dec += _build_side(dec_n, d, reverse=True)
        dec.append(input_dim)

        return enc + dec

    def solve_width(self, target_params: int, input_dim: int, bottleneck: int,
                    enc_n: int, dec_n: int, **kwargs) -> tuple[float, int, int]:
        """Binary search d ∈ [0.1, input_dim*10] for pyramid."""
        D, B = input_dim, bottleneck
        max_d = D * 10

        def _p(d_val):
            sizes = self.make_sizes(D, 0, B, enc_n, dec_n, d=d_val)
            return count_params(sizes)

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
        return round(h_start / input_dim, 6), h_start, _p(d)

    def resolve(self, input_dim: int, bottleneck: int, enc_n: int, dec_n: int,
                budget: int | None, solve_mode: str | None, b_val: float | None,
                **kwargs) -> dict:
        if solve_mode == 'b':
            if budget is None:
                raise ValueError("Pyramid: solve=b requires budget")
            b_val, h_start, _ = self.solve_width(budget, input_dim, bottleneck, enc_n, dec_n)
            d = h_start - bottleneck
            sizes = self.make_sizes(input_dim, 0, bottleneck, enc_n, dec_n, d=d)
            return _format_result(sizes, b_val, enc_n, dec_n, h_start, input_dim)
        elif solve_mode == 'n':
            raise NotImplementedError("solve=n not supported for pyramid shape")
        elif b_val is not None:
            h_start = int(input_dim * b_val)
            d = h_start - bottleneck
            sizes = self.make_sizes(input_dim, 0, bottleneck, enc_n, dec_n, d=d)
            return _format_result(sizes, b_val, enc_n, dec_n, h_start, input_dim)
        raise ValueError("Pyramid shape needs solve=b, or b fixed")
