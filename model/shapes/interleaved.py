"""Interleaved shape: wide layers alternate with pyramid layers."""

from model.shapes import Shape, register, _binary_search_b, _binary_search_n


@register
class InterleavedShape(Shape):
    name = 'interleaved'

    def make_sizes(self, input_dim: int, hidden_dim: int, bottleneck: int,
                   enc_n: int, dec_n: int, **kwargs) -> list[int]:
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

    def solve_width(self, target_params: int, input_dim: int, bottleneck: int,
                    enc_n: int, dec_n: int, **kwargs) -> tuple[float, int, int]:
        return _binary_search_b(
            lambda h: self.make_sizes(input_dim, h, bottleneck, enc_n, dec_n),
            target_params, input_dim)

    def solve_depth(self, b_val: float, target_params: int, input_dim: int,
                    bottleneck: int) -> tuple[int, int, int] | None:
        h = max(1, int(round(input_dim * b_val)))
        return _binary_search_n(
            lambda n: self.make_sizes(input_dim, h, bottleneck, n, n),
            target_params, h, lo=1, hi=20)
