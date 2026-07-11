"""Trapezoid shape: linear width variation within each side."""

from model.shapes import Shape, register, _binary_search_b


@register
class TrapezoidShape(Shape):
    name = 'trapezoid'

    def make_sizes(self, input_dim: int, hidden_dim: int, bottleneck: int,
                   enc_n: int, dec_n: int, alpha: float = 0.1, **kwargs) -> list[int]:
        def _build_side(n_layers, reverse: bool):
            if n_layers == 1:
                return [hidden_dim]
            result = []
            for i in range(1, n_layers + 1):
                idx = n_layers - i + 1 if reverse else i
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

    def solve_width(self, target_params: int, input_dim: int, bottleneck: int,
                    enc_n: int, dec_n: int, alpha: float = 0.1, **kwargs) -> tuple[float, int, int]:
        return _binary_search_b(
            lambda h: self.make_sizes(input_dim, h, bottleneck, enc_n, dec_n, alpha=alpha),
            target_params, input_dim)
