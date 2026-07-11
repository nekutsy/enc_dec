"""Rectangular shape: all hidden layers same width."""

from model.shapes import Shape, register, _binary_search_b, _binary_search_n


@register
class RectangularShape(Shape):
    name = 'rectangular'

    def make_sizes(self, input_dim: int, hidden_dim: int, bottleneck: int,
                   enc_n: int, dec_n: int, **kwargs) -> list[int]:
        return (
            [input_dim]
            + [hidden_dim] * enc_n
            + [bottleneck]
            + [hidden_dim] * dec_n
            + [input_dim]
        )

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
