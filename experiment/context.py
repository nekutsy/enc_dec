"""Runtime context — device, text. Pure infrastructure.

RuntimeContext is the non-serialisable runtime state that complements
SweepConfig (which is pure JSON-serialisable configuration).

Note: global_logger removed; use registry.db for experiment tracking.
"""

import torch
from torch import device as Device

from data import load_text
from experiment.config import OutputConfig


class RuntimeContext:
    """Transient runtime state — device, text corpus.

    Not serialisable. Not part of SweepConfig.
    Created once per process via setup_runtime().
    """

    def __init__(self, device: Device, texts: list[str]):
        self.device = device
        self.texts = texts

    @property
    def text(self) -> str:
        """Legacy: joined text. Prefer .texts for prepare_data."""
        return ''.join(self.texts)


def setup_runtime(output: OutputConfig,
                  texts: list[str] | None = None,
                  use_tf32: bool = True) -> RuntimeContext:
    """Resolve device + load text → RuntimeContext.

    Centralised replacement for the 8-line pattern duplicated across scripts.
    """
    if output.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(output.device)

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = use_tf32

    if texts is None:
        texts = load_text()

    return RuntimeContext(device, texts)
