"""Runtime context — device, text, logger. Pure infrastructure.

RuntimeContext is the non-serialisable runtime state that complements
SweepConfig (which is pure JSON-serialisable configuration).
"""

import torch
from torch import device as Device

from data import load_text
from sweep_config import OutputConfig
from logger import GlobalLogger


class RuntimeContext:
    """Transient runtime state — device, text corpus, global logger.

    Not serialisable. Not part of SweepConfig.
    Created once per process via setup_runtime().
    """

    def __init__(self, device: Device, text: str,
                 global_logger: GlobalLogger | None = None):
        self.device = device
        self.text = text
        self.global_logger = global_logger


def setup_runtime(output: OutputConfig,
                  global_logger: GlobalLogger | None = None,
                  text: str | None = None) -> RuntimeContext:
    """Resolve device + load text → RuntimeContext.

    Centralised replacement for the 8-line pattern duplicated across scripts.
    """
    if output.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(output.device)

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = False

    if text is None:
        text = load_text()

    return RuntimeContext(device, text, global_logger)
