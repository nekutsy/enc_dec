"""Backward-compatible re-exports → experiment.config + experiment.presets.

This file is kept for backward compatibility only.
For new code, import from:
  - experiment.config   (SweepConfig, ModelConfig, TrainConfig, etc.)
  - experiment.presets  (preset_ratio, preset_binary, etc.)
"""

from experiment.config import (                    # noqa: F401
    SweepConfig, ModelConfig, TrainConfig,
    SweepSpec, OutputConfig,
)
from experiment.presets import (                   # noqa: F401
    preset_ratio, preset_binary, preset_width, preset_batch,
)
