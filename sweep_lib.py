"""Backward-compatible re-exports — delegates to new module locations.

This file is kept for backward compatibility only.
For new code, import directly from:
  - training.optimizers   (Lion, Sophia, build_optimizer)
  - model.architecture    (count_params, make_*, solve_*, resolve_architecture)
  - experiment.context    (RuntimeContext, setup_runtime)
  - experiment.train_one  (train_one, compile_model, train_setup, save_paths)
"""

from training.optimizers import Lion, Sophia, build_optimizer             # noqa: F401
from model.architecture import (                                          # noqa: F401
    count_params, make_rectangular, make_pyramid, make_interleaved,
    solve_b_for_n, solve_n_for_b, solve_d_for_n,
    solve_b_for_n_interleaved, solve_n_for_b_interleaved,
    resolve_architecture, MODEL_LEVEL_VARY, TRAIN_LEVEL_VARY,
)
from experiment.context import RuntimeContext, setup_runtime              # noqa: F401
from experiment.train_one import (                                     # noqa: F401
    train_one, compile_model, save_paths,
    _train_setup as train_setup,
)

# backward-compat re-exports from logger
from logger import (                                                      # noqa: F401
    TrainingLogger, GlobalLogger, LoggerConfig, get_last_samples,
    init_log, gather_done, log_row, UNIFIED_COLUMNS,
)

# backward-compat re-export of cuda helpers
from utils import cuda_safe_cleanup, gpu_health_check                     # noqa: F401

# backward-compat import of needed types
from sweep_config import SweepConfig, ModelConfig, TrainConfig, OutputConfig  # noqa: F401
