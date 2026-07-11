"""Data package — backward-compat re-exports from data.core."""

from data.core import (
    load_text,
    _build_full_bits,
    _compute_file_offsets,
    SlidingWindowDataset,
    _build_valid_indices,
    NoisyDataset,
    prepare_data,
    export_latent_vectors,
    load_latent_vectors,
)

# Pipeline (new)
from data.pipeline import DataPipeline, DataConfig
