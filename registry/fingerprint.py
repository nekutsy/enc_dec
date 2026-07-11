"""Deterministic hashes for architecture and training configs.

Auto-derived from dataclass asdict() — new fields in ModelConfig/TrainConfig
are automatically included without manual synchronisation.

Exclusions:
  _ARCH_EXCLUDE    — fields already encoded in sizes (seq_len, bottleneck)
  _TRAIN_EXCLUDE   — fields that don't affect training outcome (num_workers,
                     checkpoint_interval, early_stop_patience, target_samples)
"""

import hashlib
import json
from dataclasses import asdict


# Fields of ModelConfig already represented in `sizes` — excluded from hash.
# Add fields here only when they do NOT change the model architecture.
_ARCH_EXCLUDE = frozenset({'seq_len', 'bottleneck'})

# Fields of TrainConfig that do NOT affect training results.
_TRAIN_EXCLUDE = frozenset({
    'num_workers', 'checkpoint_interval', 'early_stop_patience',
    'target_samples',
})


def arch_fingerprint(sizes: list[int], mc: 'ModelConfig') -> str:
    """Deterministic 12-char hash from ModelConfig + layer sizes.

    New fields in ModelConfig are automatically included via asdict(mc).
    Only fields in _ARCH_EXCLUDE are omitted (already in sizes).
    """
    data = asdict(mc)
    for field in _ARCH_EXCLUDE:
        data.pop(field, None)
    data['_sizes'] = tuple(sizes)
    raw = json.dumps(data, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def training_hash(tc: 'TrainConfig') -> str:
    """Deterministic 12-char hash from TrainConfig.

    New fields in TrainConfig are automatically included via asdict(tc).
    Only fields in _TRAIN_EXCLUDE are omitted.
    None-valued optional fields (scheduler_config) are also stripped.
    """
    data = asdict(tc)
    for field in _TRAIN_EXCLUDE:
        data.pop(field, None)
    # Strip None-valued optional fields to preserve backward compat hashes
    data = {k: v for k, v in data.items() if v is not None}
    raw = json.dumps(data, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]
