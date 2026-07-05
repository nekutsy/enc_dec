"""Registry — experiment tracking, model deduplication, run management.

Exports:
  Registry        — SQLite-backed CRUD for experiments, architectures, runs
  arch_fingerprint — deterministic hash from ModelConfig + sizes
  training_hash   — deterministic hash from TrainConfig
  Workspace       — file layout: sessions/runs/{id}/*
"""

from registry.fingerprint import arch_fingerprint, training_hash
from registry.db import Registry
