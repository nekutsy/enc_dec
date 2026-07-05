"""Orchestration — run and sweep management.

Exports:
  Workspace  — file layout: sessions/runs/{id}/*
  Run        — single training (replaces experiment/train_one)
  Sweep      — sweep orchestration with pluggable strategies
"""

from orchestration.workspace import Workspace
from orchestration.run import Run
from orchestration.sweep import Sweep
