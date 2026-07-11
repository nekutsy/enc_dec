"""Declarative CLI builder — auto-generate argparse from dataclass fields.

Adding a field to ModelConfig/TrainConfig → automatically available as CLI arg
without manual registration. Special fields (size strings, scheduler_config)
are handled via overrides.

Usage:
    from experiment.config import ModelConfig, TrainConfig
    from cli.args import add_dataclass_args, apply_dataclass_args

    p = argparse.ArgumentParser()
    add_dataclass_args(p, ModelConfig, prefix='', overrides={'bottleneck': {'nargs': '?'}})
    add_dataclass_args(p, TrainConfig, prefix='')
    args = p.parse_args()
    mc = apply_dataclass_args(args, ModelConfig())
    tc = apply_dataclass_args(args, TrainConfig())
"""

from __future__ import annotations

import argparse
import dataclasses
from typing import Any


# Special fields that are handled manually (CLI has custom parsing)
_MANUAL_FIELDS: dict[str, set[str]] = {
    'ModelConfig': {'enc_n', 'dec_n'},     # --enc-n, --dec-n handled with --n interactino
    'TrainConfig': {
        'pretrain_run_id',                  # --pretrain-from handled specially
        'scheduler_config',                 # --scheduler-params handled specially
        # GreedySimple params have their own --gs-* flags for backward compat
        'greedy_simple_inc', 'greedy_simple_dec', 'greedy_simple_patience',
        'greedy_simple_warmup', 'greedy_simple_min_lr', 'greedy_simple_max_lr',
        'greedy_simple_warmup_start',
        # GreedyGrad has many fields, handled by --scheduler-params
        'greedy_grad_window', 'greedy_grad_alpha', 'greedy_grad_momentum',
        'greedy_grad_explore', 'greedy_grad_min_lr', 'greedy_grad_max_lr',
        'greedy_grad_warmup', 'greedy_grad_plateau_patience',
        'greedy_grad_plateau_multiplier', 'greedy_grad_plateau_cooldown',
        'greedy_factor', 'greedy_beta', 'greedy_lock_steps',
        'greedy_probe_patience', 'greedy_probe_factor', 'greedy_probe_spike_ratio',
        'greedy_probe_lock', 'greedy_cooldown',
    },
}

# Fields excluded from CLI entirely (runtime-only, or set by other means)
_EXCLUDE_FIELDS = {'enc_n', 'dec_n', 'pretrain_run_id', 'scheduler_config',
                   'greedy_simple_inc', 'greedy_simple_dec', 'greedy_simple_patience',
                   'greedy_simple_warmup', 'greedy_simple_min_lr', 'greedy_simple_max_lr',
                   'greedy_simple_warmup_start',
                   'greedy_grad_window', 'greedy_grad_alpha', 'greedy_grad_momentum',
                   'greedy_grad_explore', 'greedy_grad_min_lr', 'greedy_grad_max_lr',
                   'greedy_grad_warmup', 'greedy_grad_plateau_patience',
                   'greedy_grad_plateau_multiplier', 'greedy_grad_plateau_cooldown',
                   'greedy_factor', 'greedy_beta', 'greedy_lock_steps',
                   'greedy_probe_patience', 'greedy_probe_factor', 'greedy_probe_spike_ratio',
                   'greedy_probe_lock', 'greedy_cooldown',
                   'num_workers',
                   }


def _field_to_arg_name(field_name: str) -> str:
    """Convert snake_case dataclass field to --kebab-case CLI arg."""
    return '--' + field_name.replace('_', '-')


def _field_to_dest(field_name: str) -> str:
    """Convert snake_case to argparse dest (keeps underscores)."""
    return field_name.replace('-', '_')


def _infer_type_and_default(field: dataclasses.Field) -> tuple[type, Any]:
    """Infer argparse type and default from dataclass field."""
    default = field.default if field.default is not dataclasses.MISSING else None

    # Get type from annotation
    annotation = field.type
    origin = getattr(annotation, '__origin__', None)

    if origin is not None:
        # Handle Optional[X] / X | None
        args = getattr(annotation, '__args__', ())
        if type(None) in args:
            for a in args:
                if a is not type(None):
                    annotation = a
                    break

    # Map Python types to argparse-compatible types
    type_mapping = {
        int: int, float: float, str: str, bool: None,  # bool → store_true/store_false
    }
    arg_type = type_mapping.get(annotation, str)
    return arg_type, default if default is not dataclasses.MISSING else None


def _get_help(field: dataclasses.Field) -> str:
    """Extract help text from field metadata or name."""
    return field.metadata.get('help', '')


def add_dataclass_args(parser: argparse.ArgumentParser, dc_class: type,
                       prefix: str = '', defaults: dict | None = None,
                       overrides: dict | None = None):
    """Add argparse arguments from dataclass fields.

    Skips fields in _EXCLUDE_FIELDS.
    overrides: {field_name: extra_argparse_kwargs}
    """
    overrides = overrides or {}
    cls_name = dc_class.__name__

    for field in dataclasses.fields(dc_class):
        name = field.name
        if name in _EXCLUDE_FIELDS:
            continue

        arg_type, default = _infer_type_and_default(field)

        # Override default from dict
        if defaults and name in defaults:
            default = defaults[name]

        kwargs = {}

        # Boolean → store_true/store_false
        if arg_type is None and isinstance(default, bool):
            if default:
                kwargs['action'] = 'store_false'
            else:
                kwargs['action'] = 'store_true'
        else:
            kwargs['type'] = arg_type
            if default is not None:
                kwargs['default'] = default

        kwargs.update(overrides.get(name, {}))
        arg_name = _field_to_arg_name(name)
        parser.add_argument(arg_name, **kwargs)


def apply_dataclass_args(args: argparse.Namespace, instance: Any) -> Any:
    """Apply parsed CLI args to a dataclass instance. Returns the instance."""
    for field in dataclasses.fields(instance):
        name = field.name
        if name in _EXCLUDE_FIELDS:
            continue
        dest = _field_to_dest(name)
        if hasattr(args, dest):
            val = getattr(args, dest)
            if val is not None or not hasattr(instance, name) or getattr(instance, name) is None:
                setattr(instance, name, val)
    return instance
