"""ModelFactory — создание и конфигурация Autoencoder из architecture dict.

Usage:
    factory = ModelFactory(arch_dict, model_config, device)
    model = factory.build()
    model = factory.load_pretrain(model, pretrain_run_id, workspace)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import Autoencoder


class ModelFactory:
    """Create an Autoencoder from architecture + config."""

    def __init__(self, arch: dict, mc, device: torch.device):
        self._arch = arch
        self._mc = mc
        self._device = device

    def build(self) -> nn.Module:
        """Instantiate model on target device. Raises RuntimeError on OOM."""
        sizes = self._arch['sizes']
        try:
            return Autoencoder(
                sizes,
                activation=self._mc.activation,
                normalization=self._mc.normalization,
                init_gain=self._mc.init_gain,
                norm_bottleneck=self._mc.norm_bottleneck,
                norm_last=self._mc.norm_last,
                dropout=self._mc.dropout,
                residual=self._mc.residual,
                residual_norm=self._mc.residual_norm,
                enc_n=self._arch.get('enc_n'),
            ).to(self._device)
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError('oom')
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                raise RuntimeError('oom')
            raise

    @staticmethod
    def load_pretrain(model: nn.Module, pretrain_run_id: str,
                      workspace) -> nn.Module:
        """Load donor weights into model. Returns model."""
        from pathlib import Path

        donor_dir = workspace._find_run_dir(pretrain_run_id)
        if donor_dir is None:
            raise FileNotFoundError(f"Donor run {pretrain_run_id} not found")

        donor_path = Path(donor_dir) / 'model.pth'
        if not donor_path.is_file():
            donor_path = Path(donor_dir) / 'best.pth'
        if not donor_path.is_file():
            raise FileNotFoundError(f"Donor {pretrain_run_id}: no checkpoint")

        state = torch.load(str(donor_path), map_location='cpu', weights_only=True)
        unwrapped = model._orig_mod if hasattr(model, '_orig_mod') else model
        unwrapped.load_state_dict(state)
        del state
        n_params = len(unwrapped.state_dict())
        print(f'  pretrain: loaded weights from {donor_path} ({n_params} tensors)')
        return model

    @staticmethod
    def compile_model(model: nn.Module, device: torch.device) -> nn.Module:
        """Compile model for GPU — skip if VRAM-tight."""
        if device.type != 'cuda':
            return model
        n_params = sum(p.numel() for p in model.parameters())
        if n_params > 50_000:
            print(f'  ⚠ skipping torch.compile ({n_params:,} params — 8GB VRAM tight)')
        return model
