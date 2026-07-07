"""Run — single model training, registry-aware.

Replaces experiment/train_one.py. Uses Registry for deduplication and tracking,
Workspace for file layout.

Key difference from train_one: Run.find_or_create() checks the registry
before training — if an identical model was already trained, it skips.
"""

import csv
import gc
import json
import os
import sys
import time as time_mod

import torch
import torch.nn as nn

from configs import UNICODE_BITS
from model import Autoencoder
from data import prepare_data, NoisyDataset
from training import (
    run_training, build_scheduler,
    save_checkpoint, load_optimizer, load_plat_scheduler, load_step_scheduler,
)
from training.optimizers import build_optimizer
from utils import cuda_safe_cleanup, gpu_health_check
from logger import TrainingLogger, LoggerConfig, get_last_samples

from registry.fingerprint import arch_fingerprint, training_hash
from registry.db import Registry, RunResult
from orchestration.workspace import Workspace


def compile_model(model, device):
    """Compile model for GPU — skip unsafe cases."""
    if device.type != 'cuda':
        return model
    n_params = sum(p.numel() for p in model.parameters())
    if n_params <= 50_000:
        return model
    has_bn = any(isinstance(m, nn.BatchNorm1d) for m in model.modules())
    if has_bn:
        return model
    try:
        return torch.compile(model, mode='default')
    except Exception as e:
        print(f'  ⚠ torch.compile failed ({e}) — running uncompiled')
        return model


class Run:
    """One training run — find-or-create semantics via Registry.

    Usage:
        # Find existing or create new
        run, created = Run.find_or_create(arch, config, registry, workspace)
        if run is None:
            print("Already done — skip")
        else:
            result = run.execute(runtime, ...)
    """

    def __init__(self, run_id: str, arch: dict, mc, tc,
                 registry: Registry, workspace: Workspace,
                 exp_name: str = '', model_name: str = ''):
        self.run_id = run_id
        self.arch = arch
        self.mc = mc
        self.tc = tc
        self.registry = registry
        self.ws = workspace
        self.exp_name = exp_name
        self.model_name = model_name

    # ── Factory: find existing or create ───────────────────

    @classmethod
    def find_or_create(cls, arch: dict, mc, tc,
                       registry: Registry, workspace: Workspace,
                       exp_name: str = '') -> tuple['Run | None', bool]:
        """Find completed run or create new. Returns (Run, created_new).

        Returns (None, False) if an identical completed run already exists.
        """
        sizes = arch['sizes']
        fp = arch_fingerprint(sizes, mc)
        th = training_hash(tc)

        # Register architecture if new
        import json as _json
        registry.register_architecture(
            fp, arch['n_params'], mc.seq_len,
            getattr(mc, 'shape', 'rectangular'),
            _json.dumps(mc.to_dict() if hasattr(mc, 'to_dict') else mc),
        )

        # Check for completed run with enough samples
        done = registry.get_completed_run(fp, th, min_samples=tc.target_samples)
        if done:
            model_name = done.get('model_name', '')
            run = cls(done['id'], arch, mc, tc, registry, workspace, exp_name, model_name)
            return run, False  # exists, fully done

        # Find existing (possibly partial) run or create new
        model_name = cls._model_name(arch, mc, vary_value=None)
        run_id, created = registry.find_or_create_run(
            fp, th, model_name,
            _json.dumps(mc.to_dict() if hasattr(mc, 'to_dict') else mc),
            _json.dumps(tc.to_dict() if hasattr(tc, 'to_dict') else tc),
        )
        workspace.write_meta(run_id, arch, mc, tc, exp_name, model_name)
        run = cls(run_id, arch, mc, tc, registry, workspace, exp_name, model_name)
        return run, created

    @staticmethod
    def _model_name(arch: dict, mc, vary_value=None) -> str:
        """Human-readable model name."""
        shape = getattr(mc, 'shape', 'rectangular')
        parts = [shape[:4], f's{mc.seq_len}']
        if shape == 'trapezoid':
            alpha = getattr(mc, 'trapezoid_alpha', 0.1)
            parts.append(f'a{alpha}')
        parts.append(f'n{arch.get("n", arch.get("n_hidden", "?"))}')
        if arch.get('b') is not None:
            parts.append(f'b{arch["b"]:.4g}')
        return '_'.join(parts)

    # ── Execution ─────────────────────────────────────────

    def execute(self, runtime, log_config: LoggerConfig | None = None,
                no_val: bool = True) -> RunResult:
        """Train the model. Returns RunResult. Updates registry on completion."""
        self.registry.start_run(self.run_id)

        sizes = self.arch['sizes']
        n_params = self.arch['n_params']
        seq_len = self.mc.seq_len
        bottleneck = self.mc.bottleneck if self.mc.bottleneck is not None else seq_len
        device = runtime.device
        text = runtime.text
        bs = self.tc.batch_size
        target_samples = self.tc.target_samples

        model_path = str(self.ws.model_path(self.run_id, self.model_name))
        best_path = str(self.ws.best_path(self.run_id, self.model_name))
        csv_path = str(self.ws.log_csv_path(self.run_id, self.model_name))
        model_dir = str(self.ws.run_dir(self.run_id, self.model_name))

        arch_str = '→'.join(str(s) for s in sizes)
        print(f'  run: {self.run_id}')
        print(f'  arch: {arch_str}')
        print(f'  params: {n_params:,}  batch: {bs}')

        # Already done? (double-check in case of race)
        if os.path.isfile(csv_path):
            last_samples = get_last_samples(csv_path)
            if last_samples >= target_samples:
                result = self._read_result_from_csv(csv_path, last_samples)
                self.registry.finish_run(self.run_id, result)
                self.ws.write_result(self.run_id, result, self.model_name)
                return result

        # ── Data ──
        train_ds, val_ds = prepare_data(text, seq_len, self.tc.train_ratio)
        if self.tc.noise_prob > 0.0:
            train_ds = NoisyDataset(
                train_ds, noise_prob=self.tc.noise_prob, noise_std=self.tc.noise_std)
            print(f'  noise: prob={self.tc.noise_prob}, std={self.tc.noise_std}')

        # ── Model ──
        try:
            model = Autoencoder(
                sizes, activation=self.mc.activation,
                normalization=self.mc.normalization,
                init_gain=self.mc.init_gain,
                norm_bottleneck=self.mc.norm_bottleneck,
                norm_last=self.mc.norm_last,
                dropout=self.mc.dropout,
            ).to(device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and 'out of memory' not in str(e).lower():
                raise
            print('  ⚠ OOM during creation')
            cuda_safe_cleanup()
            result = RunResult(status='oom')
            self.registry.finish_run(self.run_id, result)
            self.ws.write_result(self.run_id, result, self.model_name)
            return result

        try:
            model = compile_model(model, device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and 'out of memory' not in str(e).lower():
                raise
            print('  ⚠ OOM during compile')
            cuda_safe_cleanup()
            result = RunResult(status='oom')
            self.registry.finish_run(self.run_id, result)
            self.ws.write_result(self.run_id, result, self.model_name)
            return result

        # ── Optimizer ──
        optimizer = build_optimizer(model, self.tc, device)

        total_batches = int(target_samples / bs) + 1
        start_samples, ckpt_path = self._train_setup(
            model, optimizer, csv_path, model_path, model_dir, device)

        step_scheduler, checkpoint_scheduler = build_scheduler(
            optimizer, self.tc, total_batches, start_samples=start_samples)
        if start_samples > 0 and checkpoint_scheduler is not None:
            load_plat_scheduler(checkpoint_scheduler, ckpt_path)
        if start_samples > 0 and step_scheduler is not None:
            load_step_scheduler(step_scheduler, ckpt_path)

        criterion = nn.BCEWithLogitsLoss()

        # ── Logger ──
        lc = log_config or LoggerConfig.full()
        train_logger = TrainingLogger(csv_path, config=lc, model_name=self.model_name,
                                       log_path=str(self.ws.log_txt_path(self.run_id, self.model_name)))

        header_lines = [
            f'run: {self.run_id}',
            f'arch: {arch_str}',
            f'params: {n_params:,}  batch: {bs}',
        ]
        if self.tc.noise_prob > 0.0:
            header_lines.append(f'noise: prob={self.tc.noise_prob}, std={self.tc.noise_std}')
        train_logger.log_header(header_lines)

        rem = max(0, target_samples - start_samples)
        if rem <= 0:
            result = self._read_result_from_csv(csv_path, start_samples)
            self.registry.finish_run(self.run_id, result)
            self.ws.write_result(self.run_id, result, self.model_name)
            return result

        print(f'  training {rem:,} samples...')
        t_start = time_mod.time()

        try:
            final_samples = run_training(
                start_samples, target_samples, model, optimizer, criterion,
                train_ds, val_ds, train_logger, model_path, bs,
                seq_len, self.tc.grad_clip, self.tc.num_workers,
                step_scheduler=step_scheduler,
                checkpoint_scheduler=checkpoint_scheduler,
                early_stop_patience=self.tc.early_stop_patience,
                no_val=no_val,
                val_interval=self.tc.checkpoint_interval,
            )

            dur = time_mod.time() - t_start
            final_train_loss = self._read_final_loss(csv_path, train_logger)
            final_val_loss = self._read_final_val_loss(csv_path, train_logger)
            speed_avg = final_samples / dur if dur > 0 else 0.0
            print(f'  done: {final_samples:,} samples in {dur:.0f}s '
                  f'({speed_avg:,.0f} sps)  train={final_train_loss:.6f}')

            result = RunResult(
                final_train_loss=final_train_loss,
                final_val_loss=final_val_loss,
                total_samples=final_samples,
                duration_seconds=round(dur, 1),
                status='done',
            )

        except torch.cuda.OutOfMemoryError:
            print('  ⚠ OOM')
            result = RunResult(status='oom', error_message='CUDA OOM')
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print('  ⚠ OOM')
                result = RunResult(status='oom', error_message=str(e))
            else:
                raise
        except KeyboardInterrupt:
            print('\n  ⚠ Interrupted')
            result = RunResult(
                total_samples=start_samples,
                status='interrupted',
                error_message='KeyboardInterrupt',
            )
        finally:
            del model
            del optimizer
            gc.collect()
            cuda_safe_cleanup()

        self.registry.finish_run(self.run_id, result)
        self.ws.write_result(self.run_id, result, self.model_name)
        return result

    # ── Internal helpers ───────────────────────────────────

    def _train_setup(self, model, optimizer, csv_path, model_path,
                     model_dir, device):
        """Load checkpoint if available → (start_samples, effective_path)."""
        start_samples = get_last_samples(csv_path)
        effective_path = model_path
        if start_samples > 0:
            if not os.path.isfile(effective_path):
                best_path = os.path.join(model_dir, 'best.pth')
                if os.path.isfile(best_path):
                    effective_path = best_path
                    print('  Using best checkpoint (model.pth missing)')
            if not os.path.isfile(effective_path):
                print('  No checkpoint found — starting from scratch')
                return 0, model_path
            print(f'  Resuming from {start_samples:,} samples. Loading checkpoint...')
            state = torch.load(effective_path, map_location=device, weights_only=True)
            has_prefix = any(k.startswith('_orig_mod.') for k in state.keys())
            unwrapped = (model._orig_mod if hasattr(model, '_orig_mod') else model)
            if has_prefix:
                state = {k[len('_orig_mod.'):]: v for k, v in state.items()}
            unwrapped.load_state_dict(state)
            load_optimizer(optimizer, effective_path, device)
        return start_samples, effective_path

    def _read_final_loss(self, csv_path, train_logger) -> float:
        """Read final train_loss from CSV."""
        loss = train_logger.ema_loss or 0.0
        try:
            with open(csv_path) as f:
                reader = csv.reader(f)
                header = next(reader)
                last_row = None
                for row in reader:
                    last_row = row
                if last_row:
                    try:
                        col = header.index('train_loss')
                        loss = float(last_row[col])
                    except (ValueError, IndexError):
                        loss = (float(last_row[2]) if len(last_row) > 2 else loss)
        except Exception:
            pass
        return loss

    def _read_final_val_loss(self, csv_path, train_logger) -> float | None:
        """Read final val_loss from CSV."""
        try:
            with open(csv_path) as f:
                reader = csv.reader(f)
                header = next(reader)
                if 'val_loss' not in header:
                    return None
                last_row = None
                for row in reader:
                    last_row = row
                if last_row:
                    try:
                        col = header.index('val_loss')
                        val = last_row[col]
                        return float(val) if val else None
                    except (ValueError, IndexError):
                        return None
        except Exception:
            pass
        return None

    def _read_result_from_csv(self, csv_path, total_samples) -> RunResult:
        """Read result from existing CSV for already-done check."""
        loss = 0.0
        try:
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                last_row = None
                for row in reader:
                    last_row = row
            if last_row:
                loss = float(last_row.get('train_loss', 0) or 0)
        except Exception:
            pass
        print(f'  already done ({total_samples:,} samples, train={loss:.6f})')
        return RunResult(
            final_train_loss=loss,
            total_samples=total_samples,
            duration_seconds=0.0,
            status='done',
        )
