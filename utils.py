"""System-level utilities — CUDA helpers, GPU health checks."""

import atexit

import torch


def cuda_safe_cleanup():
    """Sync CUDA to avoid GPU ERR — call from MAIN THREAD only.

    Never call from signal handlers or subprocess forks.
    Robust against already-broken CUDA contexts.
    """
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()


atexit.register(cuda_safe_cleanup)


def gpu_health_check():
    """Check GPU is usable before starting training. Returns True if OK."""
    if not torch.cuda.is_available():
        return True
    try:
        torch.cuda.get_device_properties(0)
        t = torch.zeros(1, device='cuda')
        del t
        torch.cuda.empty_cache()
        return True
    except Exception:
        return False
