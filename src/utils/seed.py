"""Reproducibility utilities.

Note: os.environ["PYTHONHASHSEED"] only affects hash randomization for a
*new* interpreter process — setting it here after startup is a no-op for the
current process, but is still set so subprocesses spawned later (e.g. via
DeepSpeed/dataloader workers with fresh Python interpreters) inherit it.
"""

import os
import random

import numpy as np


def set_seed(seed: int = 42, full_determinism: bool = False) -> None:
    """Set all random seeds for reproducibility.

    Sets seeds for: random, numpy, torch (if available).
    Also configures deterministic behavior in CUDA when possible.

    Args:
        seed: Seed value applied to all RNGs.
        full_determinism: If True, additionally calls
            ``torch.use_deterministic_algorithms(True)`` and sets
            ``CUBLAS_WORKSPACE_CONFIG``. This makes individual ops
            bit-reproducible but measurably slows training (some kernels fall
            back to slow deterministic implementations, and a few ops with no
            deterministic implementation will raise). Intended for debugging
            /golden tests, not full CPT/SFT runs — see
            docs/CPT_BEST_PRACTICES_RESEARCH.md for the tradeoff.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if full_determinism:
        # Must be set before CUDA is initialized to take effect.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        if full_determinism:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
