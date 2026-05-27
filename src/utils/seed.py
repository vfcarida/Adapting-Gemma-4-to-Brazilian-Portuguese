"""
src/utils/seed.py
─────────────────
Global reproducibility seed management.
Sets seeds for ``random``, ``numpy``, ``torch``, and ``transformers``.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set seeds across all relevant libraries for reproducibility.

    Parameters
    ----------
    seed : int
        The seed value to use everywhere.
    deterministic : bool
        If ``True``, enables CUDA deterministic mode and disables
        benchmark-mode cuDNN auto-tuner.  This may reduce performance
        slightly but guarantees bitwise-reproducible results on the same
        hardware.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch ≥ 1.8 — algorithm selection determinism
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            # Older PyTorch versions without warn_only
            torch.use_deterministic_algorithms(True)

    # HF Transformers seed (sets its own internal RNG)
    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
    except ImportError:
        pass
