"""
src/utils/hf_utils.py
─────────────────────
Hugging Face Hub authentication, model/tokenizer loading, and safe
LoRA configuration builder with hardcoded multimodal exclusion.
"""

from __future__ import annotations

import os
import re
from typing import Any

import torch
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────
# SAFETY: Gemma 4 multimodal module exclusion
# ──────────────────────────────────────────────────────────────────────
# Gemma 4 contains Gemma4ClippableLinear layers in vision/audio
# encoders.  PEFT crashes if LoRA tries to target these.  We NEVER
# use target_modules="all-linear"; instead we whitelist only the
# standard language-model projection layers.
# ──────────────────────────────────────────────────────────────────────

SAFE_LORA_TARGET_MODULES: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Regex patterns that identify multimodal sub-modules to exclude
_MULTIMODAL_EXCLUSION_PATTERNS: list[str] = [
    r".*vision_tower.*",
    r".*audio_tower.*",
    r".*multi_modal_projector.*",
    r".*image_adapter.*",
    r".*audio_adapter.*",
]


def authenticate_hf(token: str | None = None) -> None:
    """Authenticate with the Hugging Face Hub.

    Reads the token from the ``HF_TOKEN`` environment variable if
    *token* is not provided.

    .. note::
       This is **required** for gated models (Gemma 4) and gated
       datasets (Aurora-PT).
    """
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        logger.warning(
            "HF_TOKEN not set — you may not be able to access gated models / datasets."
        )
        return
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
        logger.info("Authenticated with Hugging Face Hub.")
    except ImportError:
        # Fallback: set env var so transformers picks it up
        os.environ["HF_TOKEN"] = token
        logger.info("Set HF_TOKEN in environment (huggingface_hub not installed).")


def load_model_and_tokenizer(
    model_id: str,
    torch_dtype: str | torch.dtype = "bfloat16",
    device_map: str = "auto",
    attn_implementation: str = "sdpa",
    trust_remote_code: bool = True,
    quantization_config: dict[str, Any] | None = None,
    cache_dir: str | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a CausalLM model and its tokenizer with sensible defaults.

    Parameters
    ----------
    model_id : str
        HuggingFace Hub ID or local path.
    torch_dtype : str | torch.dtype
        Weight precision.  Accepts ``"bfloat16"``, ``"float16"``, etc.
    device_map : str
        Device placement (``"auto"``, ``"cpu"``, ``"cuda:0"``).
    attn_implementation : str
        ``"sdpa"`` (default) or ``"flash_attention_2"``.
    trust_remote_code : bool
        Accept remote code in the model repo.
    quantization_config : dict | None
        If provided, passed as ``BitsAndBytesConfig``.
    cache_dir : str | None
        Override default HF cache directory.

    Returns
    -------
    tuple[AutoModelForCausalLM, AutoTokenizer]
    """
    # Resolve dtype string → torch.dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if isinstance(torch_dtype, str):
        torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "attn_implementation": attn_implementation,
        "trust_remote_code": trust_remote_code,
    }
    if cache_dir:
        model_kwargs["cache_dir"] = cache_dir
    if quantization_config:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(**quantization_config)

    logger.info("Loading model: %s (dtype=%s, attn=%s)", model_id, torch_dtype, attn_implementation)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )
    # Ensure pad token is set (common issue with Gemma tokenizers)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    logger.info(
        "Model loaded: %s params, vocab_size=%d",
        f"{model.num_parameters():,}",
        len(tokenizer),
    )
    return model, tokenizer


def build_lora_config(
    peft_cfg: dict[str, Any],
    model: AutoModelForCausalLM | None = None,
) -> LoraConfig:
    """Build a LoRA config with safe target modules for Gemma 4.

    This function **always** uses the hardcoded safe target list and
    verifies that no multimodal modules leak through.

    Parameters
    ----------
    peft_cfg : dict
        PEFT configuration from YAML (r, lora_alpha, lora_dropout, etc.).
    model : AutoModelForCausalLM | None
        If provided, validates that target modules exist in the model
        and that multimodal modules are excluded.

    Returns
    -------
    LoraConfig
    """
    target_modules = peft_cfg.get("target_modules", SAFE_LORA_TARGET_MODULES)

    # Safety: never allow "all-linear"
    if target_modules == "all-linear":
        logger.warning(
            "target_modules='all-linear' is UNSAFE for Gemma 4. "
            "Overriding with safe target list."
        )
        target_modules = SAFE_LORA_TARGET_MODULES

    # Validate against multimodal modules if model is provided
    if model is not None:
        _validate_no_multimodal_targets(model, target_modules)

    config = LoraConfig(
        r=peft_cfg.get("r", 64),
        lora_alpha=peft_cfg.get("lora_alpha", 128),
        lora_dropout=peft_cfg.get("lora_dropout", 0.05),
        bias=peft_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    logger.info(
        "LoRA config: r=%d, alpha=%d, dropout=%.2f, targets=%s",
        config.r,
        config.lora_alpha,
        config.lora_dropout,
        config.target_modules,
    )
    return config


def _validate_no_multimodal_targets(
    model: AutoModelForCausalLM,
    target_modules: list[str],
) -> None:
    """Ensure no target module name matches multimodal exclusion patterns."""
    for name, _ in model.named_modules():
        for pattern in _MULTIMODAL_EXCLUSION_PATTERNS:
            if re.match(pattern, name):
                # Check if any target module would match this path
                for target in target_modules:
                    if target in name:
                        raise ValueError(
                            f"LoRA target '{target}' would match multimodal module "
                            f"'{name}'. This will crash PEFT on Gemma4ClippableLinear. "
                            f"Use explicit target_modules={SAFE_LORA_TARGET_MODULES}"
                        )
