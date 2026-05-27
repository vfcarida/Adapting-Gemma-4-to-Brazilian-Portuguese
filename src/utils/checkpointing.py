"""
src/utils/checkpointing.py
──────────────────────────
Checkpoint save / load wrappers with LoRA adapter merge utilities.
Enforces ``safetensors`` format by default.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def save_checkpoint(
    model: PreTrainedModel | PeftModel,
    tokenizer: AutoTokenizer,
    output_dir: str | Path,
    safe_serialization: bool = True,
) -> Path:
    """Save model + tokenizer to *output_dir*.

    If the model is a ``PeftModel`` it saves **only the adapter
    weights** (not the full base model), which is the standard PEFT
    behaviour.

    Parameters
    ----------
    model : PreTrainedModel | PeftModel
        The model (or LoRA-wrapped model) to persist.
    tokenizer : AutoTokenizer
        Associated tokenizer.
    output_dir : str | Path
        Destination directory.
    safe_serialization : bool
        Use ``safetensors`` format (recommended).

    Returns
    -------
    Path
        The resolved output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    tokenizer.save_pretrained(output_dir)
    logger.info("Checkpoint saved → %s", output_dir)
    return output_dir


def load_checkpoint(
    checkpoint_path: str | Path,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
) -> tuple[PreTrainedModel, AutoTokenizer]:
    """Load a full model + tokenizer from a local checkpoint.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to the saved checkpoint directory.
    device_map : str
        Device placement strategy.
    torch_dtype : torch.dtype
        Data type for model weights.
    trust_remote_code : bool
        Whether to trust remote code in the checkpoint.

    Returns
    -------
    tuple[PreTrainedModel, AutoTokenizer]
        The loaded model and tokenizer.
    """
    checkpoint_path = Path(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=trust_remote_code,
    )
    logger.info("Checkpoint loaded ← %s", checkpoint_path)
    return model, tokenizer


def merge_lora_weights(
    base_model_id: str,
    adapter_path: str | Path,
    output_dir: str | Path,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    safe_serialization: bool = True,
) -> Path:
    """Merge LoRA adapter weights back into the base model.

    This produces a standalone model without any PEFT dependency at
    inference time.

    Parameters
    ----------
    base_model_id : str
        HuggingFace model ID or local path of the base model.
    adapter_path : str | Path
        Path to the saved LoRA adapter directory.
    output_dir : str | Path
        Where to save the merged model.
    torch_dtype : torch.dtype
        Data type for weight loading.
    device_map : str
        Device placement strategy.
    safe_serialization : bool
        Save using safetensors format.

    Returns
    -------
    Path
        The output directory of the merged model.
    """
    output_dir = Path(output_dir)
    logger.info("Loading base model: %s", base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    logger.info("Loading LoRA adapter: %s", adapter_path)
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    logger.info("Merging adapter weights …")
    model = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=safe_serialization)

    # Copy tokenizer from adapter dir (it should have been saved there)
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    logger.info("Merged model saved → %s", output_dir)
    return output_dir
