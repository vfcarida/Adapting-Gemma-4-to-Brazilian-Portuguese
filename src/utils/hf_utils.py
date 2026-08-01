"""Utilitários de carregamento de modelos e tokenizers HuggingFace.

Este módulo centraliza toda a lógica de carregamento de modelos Gemma 4,
incluindo:
- Carregamento de tokenizer com configurações adequadas
- Carregamento para treino (com/sem quantização)
- Carregamento para inferência
- Freeze de módulos multimodais (modo text-only)
- Medição de tamanho do modelo
- Suporte a apply_chat_template via tokenizer

IMPORTANTE: Gemma 4 é multimodal. Para CPT/SFT em texto puro, devemos
congelar os encoders visuais e o projetor multimodal.
"""

import os
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_tokenizer(model_id: str, **kwargs) -> AutoTokenizer:
    """Carrega tokenizer com configurações adequadas para Gemma 4.

    Usa apply_chat_template quando disponível no tokenizer carregado.

    Args:
        model_id: ID do modelo no HF Hub ou caminho local.

    Returns:
        AutoTokenizer configurado.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        padding_side="right",
        trust_remote_code=True,
        **kwargs,
    )
    if tokenizer.pad_token is None:
        # Gemma tokenizers ship a dedicated <pad> token distinct from <eos>,
        # so this fallback should not trigger for Gemma models. It exists for
        # other base models that lack a pad token. Aliasing pad to eos here
        # would make DataCollatorForLanguageModeling mask every EOS position
        # (labels[labels == pad_token_id] = -100), which is why the CPT
        # trainer never relies on this collator to mask padding (see
        # src/train/cpt_trainer.py's PackedSequenceCollator).
        logger.warning(
            f"{model_id}: tokenizer has no pad_token; aliasing to eos_token. "
            "This is safe for inference/eval padding, but never pass "
            "DataCollatorForLanguageModeling(mlm=False) over pre-packed CPT "
            "sequences from this tokenizer without checking label masking."
        )
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_for_training(
    model_id: str,
    use_lora: bool = False,
    quantize: bool = False,
    model_config: dict[str, Any] | None = None,
) -> AutoModelForCausalLM:
    """Carrega modelo para treinamento com opções de quantização e text-only.

    Se o model_config indica text_only_mode=true, congela automaticamente
    os módulos de visão (vision_encoder, multi_modal_projector).

    Args:
        model_id: ID do modelo ou caminho local.
        use_lora: Se True, modelo será usado com PEFT (não move para GPU diretamente).
        quantize: Se True, aplica quantização 4-bit (BnB).
        model_config: Dict com configurações do modelo.

    Returns:
        Modelo carregado e pronto para treino.
    """
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
    }

    if model_config:
        model_cfg = model_config.get("model", {})
        if "attn_implementation" in model_cfg:
            attn_impl = model_cfg["attn_implementation"]
            if attn_impl == "flash_attention_2":
                try:
                    import flash_attn  # noqa: F401

                    kwargs["attn_implementation"] = attn_impl
                except ImportError:
                    logger.warning(
                        "flash_attention_2 requested but flash-attn not installed. "
                        "Falling back to sdpa (PyTorch native)."
                    )
                    kwargs["attn_implementation"] = "sdpa"
            else:
                kwargs["attn_implementation"] = attn_impl

    if quantize:
        quant_cfg = model_config.get("quantization", {}) if model_config else {}
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=quant_cfg.get("load_in_4bit", True),
            bnb_4bit_compute_dtype=getattr(
                torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
            ),
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        )
        # bitsandbytes-quantized models must be placed on a device at load
        # time (they cannot be `.to(device)`-ed afterwards). Without
        # DeepSpeed, map the whole model onto the single visible GPU (or the
        # process-local GPU under `accelerate launch` with multiple
        # processes) — the standard pattern for QLoRA training on Colab/
        # single-GPU boxes. Under DeepSpeed, `device_map` must stay unset so
        # ZeRO can shard the model itself.
        if torch.cuda.is_available() and not os.environ.get("ACCELERATE_USE_DEEPSPEED"):
            try:
                from accelerate import PartialState

                kwargs["device_map"] = {"": PartialState().local_process_index}
            except ImportError:
                kwargs["device_map"] = {"": 0}

    logger.info(f"Carregando modelo: {model_id} (quantize={quantize})")
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    if not quantize and not use_lora:
        model = model.to(torch.bfloat16)

    # Modo text-only: congela componentes multimodais
    if model_config and model_config.get("model", {}).get("text_only_mode", False):
        _freeze_multimodal_modules(model)

    return model


def _freeze_multimodal_modules(model: torch.nn.Module) -> None:
    """Congela módulos multimodais para treino text-only.

    Gemma 4 pode incluir vision_tower, multi_modal_projector, etc.
    Em CPT/SFT textual, estes devem ficar congelados para:
    1. Reduzir uso de memória (gradientes não computados)
    2. Evitar corrupção da capacidade visual
    3. Acelerar o treino

    Args:
        model: Modelo carregado.
    """
    frozen_count = 0
    # Padrões de nomes de módulos multimodais em modelos Gemma 4
    multimodal_patterns = [
        "vision_tower",
        "vision_encoder",
        "visual",
        "multi_modal_projector",
        "mm_projector",
        "image_encoder",
        "img_",
        "pixel",
    ]

    for name, param in model.named_parameters():
        if any(pattern in name.lower() for pattern in multimodal_patterns):
            param.requires_grad = False
            frozen_count += 1

    if frozen_count > 0:
        logger.info(f"Modo text-only: {frozen_count} parâmetros multimodais congelados")
    else:
        logger.info("Nenhum módulo multimodal encontrado (modelo pode ser text-only nativo)")


def load_model_for_inference(
    model_id: str,
    device: str = "auto",
    quantize: bool = False,
) -> AutoModelForCausalLM:
    """Carrega modelo para inferência (eval mode, device_map auto).

    Args:
        model_id: ID do modelo ou caminho local.
        device: Mapeamento de dispositivo ("auto" distribui automaticamente).
        quantize: Se True, aplica quantização 4-bit.

    Returns:
        Modelo em modo avaliação.
    """
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
        "device_map": device,
    }

    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Calcula tamanho do modelo em MB (apenas parâmetros)."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    return param_size / (1024 * 1024)


def get_trainable_params_info(model: torch.nn.Module) -> dict[str, Any]:
    """Retorna informações sobre parâmetros treináveis vs congelados.

    Útil para verificar que PEFT e freeze multimodal estão corretos.

    Returns:
        Dict com total_params, trainable_params, frozen_params, trainable_pct.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": frozen,
        "trainable_pct": 100.0 * trainable / max(total, 1),
    }


def supports_chat_template(tokenizer) -> bool:
    """Verifica se o tokenizer suporta apply_chat_template.

    Args:
        tokenizer: Tokenizer carregado.

    Returns:
        True se chat_template está disponível.
    """
    return hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None


def estimate_vram_gb(
    model_params_b: float,
    seq_length: int = 8192,
    batch_size: int = 1,
    grad_accum_steps: int = 16,
    use_lora: bool = False,
    lora_r: int = 64,
    gradient_checkpointing: bool = True,
    optimizer: str = "adamw",
    dtype_bytes: int = 2,
) -> dict[str, float]:
    """Estimate GPU VRAM required for training.

    Uses heuristic formulas based on model size, sequence length, and
    training configuration. Intended as a preflight check before launching
    expensive GCP instances.

    Estimates are approximate (±20%) but sufficient to select GPU type:
    - < 16 GB: T4 / L4
    - < 24 GB: A10G / RTX 4090
    - < 40 GB: A100-40GB
    - < 80 GB: A100-80GB / H100
    - > 80 GB: Multi-GPU required

    Args:
        model_params_b: Model size in billions of parameters.
        seq_length: Maximum sequence length.
        batch_size: Per-device batch size.
        grad_accum_steps: Gradient accumulation steps (does not affect VRAM).
        use_lora: Whether using LoRA (reduces optimizer states).
        lora_r: LoRA rank (affects trainable param count).
        gradient_checkpointing: Whether gradient checkpointing is enabled.
        optimizer: Optimizer name ("adamw" uses 8 bytes/param for states).
        dtype_bytes: Bytes per parameter (2 for bf16/fp16, 4 for fp32).

    Returns:
        Dict with component-wise VRAM breakdown and total estimate in GB.
    """
    params = model_params_b * 1e9

    # Model weights
    model_vram = params * dtype_bytes / 1e9

    # Trainable parameters (LoRA reduces this significantly)
    if use_lora:
        # LoRA trainable params ≈ 2 * r * hidden_dim * num_layers * num_targets
        # Rough estimate: ~2-4% of total params for typical configs
        trainable_ratio = min(0.04, (2 * lora_r * 8) / (params / 1e6))
        trainable_params = params * trainable_ratio
    else:
        trainable_params = params

    # Optimizer states (AdamW: 2 states × dtype per trainable param)
    if optimizer == "adamw":
        # Adam stores m and v in fp32 regardless of model dtype
        optimizer_vram = trainable_params * 8 / 1e9  # 4 bytes × 2 states
    else:
        optimizer_vram = trainable_params * 4 / 1e9

    # Gradients (same dtype as model for trainable params)
    gradients_vram = trainable_params * dtype_bytes / 1e9

    # Activations (highly dependent on seq_length and batch_size)
    # Use known model architecture parameters by size bracket
    import math

    if model_params_b <= 1.5:
        hidden_dim, num_layers = 2048, 22
    elif model_params_b <= 3:
        hidden_dim, num_layers = 2560, 28
    elif model_params_b <= 5:
        hidden_dim, num_layers = 3072, 34
    elif model_params_b <= 9:
        hidden_dim, num_layers = 4096, 32
    elif model_params_b <= 15:
        hidden_dim, num_layers = 5120, 40
    elif model_params_b <= 35:
        hidden_dim, num_layers = 6656, 60
    else:
        hidden_dim, num_layers = 8192, 80

    # Per-layer activation memory: batch * seq * hidden * dtype * factor
    # Factor accounts for attention QKV, intermediate states (~4x hidden)
    activation_factor = 4
    activations_per_layer = batch_size * seq_length * hidden_dim * dtype_bytes * activation_factor
    if gradient_checkpointing:
        # Only store activations at checkpoint boundaries (~sqrt(layers))
        active_layers = int(math.sqrt(num_layers)) + 1
    else:
        active_layers = num_layers

    activations_vram = activations_per_layer * active_layers / 1e9

    # CUDA kernels and fragmentation overhead (~5-10%)
    overhead_ratio = 0.08
    subtotal = model_vram + optimizer_vram + gradients_vram + activations_vram
    overhead_vram = subtotal * overhead_ratio

    total = subtotal + overhead_vram

    return {
        "model_weights_gb": round(model_vram, 2),
        "optimizer_states_gb": round(optimizer_vram, 2),
        "gradients_gb": round(gradients_vram, 2),
        "activations_gb": round(activations_vram, 2),
        "overhead_gb": round(overhead_vram, 2),
        "total_estimated_gb": round(total, 2),
        "recommended_gpu": _recommend_gpu(total),
    }


def _recommend_gpu(total_vram_gb: float) -> str:
    """Recommend GPU type based on estimated VRAM requirement."""
    if total_vram_gb <= 16:
        return "T4 (16GB) or L4 (24GB)"
    elif total_vram_gb <= 24:
        return "L4 (24GB) or A10G (24GB)"
    elif total_vram_gb <= 40:
        return "A100-40GB"
    elif total_vram_gb <= 80:
        return "A100-80GB or H100-80GB"
    else:
        return f"Multi-GPU required (~{int(total_vram_gb / 80) + 1}x A100-80GB)"
