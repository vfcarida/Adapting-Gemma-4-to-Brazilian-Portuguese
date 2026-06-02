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
        "torch_dtype": torch.bfloat16,
    }

    if model_config:
        model_cfg = model_config.get("model", {})
        if "attn_implementation" in model_cfg:
            kwargs["attn_implementation"] = model_cfg["attn_implementation"]

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
        "vision_tower", "vision_encoder", "visual",
        "multi_modal_projector", "mm_projector",
        "image_encoder", "img_", "pixel",
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
        "torch_dtype": torch.bfloat16,
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
