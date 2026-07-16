"""
Fábrica unificada para métodos PEFT (Parameter-Efficient Fine-Tuning).

Suporta criação e aplicação de configurações LoRA, DoRA, QLoRA,
Prefix Tuning e Adapter para modelos Gemma 4.
"""

from typing import Any

from peft import (
    IA3Config,
    LoraConfig,
    PrefixTuningConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# Módulos-alvo padrão para arquitetura Gemma 4
GEMMA4_DEFAULT_TARGET_MODULES: list[str] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Métodos PEFT suportados
SUPPORTED_METHODS: list[str] = ["lora", "dora", "qlora", "prefix_tuning", "adapter"]


def create_peft_config(method: str, config_dict: dict[str, Any] | None = None):
    """
    Cria a configuração PEFT apropriada para o método especificado.

    Args:
        method: Método PEFT a ser utilizado. Valores aceitos:
            "lora", "dora", "qlora", "prefix_tuning", "adapter".
        config_dict: Dicionário com parâmetros de configuração.
            Se None, utiliza valores padrão.

    Returns:
        Objeto de configuração PEFT correspondente ao método.

    Raises:
        ValueError: Se o método não for suportado.
    """
    if config_dict is None:
        config_dict = {}

    method = method.lower().strip()

    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Método '{method}' não suportado. "
            f"Métodos disponíveis: {SUPPORTED_METHODS}"
        )

    if method == "lora":
        return _create_lora_config(config_dict, use_dora=False)

    elif method == "dora":
        return _create_lora_config(config_dict, use_dora=True)

    elif method == "qlora":
        # QLoRA usa LoraConfig padrão; a quantização é tratada no carregamento do modelo
        return _create_lora_config(config_dict, use_dora=False)

    elif method == "prefix_tuning":
        return _create_prefix_tuning_config(config_dict)

    elif method == "adapter":
        return _create_adapter_config(config_dict)


def _create_lora_config(
    config_dict: dict[str, Any], use_dora: bool = False
) -> LoraConfig:
    """
    Cria configuração LoRA/DoRA/QLoRA.

    Args:
        config_dict: Parâmetros de configuração.
        use_dora: Se True, ativa o modo DoRA (Weight-Decomposed Low-Rank Adaptation).

    Returns:
        LoraConfig configurado.
    """
    defaults = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": GEMMA4_DEFAULT_TARGET_MODULES,
    }
    defaults.update(config_dict)
    defaults["use_dora"] = use_dora

    return LoraConfig(**defaults)


def _create_prefix_tuning_config(config_dict: dict[str, Any]) -> PrefixTuningConfig:
    """
    Cria configuração de Prefix Tuning.

    Args:
        config_dict: Parâmetros de configuração.

    Returns:
        PrefixTuningConfig configurado.
    """
    defaults = {
        "num_virtual_tokens": 20,
        "task_type": "CAUSAL_LM",
    }
    defaults.update(config_dict)

    return PrefixTuningConfig(**defaults)


def _create_adapter_config(config_dict: dict[str, Any]) -> IA3Config:
    """
    Cria configuração de Adapter usando IA3 (Infused Adapter by Inhibiting
    and Amplifying Inner Activations).

    Args:
        config_dict: Parâmetros de configuração.

    Returns:
        IA3Config configurado.
    """
    defaults = {
        "target_modules": GEMMA4_DEFAULT_TARGET_MODULES,
        "feedforward_modules": ["gate_proj", "up_proj", "down_proj"],
        "task_type": "CAUSAL_LM",
    }
    defaults.update(config_dict)

    return IA3Config(**defaults)


def apply_peft(model, method: str, config_dict: dict[str, Any] | None = None):
    """
    Aplica PEFT ao modelo, incluindo preparação para treinamento quantizado se necessário.

    Args:
        model: Modelo base do HuggingFace (PreTrainedModel).
        method: Método PEFT a ser aplicado.
        config_dict: Dicionário com parâmetros de configuração PEFT.

    Returns:
        Modelo com PEFT aplicado (PeftModel).
    """
    # Prepara modelo para treinamento com quantização (kbit) se detectado
    if _is_quantized(model):
        model = prepare_model_for_kbit_training(model)

    # Cria configuração e aplica PEFT
    peft_config = create_peft_config(method, config_dict)
    peft_model = get_peft_model(model, peft_config)

    return peft_model


def _is_quantized(model) -> bool:
    """
    Verifica se o modelo está quantizado (carregado com bitsandbytes).

    Args:
        model: Modelo a ser verificado.

    Returns:
        True se o modelo estiver quantizado, False caso contrário.
    """
    if hasattr(model, "is_quantized"):
        return model.is_quantized

    if hasattr(model, "config") and hasattr(model.config, "quantization_config"):
        return model.config.quantization_config is not None

    return False


def get_trainable_param_summary(model) -> dict[str, Any]:
    """
    Retorna resumo dos parâmetros treináveis do modelo.

    Args:
        model: Modelo (pode ser PeftModel ou PreTrainedModel).

    Returns:
        Dicionário com:
            - total_params: Total de parâmetros no modelo.
            - trainable_params: Número de parâmetros treináveis.
            - trainable_pct: Percentual de parâmetros treináveis.
    """
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    trainable_pct = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_pct": round(trainable_pct, 4),
    }
