# Gemma 4 Compliance — Protocolo de Conformidade

## Visão Geral

Este documento descreve como o pipeline lida com as especificidades do Gemma 4, garantindo conformidade com a API oficial.

## Camada de Abstração de Prompts

### Hierarquia de formatação

1. **Preferencial**: `tokenizer.apply_chat_template()` / `processor.apply_chat_template()`
2. **Fallback manual**: Template com tokens `<start_of_turn>` / `<end_of_turn>`
3. **Baseline (não-chat)**: Formato few-shot plain (para Sabia-7B e similares)

### Classes responsáveis

| Classe | Localização | Uso |
|--------|------------|-----|
| `Gemma4PromptBuilder` | `src/data/prompt_builders.py` | Treino e inferência Gemma 4 |
| `PromptBuilder` | `src/eval/prompt_templates.py` | Avaliação (todos modelos) |
| `BaselinePromptBuilder` | `src/data/prompt_builders.py` | Modelos não-chat |

### Regra: Nunca usar tokens hardcoded diretamente

```python
# ERRADO — Não faça isso:
prompt = f"<start_of_turn>user\n{question}<end_of_turn>\n<start_of_turn>model\n"

# CORRETO — Use o builder:
builder = Gemma4PromptBuilder(tokenizer)
prompt = builder.format_for_inference(
    [{"role": "user", "content": question}],
    think_mode="off",
)
```

## Modos de Pensamento (Thinking)

### Modos suportados

| Modo | Descrição | Uso |
|------|-----------|-----|
| `"off"` | Sem pensamento (padrão) | Avaliação padrão |
| `"on"` | Ativa raciocínio com `<think>` | Tarefas complexas |
| `"budget"` | Canal vazio `<think></think>` | Modelos que esperam o formato |

### Protocolo de multi-turn com pensamento

**Regra crítica**: Ao construir prompts multi-turn, SEMPRE remover pensamentos de turnos anteriores antes de incluí-los no histórico.

```python
# Turno 1: modelo gera com pensamento
raw_output = "<think>Calculando: 2+2=4</think>A resposta é 4."

# Antes de incluir no histórico do turno 2:
cleaned = builder.strip_thought(raw_output)  # "A resposta é 4."

# Turno 2: usar output limpo no histórico
messages = [
    {"role": "user", "content": "Quanto é 2+2?"},
    {"role": "model", "content": cleaned},  # SEM pensamento
    {"role": "user", "content": "E 3+3?"},
]
```

**Exceção**: Tool/function calling pode preservar pensamento se houver lógica explícita para isso.

### API oficial: `enable_thinking`

Para modelos IT com suporte nativo a thinking via `AutoProcessor`:

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("google/gemma-4-27b-it")
# enable_thinking=True é o equivalente oficial ao think_mode="on"
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
    enable_thinking=True,  # API oficial
)
```

O `Gemma4PromptBuilder` abstrai isso: quando `enable_thinking` estiver disponível no processor, ele é usado preferencialmente.

## Modo Text-Only

### Motivação

Gemma 4 é multimodal (visão + texto). Para CPT em texto puro, os módulos visuais devem ser congelados.

### Implementação

Em `src/utils/hf_utils.py`:

```python
def _freeze_multimodal_modules(model):
    """Congela módulos multimodais detectados por nome."""
    patterns = ["vision_tower", "multi_modal_projector", "pixel", "image_encoder"]
    frozen = 0
    for name, param in model.named_parameters():
        if any(p in name for p in patterns):
            param.requires_grad = False
            frozen += 1
    return frozen
```

### Configuração

```yaml
# configs/model/gemma4_e4b.yaml
model:
  text_only_mode: true
  freeze_vision_encoder: true
```

## Diferenciação de caminhos

| Caminho | Template | Think | Freeze Vision |
|---------|----------|-------|---------------|
| CPT (base) | Nenhum (next-token) | N/A | Sim |
| SFT (IT) | apply_chat_template | Configurável | Sim |
| Eval (IT) | apply_chat_template | off/on/budget | N/A |
| Eval (base/Sabia) | Few-shot plain | N/A | N/A |
| Inferência multimodal | processor | Configurável | Não |

## Testes de conformidade

```bash
# Rodar testes de conformidade Gemma 4
pytest tests/test_gemma4_compliance.py -v

# Cobertura de casos:
# - 5 testes: formato sem thinking
# - 4 testes: formato com thinking
# - 5 testes: multi-turn com pensamento
# - 4 testes: parsing de respostas
# - 4 testes: diferenciação de modos
# - 2 testes: few-shot
# - 5 testes: edge cases
```

## Limitações conhecidas

1. **AutoProcessor**: O pipeline usa `AutoTokenizer` por padrão. Para modelos com `AutoProcessor` (multimodal), é necessário carregar via processor quando imagens estão envolvidas.

2. **enable_thinking kwarg**: Disponível apenas em transformers >= 4.45 com modelos que suportam nativamente. O fallback manual funciona em versões anteriores.

3. **Tokens especiais**: Se Google atualizar os tokens de controle do Gemma 4 em versões futuras, `apply_chat_template` se adapta automaticamente. O fallback manual precisaria ser atualizado.
