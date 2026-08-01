# Gemma 4 Compliance — Protocolo de Conformidade

## Visão Geral

Este documento descreve como o pipeline lida com as especificidades do Gemma 4, garantindo conformidade com a API oficial.

> **Nota de verificação**: os tokens especiais e o formato de chat descritos
> abaixo foram **verificados ao vivo** contra o tokenizer real
> (`google/gemma-4-E2B-it`, chat template publicado 2026-07-09), não
> assumidos a partir de documentação externa. O Gemma 4 usa um formato de
> chat **diferente** do Gemma 2/3 — ver "Tokens especiais reais" abaixo.
> Isso corrigiu um bug real neste repositório: versões anteriores deste
> documento e do código assumiam os marcadores do Gemma 2/3
> (`<start_of_turn>`/`<end_of_turn>`/`<think>`), que no vocabulário do
> Gemma 4 nem sequer são tokens especiais.

## Tokens especiais reais do Gemma 4

Verificado com `AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")`:

| Token | ID | Uso |
|-------|-----|-----|
| `<bos>` | 2 | Início de sequência |
| `<eos>` | 1 | Fim de sequência |
| `<pad>` | 0 | Padding (Gemma 4 tem um `<pad>` dedicado — diferente do `<eos>`) |
| `<\|turn>` | 105 | Abre um turno: `<\|turn>role\n` |
| `<turn\|>` | 106 | Fecha um turno |
| `<\|think\|>` | 98 | Ativa o modo de raciocínio (injetado num turno de sistema) |
| `<\|channel>` | 100 (prefixo) | Abre um canal — usado como `<\|channel>thought\n` |
| `<channel\|>` | 101 | Fecha um canal |

Por comparação, `<start_of_turn>`/`<end_of_turn>`/`<think>`/`</think>`
(formato do Gemma 2/3) **fragmentam em 7 subtokens comuns cada** no Gemma 4 —
ou seja, não são tratados como marcadores de controle pelo modelo.

### Formato de um turno

```
<|turn>user
Qual a capital do Brasil?<turn|>
<|turn>model
```

### Formato com thinking ativado (`enable_thinking=True`)

```
<bos><|turn>system
<|think|>
<turn|>
<|turn>user
Qual a capital do Brasil?<turn|>
<|turn>model
```

O conteúdo de raciocínio já gerado (em texto já produzido pelo modelo, ou em
histórico multi-turn) é delimitado por `<|channel>thought\n ... \n<channel|>`
— não por `<think>...</think>`.

**Não há role `system` dedicado da forma como o Gemma 2/3 tratava** — o
turno de sistema só aparece quando `enable_thinking=True`, há `tools`, ou a
primeira mensagem já é `system`/`developer`.

## Camada de Abstração de Prompts

### Hierarquia de formatação

1. **Preferencial**: `tokenizer.apply_chat_template()` — usa o template real
   do tokenizer carregado, então cada família de modelo (Gemma 4, Gemma 3/Gaia,
   ChatML/Tucano, ...) recebe automaticamente seu próprio formato correto,
   sem precisar hardcodar por família.
2. **Fallback manual**: usado apenas se o tokenizer não tiver
   `chat_template` (ex.: um checkpoint CPT/base sem instruction tuning,
   verificado: `google/gemma-4-E2B` — o modelo BASE — não tem
   `chat_template` definido, só a variante `-it`) ou se `apply_chat_template`
   levantar exceção. Usa os tokens reais `<|turn>`/`<turn|>` listados acima.
3. **Baseline (não-chat)**: Formato few-shot plain (para Sabia-7B e
   modelos CPT-only avaliados com `is_chat_model: false`).

### Classes responsáveis

| Classe | Localização | Uso |
|--------|------------|-----|
| `Gemma4PromptBuilder` | `src/data/prompt_builders.py` | Treino e inferência Gemma 4 |
| `PromptBuilder` | `src/eval/prompt_templates.py` | Avaliação (todos modelos) — ver `benchmark_runner.py`'s `run_all`, que constrói um `PromptBuilder` por modelo usando o `is_chat_model` declarado em `configs/eval/benchmarks.yaml`'s `models_to_evaluate` |
| `BaselinePromptBuilder` | `src/data/prompt_builders.py` | Modelos não-chat |

### Regra: Nunca usar tokens hardcoded diretamente

```python
# ERRADO — Não faça isso:
prompt = f"<|turn>user\n{question}<turn|>\n<|turn>model\n"

# CORRETO — Use o builder (que prefere apply_chat_template):
builder = PromptBuilder(tokenizer=tokenizer, is_chat_model=True)
prompt = builder.format_prompt(system_msg=None, user_msg=question, think_mode="off")
```

## Modos de Pensamento (Thinking)

### Modos suportados

| Modo | Descrição | Uso |
|------|-----------|-----|
| `"off"` | Sem pensamento (padrão) | Avaliação padrão |
| `"on"` | Ativa raciocínio via `enable_thinking=True` em `apply_chat_template` | Tarefas complexas (ENEM, OAB — hipótese H5) |
| `"budget"` (apenas `Gemma4PromptBuilder`/fallback manual) | Canal de pensamento vazio `<\|channel>thought\n<channel\|>` | Modelos que esperam o formato mas sem orçamento de raciocínio |

`src.eval.prompt_templates.PromptBuilder.format_prompt` passa
`enable_thinking=(think_mode == "on")` diretamente para
`tokenizer.apply_chat_template()` (com fallback para uma chamada sem esse
kwarg via `try`/`except TypeError`, para tokenizers de outras famílias que
não o aceitam) — **não** anexa uma string `"<think>\n"` manualmente após o
prompt renderizado, que era o comportamento antigo (incorreto para o Gemma
4 real).

### Protocolo de multi-turn com pensamento

**Regra crítica**: Ao construir prompts multi-turn, SEMPRE remover pensamentos de turnos anteriores antes de incluí-los no histórico.

```python
# Turno 1: modelo gera com pensamento
raw_output = "<|channel>thought\nCalculando: 2+2=4\n<channel|>A resposta é 4."

# Antes de incluir no histórico do turno 2:
cleaned = strip_thought(raw_output)  # "A resposta é 4."
```

`strip_thought`/`extract_thought` (em `src/eval/prompt_templates.py` e
`src/data/prompt_builders.py`) reconhecem **ambos** os formatos — o real do
Gemma 4 (`<|channel>thought...<channel|>`) e o legado `<think>...</think>`
(usado por algumas outras famílias de chat template) — já que
`benchmark_runner.py` avalia múltiplos modelos de famílias diferentes e não
sabe a priori qual convenção cada um usa.

**Exceção**: Tool/function calling pode preservar pensamento se houver lógica explícita para isso (o template real do Gemma 4 tem essa lógica embutida via `preserve_thinking`).

### API oficial: `enable_thinking`

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it")
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,  # API oficial — injeta <|think|> corretamente
)
```

## Modo Text-Only

### Motivação

Gemma 4 é multimodal (visão + áudio + vídeo + texto). Para CPT em texto puro, os módulos multimodais devem ser congelados.

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
```

`text_only_mode: true` alone triggers `_freeze_multimodal_modules` (`src/utils/hf_utils.py`),
which freezes all vision/multimodal parameters by name pattern (`vision_tower`,
`multi_modal_projector`, etc. — see the function for the full list). There are no separate
`freeze_vision_encoder` / `freeze_multi_modal_projector` toggles — earlier drafts of these
configs implied they existed, but neither was ever read by any code path; both have been
removed.

## Diferenciação de caminhos

| Caminho | Template | Think | Freeze Vision |
|---------|----------|-------|---------------|
| CPT (base) | Nenhum (next-token, sem chat markup) | N/A | Sim |
| SFT (IT) | `format_gemma4_chat` (tokens reais `<\|turn>`/`<turn\|>`) | Configurável (`sft.use_think_tokens`) | Sim |
| Eval (IT/merge) | `apply_chat_template` do tokenizer real do modelo (`is_chat_model: true`) | off/on via `enable_thinking` | N/A |
| Eval (base/CPT/Sabia) | Few-shot plain (`is_chat_model: false`) | N/A | N/A |

**Nota sobre o residual merge** (`src/train/residual_merge.py`): o modelo
resultante do merge é salvo com o tokenizer do modelo **instruct**, não do
checkpoint CPT/base — o checkpoint CPT (especialmente se veio de LoRA) herda
o tokenizer do modelo base, que **não tem** `chat_template` definido; como o
merge tem por objetivo produzir um modelo que segue instruções, ele precisa
do tokenizer que carrega o template real.

## Testes de conformidade

```bash
pytest tests/test_gemma4_compliance.py -v
pytest tests/test_prompt_templates.py -v
```

## Limitações conhecidas

1. **AutoProcessor multimodal**: O pipeline usa `AutoTokenizer` por padrão (todo o treino/avaliação aqui é text-only). Para inferência multimodal (imagem/áudio/vídeo) seria necessário `AutoProcessor`.

2. **`enable_thinking` kwarg**: parte do template do PRÓPRIO tokenizer Gemma 4 (não uma API genérica do `transformers`) — funciona porque o template Jinja do tokenizer lê essa variável. Tokenizers de outras famílias que não implementam essa variável no template recebem um fallback (`PromptBuilder` tenta sem o kwarg via `except TypeError`).

3. **Tokens especiais podem mudar**: um refresh do template/tokenizer do Gemma 4 foi publicado em 2026-07-15 (após a versão verificada aqui, de 2026-07-09) — sempre reverificar `tokenizer.chat_template` ao atualizar para um checkpoint mais novo, em vez de assumir que os marcadores documentados aqui permanecem estáveis. `apply_chat_template` sempre reflete o template atual automaticamente; o fallback manual (`_format_gemma4_manual`, `chat_template_fallback` nos YAMLs de modelo) precisaria ser atualizado manualmente se os tokens mudarem.
