# Evaluation Protocol — Protocolo de Avaliação

## Princípios

1. **Reprodutibilidade**: Seeds fixas, configs versionadas, cache de resultados
2. **Rastreabilidade**: Cada predição é rastreável por item
3. **Rigor estatístico**: Bootstrap CI 95%, testes pareados, correção para múltiplas comparações
4. **Separação**: Benchmarks de desenvolvimento vs. benchmarks finais

## Estrutura de Resultados por Item

Cada predição salva no cache contém:

```json
{
  "benchmark": "enem_2024",
  "split": "test",
  "item_id": "q042",
  "model": "gemma4-e4b-cpt-aurora",
  "seed": 42,
  "prompt_hash": "sha256:abc123...",
  "generation_config": {
    "temperature": 0.0,
    "max_new_tokens": 512,
    "do_sample": false
  },
  "think_mode": "off",
  "raw_output": "A alternativa correta é C porque...",
  "stripped_output": "C",
  "parsed_answer": "C",
  "gold_label": "C",
  "correct": true,
  "score": 1.0
}
```

## Benchmarks

### Conjuntos de desenvolvimento (model selection)

Usados durante o processo de adaptação para escolher hiperparâmetros:
- ENEM (versões antigas, 2019-2021)
- ASSIN2 (validation split)
- HateBR (dev split)

### Conjuntos finais (paper/report)

Usados UMA VEZ ao final, para reportar resultados:
- ENEM 2022, 2023, 2024
- BLUEX
- OAB-Bench
- Todas as tarefas da suite completa

### Retenção EN (catastrophic forgetting)

- MMLU (500 amostras)
- HellaSwag (500 amostras)
- ARC-Challenge (500 amostras)

## Modos de Scoring

### 1. Generate + Parse (padrão)

O modelo gera texto livremente e o parser extrai a resposta:

```python
prompt = template.format_prompt(example, think_mode="off")
raw_output = model.generate(prompt)
parsed = task.parse_prediction(raw_output)
correct = (parsed == gold_label)
```

**Vantagens**: Funciona com qualquer modelo, simula uso real.
**Desvantagens**: Sensível a variações de formato, parsing pode falhar.

### 2. Logprob / Choice Scoring (múltipla escolha)

Compara log-probabilidades de cada opção:

```python
for option in ["A", "B", "C", "D"]:
    logprob = model.score(prompt + option)
choice = argmax(logprobs)
```

**Vantagens**: Determinístico, não depende de parsing.
**Desvantagens**: Requer acesso a logprobs, não simula uso real.

**Implementação**: Disponível em `BenchmarkRunner` quando `scoring_mode: "logprob"` na config.

## Normalização PT-BR

Antes de comparar predições com gold labels:

1. Strip whitespace
2. Lowercase (para labels textuais como "entailment")
3. Unicode NFC normalization
4. Remoção de acentos para comparação fuzzy (quando configurado)
5. Mapeamento de variações: "sim"/"yes"/"verdadeiro" → "sim"

## Métricas

| Métrica | Tarefas | Implementação |
|---------|---------|---------------|
| Accuracy | Múltipla escolha | `metrics.py:accuracy` |
| F1 Macro | Classificação | `metrics.py:f1_macro` |
| Pearson | STS (ASSIN2) | `metrics.py:pearson` |
| Entity Micro-F1 | NER (LeNER-Br) | `metrics.py:entity_micro_f1` |
| ROUGE-L | Sumarização | `metrics.py:rouge_l` |
| BERTScore | Geração (opcional) | `metrics.py:bertscore` |
| BoolQ Accuracy | BoolQ-PT | `metrics.py:boolq_accuracy` |

## Inferência Estatística

### Bootstrap CI 95%

```python
from src.eval.bootstrap_ci import bootstrap_ci

ci = bootstrap_ci(
    predictions, gold_labels,
    metric_fn=compute_accuracy,
    n_bootstrap=1000,
    confidence_level=0.95,
    seed=42,
)
# ci["accuracy"] = {"mean": 0.78, "ci_lower": 0.73, "ci_upper": 0.82}
```

### Testes Pareados

```python
from src.eval.stats_tests import paired_permutation_test

result = paired_permutation_test(scores_model_a, scores_model_b)
# result = {"p_value": 0.003, "significant": True, ...}
```

### Correção para Múltiplas Comparações

Método Holm (step-down) aplicado quando comparando múltiplos modelos:

```python
from src.eval.stats_tests import multiple_comparison_correction

corrected_pvalues = multiple_comparison_correction(
    [0.01, 0.03, 0.08], method="holm"
)
```

## Cache e Reprodutibilidade

- Resultados cacheados em `outputs/eval_cache/` por hash de (model + benchmark + seed + think_mode)
- Cache invalidado se config de geração mudar
- Para forçar re-avaliação: deletar cache ou mudar seed

## Limites Conhecidos

1. **Parsing frágil**: Modelos podem gerar formatos inesperados. Testes em `tests/test_parsing.py` cobrem os casos comuns.
2. **Benchmarks em PT-BR**: Alguns datasets não têm splits oficiais de test; usamos validation quando disponível.
3. **Single-seed**: Avaliação com seed única, mitigada por bootstrap CI.
4. **Saturation**: Benchmarks de múltipla escolha podem saturar em modelos muito grandes.
