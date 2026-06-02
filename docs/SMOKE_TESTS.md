# Smoke Tests — Testes de Fumaça

## O que são

Smoke tests validam o pipeline end-to-end em CPU sem dados ou modelos reais. Garantem que toda a infraestrutura funciona antes de comprometer tempo de GPU.

## Execução

```bash
# Via CLI
gemma4pt smoke

# Direto
python -m tests.smoke_test

# Com output detalhado
python -m tests.smoke_test --verbose
```

## Cobertura

O smoke test valida 13 componentes:

| # | Componente | O que testa |
|---|-----------|-------------|
| 1 | config_loading | YAML parse, merge, flatten, nested references |
| 2 | seed_reproducibility | numpy/torch seeds produzem mesmos resultados |
| 3 | prompt_builders | Gemma4PromptBuilder, BaselinePromptBuilder, think modes |
| 4 | prompt_templates | TaskPromptTemplate, TASK_INSTRUCTIONS, formatters |
| 5 | metrics_computation | accuracy, macro_f1, empty-list guard |
| 6 | bootstrap_ci | CI bounds, paired test, determinismo |
| 7 | stats_tests | Permutation test, effect size, Holm correction |
| 8 | checkpointing | save/load state, find_latest_checkpoint |
| 9 | contamination_checks | normalize, hash, exact match, MinHash |
| 10 | report_builder | CSV/Markdown generation, group averages |
| 11 | residual_merge_logic | Task arithmetic math (numpy) |
| 12 | config_override | Deep merge preserves unmodified keys |
| 13 | preflight | Environment validation framework |

## Tempo esperado

- CPU: ~2 segundos
- Sem dependência de rede, GPU, ou datasets externos

## Quando rodar

- Antes de qualquer commit
- Após instalar ou atualizar dependências
- Antes de iniciar treino em GPU
- Em CI (job `test-smoke`)

## Adicionando novos smoke tests

Edite `tests/smoke_test.py` e adicione um novo `check()`:

```python
def test_novo_componente():
    from src.modulo import funcao
    resultado = funcao(input_sintetico)
    assert condicao_esperada

check("novo_componente", test_novo_componente)
```
