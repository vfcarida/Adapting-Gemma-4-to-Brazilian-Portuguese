# Pre-Train Readiness Report

**Data:** 2026-06-01 (snapshot histórico — ver nota abaixo)
**Status:** PRONTO (pendente GPU)

> **Nota de atualização**: desde este snapshot, uma revisão completa de
> correção corrigiu bugs que impediam a execução real (crashes de
> configuração, collator que descartava o masking de labels do packing,
> IDs de modelo/dataset fabricados) e adicionou um caminho de GPU única via
> Google Colab (`colab/`). "Validado em CPU" aqui referia-se majoritariamente
> a testes que reimplementavam a lógica localmente em vez de importar o
> código real — isso também foi corrigido. Ver `README.md` para o estado atual.

## Resumo

O pipeline de adaptação Gemma 4 → PT-BR está completo e validado em CPU.
Todas as etapas de infraestrutura foram implementadas e testadas.

## Validação Executada

| Check | Status | Evidência |
|-------|--------|-----------|
| Testes unitários | 198 passed, 0 failed | `pytest tests/` |
| Smoke test E2E | 13/13 passed | `python -m tests.smoke_test` |
| Conformidade Gemma 4 | 28 testes específicos | `test_gemma4_compliance.py` |
| Golden tests | 19 testes com fixtures | `test_golden.py` |
| Configs YAML | Todos válidos | CI job `validate-configs` |
| Preflight | Pass (exceto GPU) | `gemma4pt preflight` |
| Parsing robusto | 11 golden cases | Fixtures determinísticas |
| Bootstrap CI | Propriedades validadas | Determinismo, narrowing |
| Contamination | Exact/norm/fuzzy/ngram | Testes com edge cases |

## Componentes Implementados

### Dados
- [x] Aurora-PT loader com packing
- [x] Quality manifest (lang ID, PII, toxicity)
- [x] Cluster deduplication (MinHash LSH)
- [x] Split por clusters (previne leakage)
- [x] Replay mix builder (EN 5-15%)
- [x] Contamination checker (4 métodos)

### Treino
- [x] CPT Trainer com LoRA/full FT
- [x] SFT Trainer
- [x] DPO/ORPO Trainer
- [x] PEFT Factory (LoRA, DoRA, QLoRA, prefix)
- [x] Residual Merge (task arithmetic)
- [x] Callbacks (throughput, metrics)
- [x] Checkpoint resume

### Avaliação
- [x] Benchmark runner com cache
- [x] 20+ benchmarks configurados
- [x] Prompt templates por tarefa
- [x] Think mode support (on/off/budget)
- [x] Bootstrap CI 95%
- [x] Paired permutation test
- [x] Effect size (Cohen's d)
- [x] Holm correction
- [x] Report builder (CSV, MD, plots)

### Operacional
- [x] CLI com 10+ comandos
- [x] Preflight validation
- [x] Dry-run / tiny / cpu-only modes
- [x] Run manifest generation
- [x] CI workflow (lint, test, smoke, validate)

## Etapas Pendentes (requerem GPU)

| Etapa | Requisito | Estimativa |
|-------|-----------|-----------|
| Download modelo Gemma 4 | HF token + storage | ~10min |
| Tokenizer audit real | 1×GPU qualquer | ~5min |
| CPT piloto (E4B, LoRA) | 1×A100 40GB | ~1-2h |
| CPT principal (26B, full) | 8×A100 80GB | ~24h |
| Residual merge | 24GB RAM (CPU OK) | ~30min |
| SFT | 1×A100 40GB | ~2h |
| Avaliação completa | 1×A100 | ~4h |
| DPO (opcional) | 1×A100 | ~2h |

## Riscos Identificados

1. **Model IDs**: Alguns IDs do Gemma 4 podem mudar quando Google publicar oficialmente. Configurável via YAML.
2. **Aurora-PT acesso**: Corpus requer autenticação HF. Fallback documentado.
3. **Benchmark hub_ids**: CAPITU, Math-PT, BoolQ-PT — IDs a confirmar.
4. **DeepSpeed config**: `ds_zero3.json` a ser criado quando hardware disponível.

## Comando de Validação

```bash
# Validação completa em CPU (< 5 segundos)
pytest tests/ -q && python -m tests.smoke_test && echo "READY"
```
