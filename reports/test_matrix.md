# Test Matrix

## Resumo

| Categoria | Testes | Status |
|-----------|--------|--------|
| Unit | 170 | All pass |
| Golden | 19 | All pass |
| Compliance (Gemma 4) | 28 | All pass |
| Smoke (E2E) | 13 | All pass |
| Skipped (deps opcionais) | 2 | N/A |
| **Total** | **198 + 13** | **All pass** |

## Detalhamento por Módulo

### test_gemma4_compliance.py (28 testes)
- TestNoThinking: 4 — formato sem pensamento
- TestWithThinking: 4 — formato com pensamento
- TestMultiTurnThinking: 5 — multi-turn, stripping
- TestResponseParsing: 4 — parsing de respostas
- TestModelModes: 4 — text-only / IT / baseline
- TestFewShot: 2 — construção few-shot
- TestEdgeCases: 5 — robustez

### test_golden.py (19 testes)
- TestGoldenPrompts: 2 — prompts vs fixtures
- TestGoldenParsing: 3 — letra, think strip, unicode
- TestGoldenMetrics: 5 — accuracy bounds, f1
- TestGoldenBootstrap: 3 — CI properties
- TestGoldenContamination: 6 — exact/norm/fuzzy/ngram

### test_prompt_templates.py (21 testes)
- TestGemma4Wrapping: 5
- TestStripThought: 4
- TestPromptBuilder: 4
- TestTaskPromptTemplate: 6
- TestFewShot: 2

### test_contamination.py (24 testes)
- Normalize, hash, ngrams, exact, normalized, fuzzy, ngram overlap

### test_metrics.py (16 testes)
- Accuracy, F1, Pearson, ROUGE-L, refusal

### test_bootstrap.py (5 testes)
- CI bounds, paired test, determinism

### test_data_pipeline.py (12 testes)
- Aurora loader, tokenization, packing

### test_config.py (8 testes)
- Load, merge, flatten, nested refs

### test_checkpointing.py (8 testes)
- Save/load state, find checkpoint

### test_parsing.py (16 testes)
- Letter/number extraction per task

### test_report_builder.py (13 testes)
- Table, groups, comparison, summary, plots

### test_callbacks.py + test_logging_utils.py + test_instruction_builder.py (45 testes)
- Misc infrastructure

### Smoke test (13 checks)
- Config, seed, prompts, metrics, bootstrap, stats, checkpoint, contamination, report, merge, preflight

## Classificação

| Tipo | Arquivos | Tempo |
|------|----------|-------|
| Unit (rápido, sem I/O) | test_config, test_metrics, test_parsing | <0.5s |
| Integration (componentes juntos) | test_golden, test_contamination | ~1s |
| Smoke (E2E pipeline) | smoke_test.py | ~2s |
| GPU-only | (nenhum — mockados) | N/A |

## Markers pytest

```bash
# Rodar apenas unit tests
pytest tests/ -m "not slow and not gpu"

# Rodar smoke
pytest tests/ -m smoke

# Rodar tudo exceto GPU
pytest tests/ -m "not gpu"
```
