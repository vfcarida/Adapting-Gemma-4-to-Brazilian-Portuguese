# Relatório Técnico v2 — Gemma 4 Portuguese Adaptation

## Resumo Executivo

Pipeline completo de adaptação do Google Gemma 4 ao português brasileiro usando o corpus Aurora-PT (NorBERTo, ~331B tokens GPT-2). Implementa CPT, residual merge, SFT, DPO opcional, e avaliação em 20+ benchmarks com rigor estatístico.

**Estado:** Infraestrutura completa. Pronto para execução experimental.
**Testes:** 125 passando, 9 skipped (deps opcionais). 0 falhas.
**Arquivos:** 99 (Python, YAML, Bash, Markdown).

---

## 1. Mudanças Implementadas Nesta Versão

### 1.1 Correção do Template Gemma 4 (CRÍTICO)
- **Antes:** Template hardcoded com `<start_of_turn>/<end_of_turn>`
- **Agora:** `PromptBuilder` que usa `apply_chat_template` do tokenizer preferencialmente
- Fallback manual apenas quando `apply_chat_template` não está disponível
- Suporte a `think_on`, `think_off`, e `empty_channel`
- `strip_thought()` e `extract_thought()` como funções de primeira classe
- `BaselinePromptBuilder` para modelos não-chat (Sabia-7B)

### 1.2 Modo Text-Only
- `hf_utils.py` congela automaticamente módulos multimodais quando `text_only_mode: true`
- Padrões detectados: vision_tower, multi_modal_projector, pixel, image_encoder
- Reduz uso de memória e evita corrupção de capacidade visual

### 1.3 Pipeline de Qualidade de Dados
- **`quality_manifest.py`**: Manifesto por documento (language ID, PII, toxicidade, quality score)
- **`cluster_dedup.py`**: Deduplicação por MinHash LSH + split por clusters
- **`make_splits.py`**: Pipeline completo de QC → dedup → split

### 1.4 PEFT Factory
- **`peft_factories.py`**: Factory unificada para LoRA, DoRA, QLoRA, prefix tuning
- DoRA via `use_dora=True` no LoraConfig

### 1.5 Benchmarks Expandidos
- **Novos benchmarks:** BoolQ-PT, CAPITU, Math-PT, PublicHearingBR, LeNER-Br, LegalBench-BR
- **Retenção EN:** MMLU, HellaSwag, ARC (amostras de 500)
- **Exploratórios:** ALBA, MariNER, XL-Sum-PT
- **Total:** 20+ benchmarks configurados

### 1.6 Testes Estatísticos
- **`stats_tests.py`**: Permutation test, McNemar, Wilcoxon, effect size, correção Holm

### 1.7 Trilha Experimental em 3 Níveis
- **Piloto:** E4B (validação rápida, 5B tokens)
- **Principal:** 26B-A4B (resultados para paper, 20-50B tokens)
- **Opcional:** 31B dense

### 1.8 Configurações Atualizadas
- Todas as configs com comentários explicativos em PT-BR
- Config de modelo com freeze multimodal e vocab expansion
- Config de dados com pipeline de QC completo
- Config de eval com 20+ benchmarks e retenção EN
- Matriz de ablações com 15+ experimentos planejados

---

## 2. Estrutura do Repositório (99 arquivos)

```
gemma4-pt-br-adaptation/
├── configs/                          # 11 YAML configs
│   ├── data/aurora_pt.yaml          # Corpus + QC + mixtures + budgets
│   ├── model/gemma4_e4b.yaml        # Piloto (text-only, freeze vision)
│   ├── model/gemma4_26b.yaml        # Principal
│   ├── model/gemma4_31b.yaml        # Opcional
│   ├── model/baselines.yaml         # 5 baselines com protocolo
│   ├── train/cpt_pilot.yaml         # LoRA, 5000 steps
│   ├── train/cpt_main.yaml          # Full FT, DeepSpeed
│   ├── train/sft.yaml               # SFT pós-CPT
│   ├── train/dpo.yaml               # DPO opcional
│   ├── eval/benchmarks.yaml         # 20+ benchmarks
│   └── ablations/full_matrix.yaml   # 15+ experimentos
├── src/
│   ├── data/                         # 8 módulos
│   │   ├── aurora_loader.py         # Carga e packing
│   │   ├── quality_manifest.py      # QC por documento (NEW)
│   │   ├── cluster_dedup.py         # Dedup + split cluster (NEW)
│   │   ├── make_splits.py           # Pipeline QC→split (NEW)
│   │   ├── prompt_builders.py       # Chat template Gemma 4 (NEW)
│   │   ├── tokenizer_audit.py      # Fertilidade
│   │   ├── contamination_checks.py  # Hash + MinHash
│   │   ├── replay_mix_builder.py    # EN/code replay
│   │   └── instruction_data_builder.py
│   ├── train/                        # 6 módulos
│   │   ├── cpt_trainer.py
│   │   ├── sft_trainer.py
│   │   ├── dpo_trainer.py
│   │   ├── residual_merge.py
│   │   ├── peft_factories.py       # LoRA/DoRA/QLoRA factory (NEW)
│   │   └── callbacks.py
│   ├── eval/                         # 6 módulos + 21 tasks
│   │   ├── benchmark_runner.py
│   │   ├── prompt_templates.py      # Reescrito com PromptBuilder
│   │   ├── metrics.py
│   │   ├── bootstrap_ci.py
│   │   ├── stats_tests.py          # Testes estatísticos (NEW)
│   │   ├── report_builder.py
│   │   └── tasks/                   # 21 benchmarks
│   │       ├── boolq_pt.py         # (NEW)
│   │       ├── capitu.py           # (NEW)
│   │       ├── math_pt.py          # (NEW)
│   │       ├── lener_br.py         # (NEW)
│   │       ├── legalbench_br.py    # (NEW)
│   │       ├── publichearing_br.py # (NEW)
│   │       ├── retention_en.py     # MMLU/HellaSwag/ARC (NEW)
│   │       └── ... (14 existentes)
│   └── utils/                        # 5 módulos
│       ├── hf_utils.py              # + freeze multimodal (UPDATED)
│       ├── config_utils.py
│       ├── logging_utils.py
│       ├── checkpointing.py
│       └── seed.py
├── scripts/                          # 10 scripts bash
│   └── run_data_qc.sh              # (NEW)
├── tests/                            # 12 módulos (125 testes)
├── docs/                             # 6 documentos
│   └── EXPERIMENT_PLAN.md           # Protocolo 11 passos (NEW)
├── pyproject.toml
├── Makefile
└── README.md
```

---

## 3. Decisões de Design

| Decisão | Justificativa |
|---------|---------------|
| `apply_chat_template` preferencial | Evita drift se Google atualizar o template |
| Split por clusters (não doc) | Previne vazamento de near-duplicates |
| Freeze vision por padrão | Gemma 4 é multimodal, CPT é text-only |
| DoRA como ablação | Supera LoRA em benchmarks recentes |
| Empty think channel | Modelos grandes podem esperar o canal |
| Sabia-7B always few-shot | Não é instruction-tuned, precisa demonstrações |
| Retenção EN obrigatória | Mede catastrophic forgetting quantitativamente |

---

## 4. TODOs Explícitos

| Arquivo | TODO | Prioridade |
|---------|------|-----------|
| `configs/eval/benchmarks.yaml` | Identificar hub_id para CAPITU, Math-PT, BoolQ-PT, ALBA, MariNER | ALTA |
| `configs/eval/benchmarks.yaml` | Verificar template do Tucano 2 | MÉDIA |
| `configs/train/cpt_main.yaml` | Criar config DeepSpeed (ds_zero3.json) | ALTA |
| `src/eval/tasks/lener_br.py` | Parsing robusto de NER em diferentes formatos | MÉDIA |
| `src/data/quality_manifest.py` | Integrar fasttext para language ID preciso | BAIXA |
| `src/data/quality_manifest.py` | Classificador de toxicidade (substituir keywords) | BAIXA |
| `src/eval/metrics.py` | BERTScore (requer bert-score library) | MÉDIA |
| `configs/model/gemma4_31b.yaml` | Confirmar model IDs quando disponíveis | BAIXA |
| `src/eval/tasks/retention_en.py` | Testar com datasets reais do HF Hub | ALTA |

---

## 5. Checklist de Execução

```
[ ] 1. pip install -e ".[dev]"
[ ] 2. cp .env.example .env  (configurar HF_TOKEN)
[ ] 3. bash scripts/run_data_qc.sh
[ ] 4. bash scripts/run_tokenizer_audit.sh
[ ] 5. bash scripts/run_contamination_checks.sh
[ ] 6. bash scripts/run_cpt_pilot.sh
[ ] 7. bash scripts/run_eval.sh --models "outputs/cpt_pilot/*"
[ ] 8. bash scripts/run_residual_merge.sh
[ ] 9. bash scripts/run_sft.sh
[ ] 10. bash scripts/run_cpt_main.sh  (trilha principal)
[ ] 11. bash scripts/run_eval.sh  (full suite)
[ ] 12. python3 -m src.eval.report_builder
```

---

## 6. Métricas do Projeto

| Métrica | Valor |
|---------|-------|
| Testes passando | 125 |
| Testes skipped | 9 (deps opcionais) |
| Arquivos Python | 54 |
| Arquivos YAML | 11 |
| Scripts Bash | 10 |
| Documentação | 8 docs |
| Benchmarks configurados | 20+ |
| Métodos PEFT suportados | 4 (LoRA, DoRA, QLoRA, prefix) |
| Modelos configurados | 8 (4 próprios + 4 baselines) |
| Trilhas experimentais | 3 (piloto, principal, opcional) |
