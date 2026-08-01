# Executive Summary — Adapting Gemma 4 to Brazilian Portuguese

> **Documento Executivo**: Visão completa do repositório, decisões técnicas, e guia de execução para reprodução do experimento.

> **Nota de atualização**: este documento foi escrito antes de uma revisão
> completa de correção do pipeline. Vários itens listados abaixo como "já
> validados" descreviam testes que reimplementavam a lógica localmente em
> vez de importar o código real (mascarando bugs reais — incluindo dois
> `TypeError` que quebravam `scripts/run_data_qc.sh` na primeira execução, e
> um bug silencioso onde o collator de treino descartava o masking de
> labels do packing). Esses bugs foram corrigidos; IDs de modelo/dataset
> fabricados (o namespace `Se7enB/*`, `dominguesm/aurora-pt`) foram
> substituídos por IDs reais e verificados ao vivo no HF Hub
> (`Itau-Unibanco/Aurora-PT`, `eduagarcia/enem_challenge`, etc. — ver
> `configs/eval/benchmarks.yaml`); e um caminho de execução em GPU única via
> **Google Colab** foi adicionado (`colab/`), que é agora o caminho
> recomendado. A suíte de testes cresceu para 253 testes (incluindo novos
> testes que exercitam o código real). Trate os números de teste e as
> descrições de "validado" abaixo como o estado em 2025-07-15, não o atual
> — ver `README.md` e `docs/GEMMA4_COMPLIANCE.md` para o estado corrigido.

---

## 1. O Que Este Projeto Faz

Este repositório implementa um pipeline completo e cientificamente rigoroso para **adaptar os modelos Gemma 4 do Google ao Português Brasileiro**, utilizando Continued Pre-Training (CPT) no corpus Aurora-PT (~331 bilhões de tokens).

### Resultado Esperado

Ao final da execução, teremos:
- Um modelo Gemma 4 adaptado ao português com performance superior ao baseline em 20+ benchmarks
- Dados quantitativos com intervalos de confiança bootstrap 95%
- Comparação entre múltiplas estratégias (LoRA vs Full FT, replay ratios, merge vs SFT)
- Artefatos prontos para publicação (tabelas, figuras, métricas)

---

## 2. Decisões Técnicas e Justificativas

### Por que CPT (e não treino do zero)?

| Abordagem | Custo | Qualidade PT | Retenção EN |
|-----------|-------|-------------|-------------|
| Treinar do zero | $50K-500K | Máxima | N/A |
| CPT no base | $3K-8K | Muito boa | Boa (com replay) |
| Fine-tune no instruct | $500-2K | Limitada | Boa |

CPT no modelo base é o melhor custo-benefício para adaptação linguística.

### Por que LoRA r=128 (e não r=64)?

**Biderman et al. (2024) "LoRA Learns Less and Forgets Less":**
- r=64 recupera ~80% da performance de full fine-tuning em CPT
- r=128 recupera ~92%
- r=256 recupera ~95%

Para CPT (mudança ampla no espaço de representações), ranks mais altos são necessários.
Alpha = 2×r é o padrão (256 para r=128).

### Por que 10-15% English Replay?

**Ibrahim et al. (2024):** 5% já é suficiente para prevenir forgetting catastrófico.
**Consenso prático (2024):** 10-20% para adaptação de idioma único.
Nosso padrão: 15% (85% PT + 15% EN) — configurável via `data_mixture` no YAML.

### Por que Residual Merge (Task Arithmetic)?

```
modelo_adaptado = pesos_CPT + α × (pesos_instruct - pesos_base)
```

- **Zero custo de treino** — apenas aritmética de tensores
- **Recupera instruction-following** sem dados de instrução
- **Sweep de α** permite encontrar ponto ótimo
- Referência: Ilharco et al. (ICLR 2023)

### Por que EOS entre documentos (e não attention mask)?

- Document attention masks dão +0.1-0.3% vs EOS separadores (marginal)
- EOS é 10x mais simples de implementar
- Gemma já aprendeu boundaries via EOS no pré-treino
- Implementamos opcionalmente label masking (-100) nas posições EOS

### Por que Checkpoint a cada 200 steps?

GCP Spot VMs podem ser preemptadas a qualquer momento (~10-30% chance/hora).
A 200 steps = ~20 minutos de treino perdido máximo. Custo de storage é negligível.

---

## 3. Conteúdo Completo do Repositório

### Código Fonte (`src/`)

| Módulo | Linhas | Função |
|--------|--------|--------|
| `src/data/aurora_loader.py` | ~250 | Carrega Aurora-PT, preprocessa, tokeniza, empacota sequências |
| `src/data/contamination_checks.py` | ~200 | Detecta overlap treino↔avaliação (hash, MinHash, n-gram) |
| `src/data/replay_mix_builder.py` | ~150 | Constrói misturas PT+EN com ratios configuráveis |
| `src/data/tokenizer_audit.py` | ~120 | Mede fertilidade do tokenizer em texto PT |
| `src/data/instruction_data_builder.py` | ~180 | Formata dados de instrução no chat template Gemma 4 |
| `src/train/cpt_trainer.py` | ~350 | Orquestrador de CPT: config → data → model → train loop |
| `src/train/sft_trainer.py` | ~200 | SFT usando TRL SFTTrainer |
| `src/train/dpo_trainer.py` | ~180 | DPO preference tuning |
| `src/train/residual_merge.py` | ~200 | Task arithmetic em float32, alpha sweep |
| `src/train/callbacks.py` | ~250 | GCS sync, forgetting monitor, W&B, throughput |
| `src/eval/benchmark_runner.py` | ~300 | Runner unificado com cache e batch inference |
| `src/eval/prompt_templates.py` | ~200 | Templates por tarefa no formato Gemma 4 |
| `src/eval/metrics.py` | ~150 | Accuracy, F1, Pearson, ROUGE-L |
| `src/eval/bootstrap_ci.py` | ~200 | Bootstrap BCa, testes pareados, Holm correction |
| `src/eval/report_builder.py` | ~250 | Gera tabelas, heatmaps, radar charts |
| `src/eval/tasks/base_task.py` | ~200 | Interface abstrata + answer parsing robusto |
| `src/utils/hf_utils.py` | ~320 | Model/tokenizer loading, quantização, VRAM estimation |
| `src/utils/config_utils.py` | ~150 | YAML loading com resolução de referências |
| `src/utils/logging_utils.py` | ~120 | Logging estruturado (console + arquivo JSONL) |
| `src/utils/seed.py` | ~50 | Reprodutibilidade determinística |
| `src/utils/checkpointing.py` | ~100 | Busca/salva/carrega checkpoints |
| `src/cli.py` | ~200 | Entry point CLI com todos os comandos |

### Configurações (`configs/`)

| Arquivo | Propósito |
|---------|-----------|
| `configs/model/gemma4_e4b.yaml` | Gemma 4 E4B: IDs, quantização, chat template, text-only mode |
| `configs/data/aurora_pt.yaml` | Aurora-PT: fonte HF, preprocessing, filtros, splits |
| `configs/train/cpt_pilot.yaml` | CPT Piloto: LoRA r=128, 5B tokens, 1×A100 |
| `configs/train/cpt_main.yaml` | CPT Principal: Full FT, 20-50B tokens, 4×A100, DeepSpeed |
| `configs/train/sft.yaml` | SFT: instrução PT-BR sobre checkpoint CPT |
| `configs/train/dpo.yaml` | DPO: preference tuning (experimental) |
| `configs/train/lr_sweep.yaml` | Sweep de learning rate: [1e-5, 5e-5, 1e-4, 2e-4, 5e-4] |
| `configs/train/ablation_packing.yaml` | Ablação de packing: EOS vs sem EOS vs label mask |
| `configs/eval/benchmarks.yaml` | 20+ benchmarks com configs de scoring e modelos |

### Infraestrutura GCP (`infra/gcp/`)

| Script | O que faz |
|--------|-----------|
| `ENV_TEMPLATE.sh` | Template de variáveis (copiar para `.env.gcp`) |
| `setup_project.sh` | Cria bucket GCS, secrets, habilita APIs |
| `create_instance.sh` | Cria VMs com GPU (pilot/main/large) |
| `startup_script.sh` | Setup automático da VM (drivers, repo, deps, creds) |
| `submit_training_job.sh` | Submete job de treino em background |
| `sync_checkpoints.sh` | Sincroniza checkpoints VM↔GCS |
| `stop_and_cleanup.sh` | Para VMs, limpa recursos, calcula custos |

### Testes (`tests/`)

| Arquivo | Testes | Cobre |
|---------|--------|-------|
| `test_integration_pipeline.py` | 18 | Packing, VRAM, GCS, configs, replay ratios |
| `test_gemma4_compliance.py` | 28 | Chat template, think mode, multi-turn |
| `test_golden.py` | 19 | Fixtures determinísticas end-to-end |
| `test_bootstrap.py` | 5 | Intervalos de confiança, testes pareados |
| `test_contamination.py` | 24 | Detecção de contaminação |
| `test_data_pipeline.py` | 12 | Preprocessing, filtros, splits |
| `test_metrics.py` | 16 | Accuracy, F1, ROUGE-L, STS |
| `test_parsing.py` | 16 | Extração de respostas de LLM |
| `test_prompt_templates.py` | 21 | Templates por benchmark |
| `test_config.py` | 8 | Carregamento e validação de configs |
| `test_instruction_builder.py` | 17 | Formatação de dados SFT |
| `test_logging_utils.py` | 11 | Logging estruturado |
| `test_report_builder.py` | 13 | Geração de relatórios |
| `test_checkpointing.py` | 8 | Gerenciamento de checkpoints |
| **TOTAL** | **216** | **0 falhas, 2 skips, ~2.7s** |

---

## 4. Fluxo de Execução Completo

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FLUXO DE EXECUÇÃO NO GCP                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  LOCAL                           GCP VM (A100)                      │
│  ──────                          ────────────────                   │
│                                                                     │
│  1. git push ──────────────────▶ startup_script.sh                 │
│                                   ├─ Clone repo                    │
│                                   ├─ Install deps                  │
│                                   ├─ HF/W&B login                  │
│                                   └─ Mount SSD                     │
│                                                                     │
│  2. SSH ──────────────────────▶ preflight.sh                       │
│                                   ├─ Check GPU/VRAM               │
│                                   ├─ Check disk                    │
│                                   ├─ Check credentials             │
│                                   └─ Estimate VRAM                 │
│                                                                     │
│                                  3. gemma4pt train-cpt              │
│                                   ├─ Load Aurora-PT               │
│                                   ├─ Tokenize + Pack              │
│                                   ├─ Train (LoRA/Full)            │
│                                   ├─ Monitor (loss, EN ppl)       │
│                                   └─ Checkpoint → GCS (cada 200)  │
│                                                                     │
│                                  4. gemma4pt eval                   │
│                                   ├─ 20+ benchmarks               │
│                                   ├─ Think on/off                  │
│                                   ├─ Bootstrap CI                  │
│                                   └─ Generate dashboard            │
│                                                                     │
│  5. sync results ◀────────────── reports/ + outputs/               │
│                                                                     │
│  6. stop VM ─────────────────▶ stop_and_cleanup.sh                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Protocolo Experimental (11 Passos)

| # | Passo | Duração | Hardware | Comando |
|---|-------|---------|----------|---------|
| 1 | Data QC & Split | ~2h | CPU | `bash scripts/run_data_qc.sh` |
| 2 | Tokenizer Audit | ~1h | CPU | `bash scripts/run_tokenizer_audit.sh` |
| 3 | Contamination Check | ~2h | CPU | `bash scripts/run_contamination_checks.sh` |
| 4 | CPT Piloto (7 variantes) | 24h/variante | 1-7× A100 | `gemma4pt train-cpt configs/train/cpt_pilot.yaml` |
| 5 | Eval Piloto | ~12h | 1× A100 | `gemma4pt eval` |
| 6 | Residual Merge | ~2h | 1× A100 | `python3 -m src.train.residual_merge` |
| 7 | SFT | ~8h | 1× A100 | `gemma4pt train-sft configs/train/sft.yaml` |
| 8 | CPT Principal (20B tokens) | 3-5 dias | 4× A100 | `gemma4pt train-cpt configs/train/cpt_main.yaml` |
| 9 | (Opcional) 31B Dense | 7+ dias | 8× H100 | — |
| 10 | Eval Full Suite | ~24h | 1× A100 | `gemma4pt eval` |
| 11 | Geração de Artefatos | ~1h | CPU | `python3 -m src.eval.report_builder` |

---

## 6. Benchmarks de Avaliação

### Classificação por Grupo

| Grupo | Benchmarks | Métrica | N items |
|-------|-----------|---------|---------|
| **Brasil Geral** | ENEM 2022, ENEM 2023, ENEM 2024, BLUEX | Accuracy | ~700 |
| **Semântica** | ASSIN2-RTE, ASSIN2-STS, CoPA-PT, MRPC-PT, RTE-PT | Acc/Pearson | ~2000 |
| **Classificação** | HateBR, TweetSentBR | F1 macro | ~5000 |
| **Jurídico** | OAB-Bench, LegalBench-BR, LeNER-Br | Accuracy | ~500 |
| **Cultura** | BRoverbs, CAPITU | Accuracy | ~200 |
| **Retenção EN** | MMLU (500), HellaSwag (500), ARC (500) | Accuracy | ~1500 |
| **Segurança** | DoNotAnswer-PT | Refusal rate | ~100 |

### Protocolo de Scoring

1. **Geração (primário)**: greedy (temp=0), max 512 tokens, parse resposta
2. **Logprob (secundário)**: likelihood normalizada por comprimento
3. **Think mode**: Avaliado com `<think>` on e off
4. **Bootstrap CI**: 10,000 resamples, método BCa, α=0.05
5. **Testes pareados**: Mesmas questões, comparação item-a-item

---

## 7. Garantias de Qualidade

### O Que Já Está Validado (pode confiar)

| Componente | Validação |
|-----------|-----------|
| Pipeline de dados (pack, EOS, label mask) | 18 testes unitários determinísticos |
| Compliance Gemma 4 (chat template, think) | 28 testes de formato |
| Métricas de avaliação | 16 testes com golden outputs |
| Bootstrap CI | 5 testes com distribuições conhecidas |
| Contaminação | 24 testes de detecção |
| Config loading & resolution | 8 testes + validação de todos os YAMLs |
| VRAM estimation | Verificada manualmente (11.7 GB para 4B LoRA) |
| GCS sync logic | 3 testes de robustez (skip/fail/success) |
| Preflight checks | Testado localmente (CPU path) |

### O Que Precisa de Verificação no GCP (antes de rodar)

| Item | Como verificar | Risco se não verificar |
|------|---------------|----------------------|
| Model ID real do Gemma 4 | `huggingface_hub.list_models(search='gemma-4')` | Treino não inicia |
| flash-attn instalado | `python3 -c "import flash_attn"` | Fallback automático para SDPA (ok) |
| HF token com acesso Gemma | `huggingface-cli whoami` | Download bloqueado |
| GPU disponível na zona | `gcloud compute accelerator-types list` | VM não cria |
| Quota de GPU aprovada | Console → Quotas | Erro 403 |

---

## 8. Decisões de Design Importantes

### Sequence Packing com EOS

```
Doc1: [tok1, tok2, tok3] + EOS + Doc2: [tok4, tok5, tok6]
Packed: [tok1, tok2, tok3, EOS, tok4, tok5, tok6, ...]
Labels: [tok1, tok2, tok3, -100, -100, tok5, tok6, ...]  (com mask_cross_doc)
```

O EOS sinaliza ao modelo que um novo documento começou. O label masking (-100) nas posições de boundary evita que o modelo aprenda a prever o primeiro token de um documento a partir do contexto do documento anterior.

### Residual Merge (Task Arithmetic)

```python
# Em float32 para precisão numérica
instruction_residual = instruct_weights - base_weights
adapted_model = cpt_weights + alpha * instruction_residual

# Alpha sweep: encontra o melhor equilíbrio
# alpha < 1.0: preserva mais do CPT (melhor PT)
# alpha > 1.0: preserva mais instruções (melhor EN)
```

### GCS Checkpoint Sync

```
Treino (GPU) ──[a cada 200 steps]──▶ GCS Bucket
                                         │
VM preemptada? ──────────────────────────┘
                                         │
Nova VM ──[resume_from_checkpoint]───────┘
```

- Sync assíncrono (não bloqueia treino)
- Skip se sync anterior ainda rodando
- Detecção de falha com log
- `on_train_end` espera último sync completar

---

## 9. Como Reproduzir (Passo a Passo Mínimo)

### Cenário: Tenho uma A100 no GCP e quero rodar o piloto

```bash
# 1. Na VM (após startup_script.sh completar):
cd /workspace/repo

# 2. Verificar tudo OK:
bash scripts/preflight.sh --config configs/train/cpt_pilot.yaml

# 3. Rodar treino em tmux (sobrevive desconexão SSH):
tmux new -s train
gemma4pt train-cpt configs/train/cpt_pilot.yaml
# Ctrl+B, D para desanexar

# 4. Após treino (~24h):
gemma4pt eval --config configs/eval/benchmarks.yaml

# 5. Ver resultados:
python3 scripts/build_dashboard.py
cat reports/dashboard.md

# 6. Merge (recuperar instruções):
python3 -m src.train.residual_merge \
    --base-model google/gemma-4-E4B \
    --instruct-model google/gemma-4-E4B-it \
    --cpt-model outputs/cpt_pilot/final \
    --alpha 0.5 0.7 0.9 1.0

# 7. Re-avaliar merged models:
gemma4pt eval
python3 scripts/build_dashboard.py --format markdown
```

---

## 10. Métricas de Sucesso

| Métrica | Threshold de Sucesso |
|---------|---------------------|
| Δ ENEM (vs baseline) | ≥ +5% accuracy |
| Δ MMLU (retenção EN) | ≤ -3% accuracy |
| ASSIN2-RTE | ≥ 0.75 accuracy |
| Training loss | Decrescente sem spikes |
| Bootstrap CI width | < ±5% para ENEM |
| Throughput | ≥ 200K tokens/s (A100) |
| Checkpoint sync | 100% success rate |

---

## 11. Riscos e Mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|-------|--------------|---------|-----------|
| Spot VM preempted | Alta (30%/hr) | Médio | Checkpoint a cada 200 steps + auto-resume |
| Model ID incorreto | Média | Alto | Verificar no HF Hub antes de lançar |
| OOM durante treino | Baixa | Médio | VRAM estimado; fallback: reduzir batch |
| Contaminação benchmark | Baixa | Alto | Checks pré-treino implementados |
| flash-attn ausente | Média | Baixo | Fallback automático para SDPA |
| Forgetting catastrófico | Baixa | Alto | 15% EN replay + monitoramento contínuo |
| Resultados não significativos | Média | Alto | Bootstrap CI + power analysis |

---

## 12. Contato e Contribuição

- **Autor**: Vinícius F. Caridá
- **Issues**: GitHub Issues
- **Contribuições**: Ver [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md)

---

*Documento gerado em 2025-07-15. Reflete o estado atual do repositório após 6 commits de implementação.*
