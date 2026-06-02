# Plano Experimental — Adaptação Gemma 4 ao Português Brasileiro

## Objetivo Científico

Comparar estratégias de adaptação linguística do Gemma 4 ao pt-BR, medindo:
1. Ganho em benchmarks PT após CPT
2. Custo de catastrophic forgetting (retenção EN)
3. Eficácia de diferentes métodos PEFT (LoRA vs DoRA vs QLoRA vs Full)
4. Eficácia de replay ratio (0%, 5%, 10%, 15% inglês)
5. Residual Merge vs SFT para recuperação de instruction-following
6. Impacto do think mode em tarefas de raciocínio

## Protocolo em 11 Passos

### Passo 1: Data QC e Splitting (Cluster-Level)

```bash
bash scripts/run_data_qc.sh
```

**Produz:**
- `outputs/data_qc/quality_manifest.json` — Manifesto de qualidade por documento
- `outputs/data_qc/split_indices.json` — Índices train/val por cluster
- `outputs/data_qc/split_stats.json` — Estatísticas dos splits
- `outputs/data_qc/domain_stats.json` — Distribuição por domínio

**Critério de aceite:** Documentos near-duplicates no mesmo split. Zero vazamento.

### Passo 2: Auditoria de Tokenizer

```bash
bash scripts/run_tokenizer_audit.sh
```

**Produz:**
- `outputs/tokenizer_audit.json` — Fertilidade tokens/palavra, tokens/char
- Comparação Gemma 4 vs BERTimbau vs Sabia tokenizer
- Cálculo do orçamento real: 331B tokens GPT-2 → X tokens Gemma 4

**Critério de aceite:** Orçamento em tokens Gemma 4 calculado e documentado.

### Passo 3: Contamination Checks

```bash
bash scripts/run_contamination_checks.sh
```

**Produz:**
- `outputs/contamination/report.json` — Taxa de contaminação por benchmark
- Métodos: exact hash, normalized, MinHash LSH (Jaccard ≥ 0.8), 5-gram overlap

**Critério de aceite:** Contaminação < 1% em todos os benchmarks obrigatórios.

### Passo 4: CPT Piloto (E4B, 5B tokens)

```bash
bash scripts/run_cpt_pilot.sh
```

**Variantes a testar em paralelo (se GPUs disponíveis):**
| ID | Método | Replay |
|----|--------|--------|
| B1 | LoRA r=64 | 15% EN |
| B2 | DoRA r=64 | 15% EN |
| B3 | QLoRA r=64 | 15% EN |
| C1 | LoRA r=64 | 0% (PT only) |
| C2 | LoRA r=64 | 5% EN |
| C3 | LoRA r=64 | 10% EN |
| C4 | LoRA r=64 | 15% EN |

**Critério de aceite:** Perplexidade decrescente. Nenhum NaN/Inf nos gradientes.

### Passo 5: Avaliação do Piloto

```bash
bash scripts/run_eval.sh --config configs/eval/benchmarks.yaml --models "outputs/cpt_pilot/*"
```

**Métricas comparadas:**
- Benchmarks PT (14 obrigatórios)
- Retenção EN (MMLU, HellaSwag, ARC samples)
- Selecionar melhor: método PEFT × replay ratio

### Passo 6: Residual Merge

```bash
bash scripts/run_residual_merge.sh
```

**Fórmula:** `adapted = cpt_weights + α × (instruct_weights - base_weights)`

**Alpha sweep:** [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

### Passo 7: SFT

```bash
bash scripts/run_sft.sh
```

**Dados:** Instrução PT-BR (separado de Aurora-PT usado no CPT)
**Base:** Checkpoint CPT (não modelo original)

### Passo 8: Trilha Principal (26B-A4B)

Repete a melhor receita do piloto:
```bash
bash scripts/run_cpt_main.sh
```

**Orçamentos:** 20B tokens e 50B tokens (comparar scaling)

### Passo 9: (Opcional) 31B Dense

Se orçamento computacional permitir.

### Passo 10: Avaliação Full Suite

```bash
bash scripts/run_eval.sh
```

**Inclui:**
- 14+ benchmarks PT obrigatórios
- 3 benchmarks de retenção EN
- Benchmarks exploratórios habilitados
- Think_on vs think_off em benchmarks de raciocínio
- Bootstrap CI 95% para todos os scores
- Testes pareados com correção de Holm

### Passo 11: Geração de Artefatos para Paper

```bash
python3 -m src.eval.report_builder
```

**Produz:**
- `reports/results_full.csv`
- `reports/results_pivot.csv`
- `reports/group_averages.csv`
- `reports/summary.md`
- `reports/figures/heatmap.png`
- `reports/figures/radar_chart.png`
- `reports/figures/scaling_curves.png`
- `reports/figures/retention_plot.png`
- `reports/figures/tokenizer_fertility.png`

## Hipóteses a Testar

| ID | Hipótese | Métrica Principal |
|----|----------|-------------------|
| H1 | CPT no Aurora-PT melhora benchmarks PT | Δ accuracy vs baseline |
| H2 | English replay previne catastrophic forgetting | Δ MMLU/ARC/HellaSwag |
| H3 | Residual merge recupera instruction-following sem SFT | Accuracy em ENEM/OAB |
| H4 | CPT + SFT > CPT + Residual Merge | Macro-avg todos benchmarks |
| H5 | Think mode melhora raciocínio complexo | Δ ENEM/OAB/Math-PT |
| H6 | DoRA ≥ LoRA para CPT em PT | Perplexidade + downstream |
| H7 | 15% EN > 5% EN para retenção | Δ benchmarks EN |
| H8 | 50B tokens > 20B tokens (scaling) | Log-linear improvement |

## Hardware Requerido

| Fase | GPU | Duração Estimada |
|------|-----|------------------|
| Data QC | CPU | ~2h |
| CPT Piloto (7 variantes) | 7× A100 80GB (ou sequencial em 1×) | ~7 dias |
| Avaliação Piloto | 1× A100 40GB | ~12h |
| CPT Principal 20B | 4× A100 80GB | ~3 dias |
| CPT Principal 50B | 4× A100 80GB | ~7 dias |
| Avaliação Final | 1× A100 40GB | ~24h |

## Riscos e Mitigações

| Risco | Mitigação |
|-------|-----------|
| OOM durante CPT | Gradient checkpointing + reduzir batch_size |
| Contaminação train↔eval | Contamination checks pré-treino |
| Instabilidade de treino | Gradient clipping + warmup + LR sweep |
| Catastrophic forgetting | Replay buffer EN + monitoramento contínuo |
| Resultados não-significativos | Bootstrap CI + power analysis prévia |
