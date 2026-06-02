# Train Ready — Checklist de Prontidão para Treino

## Checklist Completo

### Ambiente
- [ ] Python >= 3.10 instalado
- [ ] `pip install -e ".[dev]"` sem erros
- [ ] `gemma4pt preflight` passa (sem falhas)
- [ ] GPU disponível (NVIDIA A100/H100 recomendado)
- [ ] CUDA >= 12.1 instalado
- [ ] >= 50GB de disco livre
- [ ] HuggingFace token configurado (`huggingface-cli login`)

### Dados
- [ ] Aurora-PT acessível via HF Hub
- [ ] `gemma4pt data-validate` passa
- [ ] Contamination check executado (`gemma4pt contamination-check`)
- [ ] Manifesto de qualidade gerado

### Modelos
- [ ] Tokenizer do Gemma 4 acessível
- [ ] Audit de fertilidade executado (`gemma4pt tokenizer-audit`)
- [ ] Modelo base baixado ou acessível

### Validação
- [ ] `pytest tests/` — todos os testes passam
- [ ] `gemma4pt smoke` — smoke test end-to-end passa
- [ ] Configs YAML validados (sem erros de parse)

### Documentação
- [ ] `docs/GEMMA4_COMPLIANCE.md` revisado
- [ ] `docs/EXPERIMENT_PLAN.md` revisado
- [ ] Hipóteses definidas antes do treino

---

## Comandos de Validação Rápida

```bash
# 1. Instalar
pip install -e ".[dev]"

# 2. Preflight
gemma4pt preflight

# 3. Testes
pytest tests/ -v

# 4. Smoke test
gemma4pt smoke

# 5. Validar dados (dry-run)
gemma4pt data-validate --dry-run

# 6. Manifesto
gemma4pt manifest --config configs/train/cpt_pilot.yaml
```

## Execução do Treino

### Piloto (E4B, LoRA, ~1h em 1×A100)
```bash
gemma4pt train-cpt configs/train/cpt_pilot.yaml
```

### Principal (26B, Full FT, ~24h em 8×A100)
```bash
gemma4pt train-cpt configs/train/cpt_main.yaml
```

### Com dry-run (valida config sem executar)
```bash
gemma4pt train-cpt configs/train/cpt_pilot.yaml --dry-run
```

### Com modo tiny (10 steps, para debug)
```bash
gemma4pt train-cpt configs/train/cpt_pilot.yaml --tiny
```

## Pós-Treino

```bash
# Residual merge
gemma4pt merge \
  --base-model google/gemma-4-E4B \
  --instruct-model google/gemma-4-E4B-it \
  --cpt-model outputs/cpt_pilot/final \
  --alpha 0.7 0.8 0.9 1.0 1.1

# Avaliação
gemma4pt eval --model outputs/residual_merge/alpha_1.00

# Relatório
gemma4pt report
```

## Etapas que dependem estritamente de GPU

| Etapa | Requisito mínimo | Alternativa CPU |
|-------|-----------------|-----------------|
| CPT (piloto LoRA) | 1×A100 40GB | `--tiny` com modelo dummy |
| CPT (principal full) | 8×A100 80GB | Não viável |
| SFT | 1×A100 40GB | `--tiny` |
| Residual Merge | 24GB RAM (CPU) | Funciona em CPU |
| Avaliação (inferência) | 1×A100 | Cache de resultados |
| Smoke test | CPU apenas | N/A |
| Bootstrap CI | CPU apenas | N/A |
| Report | CPU apenas | N/A |
