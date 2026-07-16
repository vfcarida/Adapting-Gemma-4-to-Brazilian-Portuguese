# Known GPU-Only Steps

Etapas que **não podem** ser executadas sem GPU e as alternativas CPU implementadas.

## Etapas que requerem GPU

| Etapa | Por quê | Alternativa CPU implementada |
|-------|---------|------------------------------|
| CPT (Continued Pretraining) | Forward/backward de modelo 4B+ | `--tiny` (10 steps), `--dry-run` (valida config) |
| SFT (Supervised Fine-Tuning) | Idem | `--tiny`, `--dry-run` |
| DPO/ORPO | Idem + modelo de referência | `--dry-run` |
| Avaliação (inferência) | Geração de texto com modelo grande | Cache de resultados, mock predictions |
| Flash Attention | Kernel CUDA específico | Fallback para attention padrão |
| BitsAndBytes quantization | CUDA kernel | Skipped em CPU |

## Etapas que funcionam em CPU

| Etapa | Tempo CPU | Notas |
|-------|-----------|-------|
| Preflight | <1s | Completo |
| Data validation | ~10s | Streaming dataset |
| Quality manifest | ~minutes | Depende do tamanho do corpus |
| Cluster dedup | ~minutes | MinHash é CPU-bound |
| Contamination check | ~minutes | Depende do tamanho |
| Tokenizer audit | ~30s | Com tokenizer local |
| Residual merge | ~30min | 24GB RAM suficiente |
| Bootstrap CI | <1s | Numpy puro |
| Stats tests | <1s | Scipy |
| Report generation | <1s | Pandas + matplotlib |
| Config validation | <1s | YAML parse |
| Smoke test | ~2s | Tudo sintético |

## Validação por dry-run

Cada etapa GPU tem um modo `--dry-run` que:
1. Carrega e valida a config
2. Verifica paths e dependências
3. Reporta o que seria executado
4. Não baixa modelos nem inicia treino

```bash
# Validar CPT sem executar
gemma4pt train-cpt configs/train/cpt_pilot.yaml --dry-run

# Validar avaliação
gemma4pt eval --dry-run

# Validar merge
gemma4pt merge --base-model X --instruct-model Y --cpt-model Z --dry-run
```

## Tiny mode

O modo `--tiny` executa a etapa real mas com recursos mínimos:
- max_steps=10
- batch_size=1
- Serve para validar que o código executa sem erros
- Requer pelo menos uma GPU pequena (ou CPU com paciência)

## O que fazer quando GPU ficar disponível

1. `gemma4pt preflight` — verifica CUDA
2. `gemma4pt train-cpt configs/train/cpt_pilot.yaml --tiny` — valida com 10 steps
3. Se OK: `gemma4pt train-cpt configs/train/cpt_pilot.yaml` — treino real
4. `gemma4pt eval --model outputs/cpt_pilot/final`
5. `gemma4pt report`
