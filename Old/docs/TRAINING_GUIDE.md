# Training Guide

## Training Stages

### Stage 1: Continued Pretraining (CPT)

**Goal**: Adapt the base model's language distribution to Portuguese.

**Input**: Gemma 4 base checkpoint + Aurora-PT packed sequences
**Output**: Portuguese-adapted base model (no instruction capability yet)

Key settings:
- Learning rate: 5e-5 (full model) or 2e-4 (LoRA)
- Scheduler: Cosine with 3-5% warmup
- Sequence length: 8192 tokens (packed)
- Batch size: ~1M tokens per effective step
- Mixed precision: bfloat16
- Gradient checkpointing: enabled (saves ~40% VRAM)

```bash
# Pilot (LoRA, validates pipeline)
python3 -m src.train.cpt_trainer --config configs/train/cpt_pilot.yaml

# Full (all parameters)
python3 -m src.train.cpt_trainer --config configs/train/cpt_main.yaml
```

### Stage 2: Residual Merge (Alternative to SFT)

**Goal**: Recover instruction-following without additional training.

**Method** (Task Arithmetic):
```
instruction_residual = IT_weights - base_weights
adapted_model = CPT_weights + alpha * instruction_residual
```

**Alpha sweep**: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
- alpha < 1.0: Partial instruction recovery
- alpha = 1.0: Full residual transfer
- alpha > 1.0: Amplified instruction behavior (may degrade)

```bash
python3 -m src.train.residual_merge \
    --base-model google/gemma-4-E4B \
    --instruct-model google/gemma-4-E4B-it \
    --cpt-model outputs/cpt_main/final \
    --alpha 0.5 0.7 0.8 0.9 1.0 1.1 1.2
```

### Stage 3: Supervised Fine-Tuning (SFT)

**Goal**: Teach the CPT model to follow instructions in Portuguese.

**Input**: CPT checkpoint + Portuguese instruction dataset
**Output**: Instruction-following model

Key differences from CPT:
- Trains only on model responses (prompt tokens masked with -100)
- Uses Gemma 4 chat template format
- Much smaller dataset (10K-100K examples vs. billions of tokens)
- Higher learning rate decay, fewer epochs (1-3)

```bash
python3 -m src.train.sft_trainer --config configs/train/sft.yaml
```

### Stage 4: DPO (Optional)

**Goal**: Align model outputs with human preferences.

**Prerequisites**: Requires a preference dataset (chosen/rejected pairs) in Portuguese.
Currently no public pt-BR preference dataset exists — this stage is experimental.

```bash
python3 -m src.train.dpo_trainer --config configs/train/dpo.yaml
```

## Hardware Requirements

| Stage | Pilot (LoRA) | Full |
|-------|-------------|------|
| CPT E4B | 1x A100 80GB | 2x A100 80GB |
| CPT 26B-A4B | 2x A100 80GB | 4x A100 80GB |
| Merge | 1x CPU (64GB RAM) | 1x CPU (128GB RAM) |
| SFT | 1x A100 80GB | 1x A100 80GB |
| Eval | 1x A100 80GB | 1x A100 80GB |

## LoRA vs Full Training

| Aspect | LoRA (Pilot) | Full (Main) |
|--------|-------------|-------------|
| Parameters trained | ~0.5% | 100% |
| VRAM usage | ~20GB | ~60GB+ |
| Training time | Hours | Days |
| Quality ceiling | Good | Best |
| Use case | Validation, ablations | Final model |

## Monitoring

Training logs are written to:
- `outputs/<experiment>/train_log.jsonl` — Structured metrics per step
- Console (rich formatting) — Real-time progress
- W&B (optional) — Interactive dashboards

Key metrics to watch:
- `train_loss`: Should decrease smoothly
- `eval_loss`: Should decrease; divergence from train = overfitting
- `throughput_tokens_per_sec`: Hardware utilization
- `gpu_memory_allocated_gb`: OOM prevention

## Resuming Training

All trainers auto-detect the latest checkpoint:
```bash
# Just re-run the same command — it finds checkpoint-XXXX and resumes
python3 -m src.train.cpt_trainer --config configs/train/cpt_main.yaml
```

Or specify explicitly:
```yaml
checkpointing:
  resume_from_checkpoint: "outputs/cpt_main/checkpoint-5000"
```

## Common Issues

| Issue | Solution |
|-------|----------|
| OOM | Reduce batch size, enable gradient checkpointing |
| Loss spikes | Reduce learning rate, check data quality |
| Slow training | Increase batch size, check data loading workers |
| NaN loss | Check for inf in data, reduce LR, disable tf32 |
| Merge shape mismatch | Ensure same model family/version for all 3 checkpoints |
