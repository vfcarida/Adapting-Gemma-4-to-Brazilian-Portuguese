# Architecture Guide

## Overview

This project implements a systematic pipeline for adapting Google's Gemma 4 models
to Brazilian Portuguese. The architecture follows a strict separation between
data preparation, training, evaluation, and reporting.

```
┌─────────────────────────────────────────────────────────────────┐
│                        PIPELINE FLOW                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌─────────┐ │
│  │   DATA   │───▶│ TRAINING │───▶│   MERGE   │───▶│  EVAL   │ │
│  └──────────┘    └──────────┘    └───────────┘    └─────────┘ │
│       │                │               │               │       │
│       ▼                ▼               ▼               ▼       │
│  Aurora-PT         CPT/SFT/DPO    Task Arithmetic   Benchmarks │
│  Preprocessing     Checkpoints    Merged Models     Reports    │
│  Contamination     Metrics Logs   Alpha Sweep       Figures    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

### `src/data/` - Data Pipeline

| Module | Role |
|--------|------|
| `aurora_loader.py` | Load Aurora-PT, preprocess, split by document hash |
| `tokenizer_audit.py` | Measure tokenizer fertility on Portuguese text |
| `contamination_checks.py` | Detect overlap between training data and benchmarks |
| `replay_mix_builder.py` | Build training mixtures with English/code replay |
| `instruction_data_builder.py` | Format instruction data with Gemma 4 chat template |

### `src/train/` - Training Pipeline

| Module | Role |
|--------|------|
| `cpt_trainer.py` | Continued pretraining with packed sequences |
| `sft_trainer.py` | Supervised fine-tuning using TRL's SFTTrainer |
| `dpo_trainer.py` | DPO preference tuning (optional stage) |
| `residual_merge.py` | Task arithmetic merge for instruction recovery |
| `callbacks.py` | Custom callbacks for throughput/metrics logging |

### `src/eval/` - Evaluation Pipeline

| Module | Role |
|--------|------|
| `benchmark_runner.py` | Unified runner with caching and batch inference |
| `prompt_templates.py` | Task-specific prompts with Gemma 4 format |
| `metrics.py` | Accuracy, F1, Pearson, ROUGE-L, refusal rate |
| `bootstrap_ci.py` | Bootstrap confidence intervals and paired tests |
| `report_builder.py` | Tables, plots, and summary generation |
| `tasks/` | Per-benchmark data loading and answer parsing |

### `src/utils/` - Shared Utilities

| Module | Role |
|--------|------|
| `seed.py` | Global reproducibility (torch, numpy, random) |
| `config_utils.py` | YAML loading, merging, flattening |
| `logging_utils.py` | Console + file logging, JSONL metrics logger |
| `checkpointing.py` | Find/save/load training checkpoints |
| `hf_utils.py` | Model/tokenizer loading with quantization |

## Configuration System

All experiments are configured via YAML files in `configs/`:

```
configs/
├── data/          # Dataset sources, preprocessing, mixtures
├── model/         # Model IDs, quantization, chat template
├── train/         # Training hyperparameters (CPT, SFT, DPO)
├── eval/          # Benchmarks, models to evaluate, report settings
└── ablations/     # Full experiment matrix
```

Configs support:
- **Nested references**: `model_config: "configs/model/gemma4_e4b.yaml"` auto-resolves
- **CLI overrides**: `--override training.learning_rate=1e-5`
- **Deep merging**: Override only specific nested keys

## Design Decisions

### Why CPT on BASE, not IT?
Continued pretraining adapts the model's language distribution. Starting from the
base checkpoint ensures we learn Portuguese representations without interference
from instruction-tuning artifacts. Instruction capability is recovered separately
via residual merge or SFT.

### Why document-level splitting?
Splitting by document (using content hash) ensures no sentence from a validation
document leaks into training. This is stricter than random splitting, which could
split paragraphs of the same document across train/val.

### Why packed sequences?
Packing concatenates multiple documents into fixed-length sequences, eliminating
padding waste. For a 8192-token sequence length, this maximizes GPU utilization
and training throughput.

### Why residual merge?
Task arithmetic (`cpt_weights + alpha * (instruct - base)`) is a training-free
method to recover instruction-following from the original IT model. It's faster
than SFT and provides a strong baseline to compare against.

### Why separate think_on/think_off evaluation?
Gemma 4 supports a "thinking" mode where it reasons before answering. Different
tasks may benefit differently from this. Evaluating both modes reveals when
extended reasoning helps vs. hurts.
