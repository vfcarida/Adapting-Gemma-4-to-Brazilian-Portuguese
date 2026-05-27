# 🇧🇷 Adapting Gemma 4 to Brazilian Portuguese

> **Production-grade pipeline for computationally adapting Google Gemma 4 to Portuguese (pt-BR) via the Aurora-PT corpus (331B tokens).**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Framework: HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)

---

## 📋 Overview

This repository implements a **four-stage adaptation pipeline** for Google Gemma 4 models to Brazilian Portuguese:

```
┌─────────────┐     ┌───────────┐     ┌────────────────┐     ┌─────────┐
│  Base Model  │────▶│    CPT    │────▶│ Residual Merge │────▶│   SFT   │
│  (Gemma 4)   │     │ Aurora-PT │     │ Task Arithmetic│     │  PT-BR  │
└─────────────┘     └───────────┘     └────────────────┘     └─────────┘
```

| Stage | Method | Data Source |
|-------|--------|-------------|
| **1. Baseline** | Pure checkpoint evaluation | — |
| **2. CPT** | CausalLM next-token prediction + LoRA | Aurora-PT (331B tokens) + EN replay |
| **3. Merge** | Task Arithmetic: `cpt + α(instruct − base)` | Instruct model weights |
| **4. SFT** | TRL SFTTrainer with chat template | PT-BR instruction data |

### Target Models

| Model | Type | Active Params | Context |
|-------|------|--------------|---------|
| `google/gemma-4-E4B` | Dense (pilot) | ~4.5B | 128K |
| `google/gemma-4-26B-A4B` | MoE (main) | ~3.8B active / 26B total | 256K |

### Evaluation Baselines

- `CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it`
- `maritaca-ai/sabia-7b`
- `Polygl0t/Tucano2-qwen-3.7B-Instruct`

---

## 🚀 Quick Start

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/vfcarida/Adapting-Gemma-4-to-Brazilian-Portuguese
cd Adapting-Gemma-4-to-Brazilian-Portuguese

# Install dependencies
make install

# Configure credentials
cp .env.example .env
# Edit .env with your HF_TOKEN, WANDB_API_KEY, etc.
```

### 2. Data Quality Checks

```bash
# Tokenizer fertility analysis
make tokenizer-audit

# Three-tier contamination check (exact + normalized + MinHash)
make contamination-check
```

### 3. Training

```bash
# Pilot CPT on Gemma-4-E4B (single GPU)
make cpt-pilot

# Main CPT on Gemma-4-26B-A4B (multi-GPU + DeepSpeed)
make cpt-main

# Residual merge with alpha sweep
make merge

# Supervised fine-tuning
make sft
```

### 4. Evaluation

```bash
# Evaluate all models on 13 PT-BR benchmarks
make eval

# Generate comparison report
make report
```

### 5. Full Pipeline

```bash
make all  # End-to-end: audit → contamination → CPT → merge → SFT → eval → report
```

---

## 📁 Repository Structure

```
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── Makefile
├── .env.example
├── configs/
│   ├── cpt_pilot.yml          # E4B CPT config
│   ├── cpt_main.yml           # 26B-A4B CPT + DeepSpeed config
│   ├── sft.yml                # SFT config with label masking
│   ├── eval.yml               # Evaluation suite config
│   ├── merge.yml              # Task arithmetic merge config
│   └── ds_zero3.json          # DeepSpeed ZeRO-3 config
├── data/                      # Downloaded datasets (gitignored)
├── model/                     # Downloaded models (gitignored)
├── reports/                   # Generated reports & metrics
├── src/
│   ├── data/
│   │   ├── aurora_loader.py         # Streaming Aurora-PT with packed sequences
│   │   ├── tokenizer_audit.py       # Fertility analysis (tokens/char, tokens/word)
│   │   ├── contamination_checks.py  # 3-tier decontamination
│   │   ├── replay_mix_builder.py    # PT/EN replay mixing
│   │   └── instruction_data_builder.py  # Gemma 4 chat template formatter
│   ├── train/
│   │   ├── cpt_trainer.py     # CausalLM CPT with LoRA (NOT SFTTrainer)
│   │   ├── sft_trainer.py     # TRL SFTTrainer for instruction data
│   │   ├── residual_merge.py  # Task Arithmetic weight merging
│   │   └── callbacks.py       # JSONL logging, perplexity, early stopping
│   ├── eval/
│   │   ├── benchmark_runner.py   # Unified eval orchestrator
│   │   ├── prompt_templates.py   # Gemma 4 template + think mode
│   │   ├── metrics.py            # macro-F1, Pearson, approval rate
│   │   ├── bootstrap_ci.py       # Bootstrap CIs + paired tests
│   │   ├── report_builder.py     # Markdown report generation
│   │   └── tasks/                # 13 PT-BR benchmark definitions
│   └── utils/
│       ├── logging_utils.py   # Structured logging + JSONL writer
│       ├── seed.py            # Global reproducibility
│       ├── checkpointing.py   # Save/load + LoRA merge
│       ├── hf_utils.py        # HF auth + safe LoRA config
│       └── config_utils.py    # YAML loader + CLI factory
└── scripts/
    ├── run_tokenizer_audit.sh
    ├── run_contamination_checks.sh
    ├── run_baselines.sh
    ├── run_cpt_pilot.sh
    ├── run_cpt_main.sh
    ├── run_residual_merge.sh
    ├── run_sft.sh
    ├── run_eval.sh
    └── run_all.sh
```

---

## 📊 Evaluation Benchmarks

| Benchmark | Domain | Metric | Few-shot |
|-----------|--------|--------|----------|
| ENEM | Education (national exam) | Approval Rate | 3 |
| BluEx | Education (university entrance) | Approval Rate | 3 |
| OAB-Bench | Legal (bar exam) | Approval Rate | 3 |
| ASSIN2-RTE | NLI (textual entailment) | macro-F1 | 15 |
| ASSIN2-STS | Semantic similarity | Pearson r | 15 |
| HateBR | Hate speech detection | macro-F1 | 25 |
| TweetSentBR | Sentiment analysis | macro-F1 | 25 |
| COPA-PT | Causal reasoning | Accuracy | 0 |
| BRoverbs | Proverb completion | Accuracy | 5 |
| MRPC-PT | Paraphrase detection | macro-F1 | 5 |
| RTE-PT | Textual entailment | Accuracy | 15 |
| DoNotAnswer-PT | Safety / refusal | Refusal Rate | 0 |
| TugueSICE-PT | Language understanding | Accuracy | 5 |

All evaluations run in both **think_on** and **think_off** modes with temperature=0.0.

---

## ⚙️ Key Design Decisions

### Golden Rule: Aurora-PT Data Handling
> Aurora-PT is unstructured text. It is **never** used with `SFTTrainer`. All Aurora-PT training uses standard `CausalLM` next-token prediction with packed sequences.

### LoRA Safety on Gemma 4
Gemma 4 contains `Gemma4ClippableLinear` layers in vision/audio towers. We **never** use `target_modules="all-linear"`. Instead, we whitelist only language model projections:
```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### Task Arithmetic Formula
```
inst_residual    = instruct_weights − base_weights
adapted_instruct = cpt_weights     + (α × inst_residual)
```

---

## 📝 Requirements

- Python ≥ 3.10
- CUDA-capable GPU (A100-80GB recommended for E4B; multi-GPU for 26B-A4B)
- HuggingFace account with access to Gemma 4 models and Aurora-PT dataset
- Weights & Biases account (optional, for experiment tracking)

---

## 📜 License

Apache 2.0
