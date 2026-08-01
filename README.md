<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.6+-ee4c2c?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-5.5+-yellow?logo=huggingface" alt="Transformers">
  <img src="https://img.shields.io/badge/Tests-253%20passed-green?logo=pytest" alt="Tests">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="License">
</p>

# 🇧🇷 Adapting Gemma 4 to Brazilian Portuguese

> Systematic adaptation of Google's Gemma 4 models to Brazilian Portuguese using Continued Pre-Training (CPT) on the Aurora-PT corpus (~331B tokens), with rigorous evaluation across 20+ benchmarks.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#-quick-start)
- [Google Colab (Single-GPU Path)](#-google-colab-single-gpu-path)
- [GCP Deployment (Step-by-Step)](#-gcp-deployment-step-by-step)
- [Project Structure](#-project-structure)
- [Pipeline Stages](#-pipeline-stages)
- [Configuration System](#-configuration-system)
- [Evaluation Suite](#-evaluation-suite)
- [CLI Reference](#-cli-reference)
- [Testing](#-testing)
- [Models & Baselines](#-models--baselines)
- [Research Background](#-research-background)
- [Reproducibility](#-reproducibility)
- [Documentation Map](#-documentation-map)
- [Hardware Requirements](#-hardware-requirements)
- [Troubleshooting](#-troubleshooting)

---

## Overview

This repository implements a complete, production-ready pipeline for adapting Google's Gemma 4 large language models to Brazilian Portuguese. The approach combines:

1. **Continued Pre-Training (CPT)** on the Aurora-PT corpus with English replay to prevent catastrophic forgetting
2. **Residual Merge** (Task Arithmetic) to recover instruction-following without additional training
3. **Supervised Fine-Tuning (SFT)** as an alternative instruction recovery path
4. **Rigorous Evaluation** with bootstrap confidence intervals across 20+ Portuguese benchmarks

The project follows scientific best practices from recent CPT research (Sabiá, Tucano, Biderman et al. 2024, Ibrahim et al. 2024). Two hardware paths are supported: **a single-GPU Google Colab notebook** (`colab/Gemma4_PTBR_Colab.ipynb` — QLoRA pilot on Gemma 4 E2B, the recommended way to run this project end-to-end without provisioning cloud infrastructure) and **Google Cloud Platform (GCP) with 1-4x A100/H100** for full-scale runs (E4B/26B-A4B, full fine-tune, the complete 20+ benchmark suite).

### Scientific Objectives

| Hypothesis | What we test |
|------------|-------------|
| H1 | CPT on Aurora-PT improves Portuguese benchmarks |
| H2 | English replay (10-15%) prevents catastrophic forgetting |
| H3 | Residual merge recovers instruction-following without SFT |
| H4 | CPT + SFT > CPT + Residual Merge overall |
| H5 | Think mode improves complex reasoning (ENEM, OAB) |
| H6 | DoRA ≥ LoRA for CPT adaptation |
| H7 | Higher LoRA rank (r=128) outperforms r=64 for CPT |
| H8 | 50B tokens > 20B tokens (scaling law) |

---

## 🚀 Quick Start

### Local Setup (CPU — tests, validation, development)

```bash
# 1. Clone the repository
git clone https://github.com/vfcarida/Adapting-Gemma-4-to-Brazilian-Portuguese.git
cd Adapting-Gemma-4-to-Brazilian-Portuguese

# 2. Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Run full test suite (253 tests, ~4s, no GPU needed)
pytest tests/ -q

# 5. Validate environment readiness
bash scripts/preflight.sh

# 6. (Optional) Smoke test
gemma4pt smoke
```

### GPU Setup (Training & Evaluation)

```bash
# Install with GPU dependencies
pip install -e ".[gpu,eval]"

# Validate GPU and VRAM
bash scripts/preflight.sh --config configs/train/cpt_pilot.yaml

# Launch pilot training
gemma4pt train-cpt configs/train/cpt_pilot.yaml
```

---

## 🔬 Google Colab (Single-GPU Path)

**This is the recommended way to train and evaluate a model end-to-end without any cloud setup.** Open [`colab/Gemma4_PTBR_Colab.ipynb`](colab/Gemma4_PTBR_Colab.ipynb) in Google Colab and run it cell by cell — it covers the complete pipeline on a single GPU (T4 16GB free tier, L4 24GB, or A100 40GB):

1. Environment setup (correct dependency versions, without touching Colab's preinstalled PyTorch/CUDA)
2. QLoRA Continued Pre-Training pilot on **Gemma 4 E2B** (`google/gemma-4-E2B` — Apache 2.0, not gated, the smallest real Gemma 4 model) over the **Aurora-PT** corpus with English replay
3. Residual merge (task arithmetic) to recover instruction-following without additional training
4. Evaluation on a fast subset of real Portuguese benchmarks (ENEM, BLUEX, ASSIN2-RTE, HateBR, OAB) + English retention (MMLU), with bootstrap/Wilson confidence intervals
5. Results dashboard, generated inline

See [`colab/README.md`](colab/README.md) for the full walkthrough, VRAM/session-time expectations, and how to persist checkpoints across sessions (Colab has no persistent disk — checkpoints push to your HF Hub). The Colab-specific configs are [`configs/train/cpt_colab_pilot.yaml`](configs/train/cpt_colab_pilot.yaml) and [`configs/eval/benchmarks_colab.yaml`](configs/eval/benchmarks_colab.yaml).

To scale beyond a single GPU (larger models, the full 20+ benchmark suite, multi-GPU full fine-tune), continue to the GCP deployment guide below.

---

## ☁️ GCP Deployment (Step-by-Step)

This section provides a complete guide to deploy and run the full pipeline on Google Cloud Platform.

### Prerequisites

| Requirement | How to get it |
|-------------|---------------|
| GCP Account with billing | [console.cloud.google.com](https://console.cloud.google.com) |
| `gcloud` CLI installed | `curl https://sdk.cloud.google.com \| bash` |
| GPU quota (A100) | Console → IAM & Admin → Quotas → search "A100" → Request increase |
| HuggingFace token | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — needs Gemma access |
| (Optional) W&B account | [wandb.ai](https://wandb.ai) for experiment tracking |

### Step 1: Configure Environment Variables

```bash
# Copy the template and fill in your values
cp infra/gcp/ENV_TEMPLATE.sh .env.gcp

# Edit with your GCP project ID, zone, bucket name
nano .env.gcp

# Source it
source .env.gcp
```

**Required variables in `.env.gcp`:**
```bash
export GCP_PROJECT_ID="your-gcp-project-id"
export GCP_REGION="us-central1"
export GCP_ZONE="us-central1-a"
export GCS_BUCKET="gs://your-bucket-name"
export HF_TOKEN="hf_your_token_here"
export WANDB_API_KEY="your_wandb_key"  # optional
```

### Step 2: GCP Project Setup (One-time)

```bash
# This script creates: GCS bucket, Secret Manager secrets, enables APIs
./infra/gcp/setup_project.sh

# Verify GPU availability in your zone
gcloud compute accelerator-types list --filter="zone:${GCP_ZONE}" | grep -E "a100|h100"
```

### Step 3: Create GPU Instance

```bash
# For pilot experiments (1x A100 80GB, Spot VM — ~$1.50/hr)
./infra/gcp/create_instance.sh pilot

# For main training (4x A100 80GB, Spot VM — ~$6/hr)
./infra/gcp/create_instance.sh main
```

The startup script (`infra/gcp/startup_script.sh`) automatically:
- Installs NVIDIA drivers
- Clones this repository to `/workspace/repo`
- Installs Python dependencies
- Configures HF and W&B credentials from Secret Manager
- Mounts local SSD for fast I/O

### Step 4: Connect and Run

```bash
# SSH into the instance
gcloud compute ssh gemma4-pt-br-pilot --zone=${GCP_ZONE}

# Inside the VM:
cd /workspace/repo

# Run preflight checks (validates GPU, VRAM, disk, credentials)
bash scripts/preflight.sh --config configs/train/cpt_pilot.yaml

# Launch training in a tmux session (survives SSH disconnection)
tmux new-session -s training
gemma4pt train-cpt configs/train/cpt_pilot.yaml
# Ctrl+B, D to detach

# Or submit as background job from your local machine:
./infra/gcp/submit_training_job.sh cpt_pilot
```

### Step 5: Monitor Training

```bash
# Re-attach to training session
gcloud compute ssh gemma4-pt-br-pilot --zone=${GCP_ZONE} -- 'tmux attach -t training'

# Check GPU utilization
gcloud compute ssh gemma4-pt-br-pilot --zone=${GCP_ZONE} -- 'nvidia-smi'

# View training logs
gcloud compute ssh gemma4-pt-br-pilot --zone=${GCP_ZONE} -- 'tail -f /workspace/repo/outputs/cpt_pilot/train_log.jsonl'

# W&B dashboard (if configured): https://wandb.ai/your-project
```

### Step 6: Run Evaluation

```bash
# After training completes:
gemma4pt eval --config configs/eval/benchmarks.yaml

# Generate results dashboard
python3 scripts/build_dashboard.py --format markdown

# Download results to local machine
./infra/gcp/sync_checkpoints.sh download-results
```

### Step 7: Cleanup (Save Costs!)

```bash
# Sync checkpoints to GCS before stopping
./infra/gcp/sync_checkpoints.sh upload

# Stop instance (preserves disk, stops billing for compute)
./infra/gcp/stop_and_cleanup.sh stop

# Or delete everything when done
./infra/gcp/stop_and_cleanup.sh delete
```

### Cost Estimates (Spot Pricing, us-central1)

| Phase | Hardware | Duration | Estimated Cost |
|-------|----------|----------|----------------|
| Pilot (1 variant) | 1x A100 80GB | 24h | ~$50 |
| Pilot (7 variants parallel) | 7x A100 | 24h | ~$350 |
| Main CPT (20B tokens) | 4x A100 80GB | 3-5 days | ~$2,000 |
| Main CPT (50B tokens) | 4x A100 80GB | 7-12 days | ~$5,000 |
| Evaluation (full suite) | 1x A100 | 12-24h | ~$50-100 |
| Residual Merge sweep | 1x A100 | 2h | ~$10 |
| **Total recommended** | | | **$3,000–$8,000** |

---

## 📁 Project Structure

```
gemma4-pt-br-adaptation/
│
├── configs/                      # All experiment configurations (YAML)
│   ├── data/                     #   Dataset sources, preprocessing, mixtures
│   │   └── aurora_pt.yaml        #   Aurora-PT corpus configuration
│   ├── model/                    #   Model architecture, quantization
│   │   ├── gemma4_e2b.yaml       #   Gemma 4 E2B — single-GPU/Colab config
│   │   └── gemma4_e4b.yaml       #   Gemma 4 E4B model config
│   ├── train/                    #   Training hyperparameters
│   │   ├── cpt_colab_pilot.yaml  #   Colab: QLoRA on E2B, single GPU
│   │   ├── cpt_pilot.yaml        #   Pilot: LoRA r=128, 5B tokens
│   │   ├── cpt_main.yaml         #   Main: Full FT, 20-50B tokens
│   │   ├── sft.yaml              #   Supervised fine-tuning
│   │   ├── dpo.yaml              #   DPO preference tuning
│   │   ├── lr_sweep.yaml         #   Learning rate sweep
│   │   └── ablation_packing.yaml #   Packing strategy ablation
│   └── eval/                     #   Evaluation benchmarks & settings
│       ├── benchmarks.yaml       #   Full benchmark suite configuration
│       └── benchmarks_colab.yaml #   Fast ~6-benchmark subset for Colab
│
├── src/                          # Source code
│   ├── cli.py                    #   CLI entry point (typer-based)
│   ├── preflight.py              #   Environment validation module
│   ├── data/                     #   Data pipeline
│   │   ├── aurora_loader.py      #     Load, preprocess, pack sequences
│   │   ├── tokenizer_audit.py    #     Tokenizer fertility analysis
│   │   ├── contamination_checks.py #   Train↔Eval overlap detection
│   │   ├── replay_mix_builder.py #     English/code replay mixing
│   │   └── instruction_data_builder.py # SFT data formatting
│   ├── train/                    #   Training pipeline
│   │   ├── cpt_trainer.py        #     Continued pre-training orchestrator
│   │   ├── sft_trainer.py        #     Supervised fine-tuning trainer
│   │   ├── dpo_trainer.py        #     DPO preference training
│   │   ├── residual_merge.py     #     Task arithmetic model merging
│   │   └── callbacks.py          #     Monitoring & GCS sync callbacks
│   ├── eval/                     #   Evaluation pipeline
│   │   ├── benchmark_runner.py   #     Unified benchmark execution
│   │   ├── prompt_templates.py   #     Gemma 4 format prompts
│   │   ├── metrics.py            #     Accuracy, F1, ROUGE-L, STS
│   │   ├── bootstrap_ci.py       #     Statistical confidence intervals
│   │   ├── report_builder.py     #     Tables, plots, summaries
│   │   └── tasks/                #     Per-benchmark data loaders
│   │       └── base_task.py      #       Abstract task interface
│   └── utils/                    #   Shared utilities
│       ├── config_utils.py       #     YAML loading & resolution
│       ├── hf_utils.py           #     Model/tokenizer loading + VRAM estimation
│       ├── logging_utils.py      #     Structured logging (console + JSONL)
│       ├── seed.py               #     Reproducibility (all RNGs)
│       └── checkpointing.py      #     Checkpoint management
│
├── tests/                        # Test suite (253 tests)
│   ├── test_integration_pipeline.py  # End-to-end pipeline tests
│   ├── test_gemma4_compliance.py     # Gemma 4 format compliance
│   ├── test_golden.py                # Deterministic fixture tests
│   ├── test_bootstrap.py            # Statistical method tests
│   ├── test_contamination.py        # Contamination detection tests
│   └── fixtures/                     # Test data & golden outputs
│
├── scripts/                      # Operational scripts
│   ├── preflight.sh              #   Pre-training environment validation
│   ├── build_dashboard.py        #   Results visualization
│   ├── run_data_qc.sh            #   Data quality checks
│   ├── run_tokenizer_audit.sh    #   Tokenizer fertility audit
│   └── run_contamination_checks.sh # Benchmark contamination checks
│
├── infra/gcp/                    # GCP infrastructure automation
│   ├── QUICKSTART.md             #   GCP deployment guide
│   ├── ENV_TEMPLATE.sh           #   Environment variables template
│   ├── create_instance.sh        #   Create GPU VMs (pilot/main/large)
│   ├── startup_script.sh         #   VM auto-setup on boot
│   ├── setup_project.sh          #   One-time GCP project setup
│   ├── submit_training_job.sh    #   Submit training as background job
│   ├── sync_checkpoints.sh       #   GCS checkpoint sync
│   └── stop_and_cleanup.sh       #   Cost management & cleanup
│
├── docs/                         # Technical documentation
│   ├── ARCHITECTURE.md           #   System design & decisions
│   ├── TRAINING_GUIDE.md         #   Training stages guide
│   ├── DATA_PIPELINE.md          #   Data processing pipeline
│   ├── EVAL_PROTOCOL.md          #   Evaluation methodology
│   ├── EXPERIMENT_PLAN.md        #   11-step experiment protocol
│   ├── GEMMA4_COMPLIANCE.md      #   Gemma 4 format compliance
│   ├── CPT_BEST_PRACTICES_RESEARCH.md # Literature review
│   ├── experiment_card_template.md    # Experiment documentation template
│   └── CONTRIBUTING.md           #   Contribution guidelines
│
├── reports/                      # Generated reports & results
│   ├── TECHNICAL_REPORT.md       #   Technical report
│   ├── pretrain_readiness_report.md # Pre-training readiness
│   ├── dashboard.md              #   Results dashboard
│   └── test_matrix.md            #   Test coverage matrix
│
├── colab/                        # Google Colab single-GPU path (recommended)
│   ├── Gemma4_PTBR_Colab.ipynb   #   End-to-end notebook: train → merge → eval
│   └── README.md                 #   Walkthrough, VRAM/session expectations
│
├── pyproject.toml                # Package configuration
├── requirements.txt              # Core dependencies
└── EXECUTIVE_SUMMARY.md          # Executive project summary
```

---

## 🔄 Pipeline Stages

The project implements a 4-stage pipeline, each independently executable:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  STAGE 1     │    │  STAGE 2     │    │  STAGE 3     │    │  STAGE 4     │
│  DATA PREP   │───▶│  TRAINING    │───▶│  MERGE/SFT   │───▶│  EVALUATION  │
│              │    │              │    │              │    │              │
│ • Load corpus│    │ • CPT (LoRA) │    │ • Residual   │    │ • 20+ bench  │
│ • Filter/QC  │    │ • CPT (Full) │    │   Merge      │    │ • Bootstrap  │
│ • Pack seqs  │    │ • Monitor    │    │ • Alpha sweep│    │ • Reports    │
│ • Mix replay │    │ • Checkpoint │    │ • SFT (alt)  │    │ • Dashboard  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### Stage 1: Data Preparation

```bash
# Quality checks on Aurora-PT corpus
bash scripts/run_data_qc.sh

# Tokenizer fertility audit (how many Gemma 4 tokens per Portuguese word)
bash scripts/run_tokenizer_audit.sh

# Check for train↔eval data contamination
bash scripts/run_contamination_checks.sh
```

**Key features:**
- Document-level splitting (no cross-document leakage)
- EOS separators between packed documents
- Cross-document label masking (optional)
- Curriculum sort (shorter docs first, optional)
- English replay mixing (configurable 5-20%)

### Stage 2: Training

```bash
# Pilot: LoRA r=128 on 1x A100, ~24h
gemma4pt train-cpt configs/train/cpt_pilot.yaml

# Main: Full fine-tune on 4x A100, 3-7 days
gemma4pt train-cpt configs/train/cpt_main.yaml
```

**Key features:**
- Automatic checkpoint sync to GCS (survives Spot preemption)
- English perplexity monitoring (forgetting detection)
- DeepSpeed ZeRO-2/3 for multi-GPU
- Gradient checkpointing (~40% VRAM savings)
- Automatic resume from latest checkpoint

### Stage 3: Instruction Recovery

```bash
# Option A: Residual Merge (training-free, fast)
python3 -m src.train.residual_merge \
    --base-model google/gemma-4-E4B \
    --instruct-model google/gemma-4-E4B-it \
    --cpt-model outputs/cpt_pilot/final \
    --alpha 0.5 0.7 0.8 0.9 1.0

# Option B: SFT (requires instruction dataset)
gemma4pt train-sft configs/train/sft.yaml
```

### Stage 4: Evaluation

```bash
# Run full benchmark suite
gemma4pt eval --config configs/eval/benchmarks.yaml

# Generate comparison dashboard
python3 scripts/build_dashboard.py --format markdown

# Generate full report with figures
python3 -m src.eval.report_builder
```

---

## ⚙️ Configuration System

All experiments are driven by YAML configurations with automatic resolution:

```yaml
# configs/train/cpt_pilot.yaml
experiment:
  name: "cpt_pilot_e4b_lora"
  seed: 42

model_config: "configs/model/gemma4_e4b.yaml"   # Auto-resolved to dict
data_config: "configs/data/aurora_pt.yaml"       # Auto-resolved to dict

training:
  use_lora: true
  learning_rate: 2.0e-4
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 16
  # Effective batch: 2 × 16 × 8192 = ~262K tokens/step

lora:
  r: 128           # Research-backed: Biderman 2024 recommends r=128+ for CPT
  lora_alpha: 256
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

**Features:**
- Nested config references auto-resolve (string paths → loaded dicts)
- CLI overrides: `--override training.learning_rate=1e-5`
- Deep merging: override only specific nested keys

---

## 📊 Evaluation Suite

### 20+ Benchmarks Across 7 Groups

Every `hub_id` below is a **real, verified** HuggingFace dataset (verified live against the Hub — see `configs/eval/benchmarks.yaml` for exact IDs, splits, and the license/gating notes). `capitu` and `donotanswer_pt` are disabled by default (no verified public dataset exists yet — see the comments in that file for the closest alternatives). `math_pt` is enabled, backed by `tiagoteixeira03/MATH-PT` (`ptbr_multiple_choice` config).

| Group | Benchmarks | What it measures |
|-------|-----------|-----------------|
| **Brasil Geral** | ENEM 2022/23/24 (`maritaca-ai/enem`), BLUEX (`eduagarcia-temp/BLUEX_without_images`) | General knowledge in Portuguese |
| **Semântica** | ASSIN2-RTE/STS (`nilc-nlp/assin2`), CoPA-PT/BoolQ-PT/MRPC-PT/RTE-PT (`PORTULAN/extraglue`) | Language understanding |
| **Classificação** | HateBR (`ruanchaves/hatebr`), TweetSentBR (`eduagarcia/tweetsentbr_fewshot`) | Text classification |
| **Jurídico** | OAB-Bench (`eduagarcia/oab_exams`), LeNER-Br (`peluz/lener_br`), LegalBench-BR (`eduagarcia/portuguese_benchmark:brazilian_court_decisions_judgment`) | Legal domain |
| **Raciocínio** | Math-PT (`tiagoteixeira03/MATH-PT`) | Math word problems |
| **Cultura** | BRoverbs (`Tropic-AI/BRoverbs`) | Brazilian cultural knowledge |
| **Domínio Público** | PublicHearing-BR (`unicamp-dl/PublicHearingBR`) | Public-hearing summarization |
| **Retenção EN** | MMLU (`cais/mmlu`), HellaSwag (`Rowan/hellaswag`), ARC (`allenai/ai2_arc`) | English capability retention |
| **Exploratório** | XL-Sum-PT (`csebuetnlp/xlsum`) | Summarization |

### Statistical Rigor

- **Confidence intervals**: [Wilson score interval](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval) for single-model accuracy (holds ~95% coverage even on small exam-style benchmarks like ENEM's 180 items/year — see `docs/CPT_BEST_PRACTICES_RESEARCH.md` for why plain CLT/bootstrap under-cover at this scale, per Bowyer et al. 2025); item-resampling **percentile bootstrap (10,000 resamples)** for metrics that aren't a simple accuracy (Pearson, macro-F1, entity-F1) — see `src/eval/bootstrap_ci.py`.
- **Paired comparisons**: `paired_bootstrap_test` resamples the SAME items for both models each draw and reports the score difference + its bootstrap CI (significant iff the CI excludes zero) — not a simple win-rate.
- **Correction**: Holm-Bonferroni for multiple comparisons (`src/eval/stats_tests.py`) — pre-register a small primary benchmark family before applying this across 20+ benchmarks × 2 think-modes, or power drops to near zero.
- **Dual scoring**: Generation (default) or logprob (teacher-forced multi-token option scoring, `evaluation.use_logprob: true`) — see `src/eval/benchmark_runner.py`.
- **Think mode**: Evaluated with `think_modes: ["off", "on"]`, using Gemma 4's real `enable_thinking` chat-template parameter (not a hand-appended string — see `docs/GEMMA4_COMPLIANCE.md`).
- **Per-item results**: every prediction (prompt hash, raw output, parsed prediction, gold label, correctness) is persisted to `outputs/eval_cache/<key>_items.jsonl`, not just a metric summary — enabling post-hoc paired analysis.

---

## 💻 CLI Reference

```bash
gemma4pt --help                    # Show all commands

# Environment & Validation
gemma4pt preflight                 # Validate GPU, VRAM, credentials, configs
gemma4pt smoke                     # Quick E2E sanity check (CPU-safe)
gemma4pt data-validate             # Validate data pipeline
gemma4pt contamination-check       # Check train↔eval contamination
gemma4pt tokenizer-audit           # Report tokenizer fertility

# Training
gemma4pt train-cpt CONFIG          # Continued Pre-Training
gemma4pt train-sft CONFIG          # Supervised Fine-Tuning
gemma4pt merge                     # Residual merge with alpha sweep

# Evaluation & Reporting
gemma4pt eval                      # Run benchmark suite
gemma4pt report                    # Generate reports and figures
gemma4pt manifest                  # Save reproducibility manifest

# Meta
gemma4pt run-all                   # Full pipeline (data → train → eval → report)
```

**Global flags:** `--dry-run`, `--tiny` (minimal data), `--cpu-only`

---

## 🧪 Testing

```bash
# Full suite (253 tests, ~4s, no GPU required)
pytest tests/ -q

# By category
pytest tests/test_integration_pipeline.py -v    # Pipeline wiring
pytest tests/test_gemma4_compliance.py -v       # Gemma 4 format
pytest tests/test_golden.py -v                  # Deterministic fixtures
pytest tests/test_bootstrap.py -v               # Statistical methods
pytest tests/test_contamination.py -v           # Contamination detection
pytest tests/test_metrics.py -v                 # Evaluation metrics
pytest tests/test_data_pipeline.py -v           # Data processing

# Smoke test (13 checks, validates all components)
gemma4pt smoke
```

**Test philosophy:** All tests run on CPU in <3s. No GPU, no network, no HuggingFace Hub calls. Tests validate logic, not model quality.

---

## 🤖 Models & Baselines

### Target Models

| Role | Model ID | Parameters |
|------|----------|-----------|
| Pilot base | `google/gemma-4-E4B` | ~4B active |
| Pilot instruct | `google/gemma-4-E4B-it` | ~4B active |
| Scale base | `google/gemma-4-26B-A4B` | ~4B active (MoE) |
| Scale instruct | `google/gemma-4-26B-A4B-it` | ~4B active (MoE) |

### External Baselines

| Model | Description |
|-------|-------------|
| `CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it` | Previous Portuguese adaptation (CPT of Gemma 3 4B, ~13B tokens — see `docs/CPT_BEST_PRACTICES_RESEARCH.md`) |
| `maritaca-ai/sabia-7b` | Maritaca AI's Portuguese model (base, non-chat — evaluated with `is_chat_model: false`) |
| `TucanoBR/Tucano-2-7b-instruct` | PUCRS Portuguese model (disabled by default in `configs/eval/benchmarks.yaml` — enable if needed) |

All model and dataset IDs in this repo (`configs/model/*.yaml`, `configs/eval/benchmarks.yaml`) were verified live against the HuggingFace Hub API as of this writing. HF Hub content can still change — re-verify with `huggingface_hub.HfApi().model_info(...)` / `dataset_info(...)` before a long-running or expensive job if it's been a while.

---

## 📚 Research Background

This project's methodology is informed by recent research:

| Paper | Key Insight Applied |
|-------|-------------------|
| Biderman et al. (2024) "LoRA Learns Less and Forgets Less" | LoRA needs an LR an order of magnitude above full fine-tune's (~5e-5 to 5e-4) to converge — this holds for both CPT and instruction-tuning regimes the paper tests, not just one; full fine-tune generally learns more than even high-rank LoRA at CPT scale |
| Ibrahim et al. (2024) "Simple Strategies for CPT" | ~5% replay for a same-domain shift, ~25% for a new-language shift (English→German in their study, the closest analog to English→Portuguese here) |
| Sabiá (Maritaca, 2023) | ~10B PT tokens CPT on LLaMA → large ENEM/Poeta gains at small English cost |
| Tucano (PUCRS, 2024) | Most PT benchmarks don't scale monotonically with CPT tokens — only CALAME-PT/LAMBADA-PT/ARC-PT/HellaSwag-PT reliably do; use those as in-training progress signals |
| Ilharco et al. (2023) "Task Arithmetic" | Residual merge with α=0.5-1.0 for CPT |
| Huang et al. (2024) "Chat Vector" | `θ_target = θ_CPT + α·(θ_instruct − θ_base)`, α=0.5-1.0 — same formula as residual merge here, applied specifically to recover chat behavior post-CPT |
| Yadav et al. (2023) "TIES-Merging" | k=20%, α=1.0 when combining multiple task vectors (not currently implemented — single-vector residual merge only) |

**Full literature review:** [`docs/CPT_BEST_PRACTICES_RESEARCH.md`](docs/CPT_BEST_PRACTICES_RESEARCH.md)

---

## 🔬 Reproducibility

Every experiment is fully reproducible:

- **Fixed seeds**: Default 42 (configurable) across random/numpy/torch/CUDA; full bit-for-bit determinism (`torch.use_deterministic_algorithms`) is opt-in via `set_seed(seed, full_determinism=True)` — it measurably slows training, so it's off by default and meant for debugging/golden tests, not full training runs
- **Versioned configs**: All hyperparameters in committed YAML files
- **Run manifests**: Git SHA, package versions, resolved configs saved per run
- **Inference caching**: Eval results cached by a key covering model + benchmark + think_mode + seed + num_shots + generation settings + dataset config (see `BenchmarkRunner._cache_key`) — editing any of these invalidates the cache, rather than silently reusing stale results
- **Confidence intervals**: 95% Wilson score (accuracy) or bootstrap (other metrics), not just point estimates — see Evaluation Suite above
- **Document-level splits**: Content hash → deterministic train/val assignment
- **Experiment cards**: Template at [`docs/experiment_card_template.md`](docs/experiment_card_template.md)

---

## 📖 Documentation Map

| Document | What you'll find |
|----------|-----------------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System design, module responsibilities, design decisions |
| [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md) | Training stages (CPT → Merge → SFT → DPO), hardware, monitoring |
| [`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md) | Data loading, preprocessing, packing, replay mixing |
| [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md) | Evaluation methodology, scoring modes, statistical tests |
| [`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md) | 11-step experimental protocol with acceptance criteria |
| [`docs/GEMMA4_COMPLIANCE.md`](docs/GEMMA4_COMPLIANCE.md) | Gemma 4 chat template, think mode, multi-turn format |
| [`docs/CPT_BEST_PRACTICES_RESEARCH.md`](docs/CPT_BEST_PRACTICES_RESEARCH.md) | Literature review (20+ papers) |
| [`colab/README.md`](colab/README.md) | **Google Colab single-GPU path** — the recommended way to run this project end-to-end |
| [`infra/gcp/QUICKSTART.md`](infra/gcp/QUICKSTART.md) | GCP deployment guide with cost estimates (for scaling beyond a single GPU) |
| [`reports/TECHNICAL_REPORT.md`](reports/TECHNICAL_REPORT.md) | Technical report with methodology |

---

## 🖥️ Hardware Requirements

| Task | Minimum | Recommended |
|------|---------|-------------|
| Tests, smoke, preflight | CPU only | Any machine |
| CPT Colab Pilot (E2B, QLoRA) | 1× T4 16GB (Colab free tier) | 1× L4 24GB / A100 40GB (Colab Pro/Pro+) |
| CPT Pilot (E4B, LoRA) | 1× A100 40GB | 1× A100 80GB |
| CPT Main (26B, Full FT) | 4× A100 80GB | 4× A100 80GB (ZeRO-2) |
| Evaluation | 1× A100 40GB (or 1× T4/L4 for the Colab benchmark subset) | 1× A100 80GB |
| Residual Merge | CPU (64GB RAM) | CPU (128GB RAM) |
| Full pipeline (single GPU, Colab) | 1× T4 16GB | 1× A100 40GB |
| Full pipeline (GCP, multi-GPU) | 1× A100 80GB | 4× A100 80GB |

**VRAM estimation** is built-in:
```bash
bash scripts/preflight.sh --config configs/train/cpt_pilot.yaml
# Output: Model 8.0 GB + Optimizer 0.5 GB + Activations 2.8 GB = ~11.7 GB total
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `flash-attn not installed` | Automatic fallback to SDPA (PyTorch native). Or: `pip install flash-attn` (on Colab, prefer `attn_implementation: "sdpa"` in the model config — building flash-attn in an ephemeral session is slow/fragile) |
| OOM during training | Reduce `per_device_train_batch_size` or enable gradient checkpointing |
| Spot VM preempted (GCP) | Checkpoints auto-sync to GCS every 200 steps. Just restart the VM. |
| Colab session disconnected | Checkpoints push to your HF Hub if `output.push_to_hub: true` is set — resume via `snapshot_download` + rerun the same training command (see `colab/README.md`) |
| `model_config is required` | Ensure your training config has `model_config: "configs/model/..."` |
| Gemma 4 models fail to load | `google/gemma-4-*` are Apache 2.0 and NOT gated — no `huggingface-cli login` needed to download; a token is only required to `push_to_hub` your own checkpoints |
| Tests fail with `ModuleNotFoundError` | Run `pip install -e ".[dev]"` from project root |
| NaN in loss | Reduce learning rate, check data quality, disable tf32 |
| Slow data loading | Increase `dataloader_num_workers` (4-8 for A100; 2 on Colab, fewer CPU cores available) |

---

## 📄 License

Apache 2.0 — See LICENSE file.

---

## Citation

If you use this work, please cite:

```bibtex
@software{carida2025gemma4ptbr,
  title={Adapting Gemma 4 to Brazilian Portuguese},
  author={Caridá, Vinícius F.},
  year={2025},
  url={https://github.com/vfcarida/Adapting-Gemma-4-to-Brazilian-Portuguese}
}
```
