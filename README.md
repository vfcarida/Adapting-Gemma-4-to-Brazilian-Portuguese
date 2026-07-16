<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.2+-ee4c2c?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-4.45+-yellow?logo=huggingface" alt="Transformers">
  <img src="https://img.shields.io/badge/Tests-216%20passed-green?logo=pytest" alt="Tests">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="License">
</p>

# 🇧🇷 Adapting Gemma 4 to Brazilian Portuguese

> Systematic adaptation of Google's Gemma 4 models to Brazilian Portuguese using Continued Pre-Training (CPT) on the Aurora-PT corpus (~331B tokens), with rigorous evaluation across 20+ benchmarks.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#-quick-start)
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

The project follows scientific best practices from recent CPT research (Sabiá, Tucano, Biderman et al. 2024, Ibrahim et al. 2024) and is designed to run on Google Cloud Platform (GCP) with A100/H100 GPUs.

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

# 4. Run full test suite (216 tests, ~3s, no GPU needed)
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
│   │   └── gemma4_e4b.yaml       #   Gemma 4 E4B model config
│   ├── train/                    #   Training hyperparameters
│   │   ├── cpt_pilot.yaml        #   Pilot: LoRA r=128, 5B tokens
│   │   ├── cpt_main.yaml         #   Main: Full FT, 20-50B tokens
│   │   ├── sft.yaml              #   Supervised fine-tuning
│   │   ├── dpo.yaml              #   DPO preference tuning
│   │   ├── lr_sweep.yaml         #   Learning rate sweep
│   │   └── ablation_packing.yaml #   Packing strategy ablation
│   └── eval/                     #   Evaluation benchmarks & settings
│       └── benchmarks.yaml       #   Full benchmark suite configuration
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
├── tests/                        # Test suite (216 tests)
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

### 20+ Benchmarks Across 6 Groups

| Group | Benchmarks | What it measures |
|-------|-----------|-----------------|
| **Brasil Geral** | ENEM 2022/23/24, BLUEX | General knowledge in Portuguese |
| **Semântica** | ASSIN2-RTE, ASSIN2-STS, CoPA-PT, MRPC-PT, RTE-PT | Language understanding |
| **Classificação** | HateBR, TweetSentBR | Text classification |
| **Jurídico** | OAB-Bench, LegalBench-BR, LeNER-Br | Legal domain |
| **Cultura** | BRoverbs, CAPITU | Brazilian cultural knowledge |
| **Retenção EN** | MMLU, HellaSwag, ARC | English capability retention |
| **Segurança** | DoNotAnswer-PT | Safety alignment |

### Statistical Rigor

- **Bootstrap CI**: 10,000 resamples, BCa method, 95% confidence intervals
- **Paired tests**: Same questions compared across models
- **Correction**: Holm-Bonferroni for multiple comparisons
- **Dual scoring**: Generation (primary) + logprob (secondary)
- **Think mode**: Evaluated with `think_on` and `think_off` for all reasoning tasks

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
# Full suite (216 tests, ~3s, no GPU required)
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
| `CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it` | Previous Portuguese adaptation |
| `maritaca-ai/sabia-7b` | Maritaca AI's Portuguese model |
| Tucano 2 Instruct | PUCRS Portuguese model |

> ⚠️ **Important**: Verify actual model IDs on HuggingFace before launching training. Gemma 4 naming may differ from what's listed here. Run: `python3 -c "from huggingface_hub import HfApi; [print(m.id) for m in HfApi().list_models(search='gemma-4', author='google')]"`

---

## 📚 Research Background

This project's methodology is informed by recent research:

| Paper | Key Insight Applied |
|-------|-------------------|
| Biderman et al. (2024) "LoRA Learns Less and Forgets Less" | LoRA r=128+ for CPT (r=64 recovers only ~80%) |
| Ibrahim et al. (2024) "Simple Strategies for CPT" | 5-10% replay sufficient for forgetting prevention |
| Sabiá (Maritaca, 2023) | 7B tokens on LLaMA-7B → +20% on ENEM |
| Tucano (PUCRS, 2024) | Data quality > quantity; heavy dedup crucial |
| Ilharco et al. (2023) "Task Arithmetic" | Residual merge with α=0.5-1.0 for CPT |
| Yadav et al. (2023) "TIES-Merging" | k=20%, α=0.5 when combining multiple vectors |

**Full literature review:** [`docs/CPT_BEST_PRACTICES_RESEARCH.md`](docs/CPT_BEST_PRACTICES_RESEARCH.md)

---

## 🔬 Reproducibility

Every experiment is fully reproducible:

- **Fixed seeds**: Default 42 (configurable), deterministic ops enabled
- **Versioned configs**: All hyperparameters in committed YAML files
- **Run manifests**: Git SHA, package versions, resolved configs saved per run
- **Inference caching**: Eval results cached by MD5(model + benchmark + seed)
- **Bootstrap CI**: 95% confidence intervals, not just point estimates
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
| [`infra/gcp/QUICKSTART.md`](infra/gcp/QUICKSTART.md) | GCP deployment guide with cost estimates |
| [`reports/TECHNICAL_REPORT.md`](reports/TECHNICAL_REPORT.md) | Technical report with methodology |

---

## 🖥️ Hardware Requirements

| Task | Minimum | Recommended |
|------|---------|-------------|
| Tests, smoke, preflight | CPU only | Any machine |
| CPT Pilot (E4B, LoRA) | 1× A100 40GB | 1× A100 80GB |
| CPT Main (26B, Full FT) | 4× A100 80GB | 4× A100 80GB (ZeRO-2) |
| Evaluation | 1× A100 40GB | 1× A100 80GB |
| Residual Merge | CPU (64GB RAM) | CPU (128GB RAM) |
| Full pipeline | 1× A100 80GB | 4× A100 80GB |

**VRAM estimation** is built-in:
```bash
bash scripts/preflight.sh --config configs/train/cpt_pilot.yaml
# Output: Model 8.0 GB + Optimizer 0.5 GB + Activations 2.8 GB = ~11.7 GB total
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `flash-attn not installed` | Automatic fallback to SDPA (PyTorch native). Or: `pip install flash-attn` |
| OOM during training | Reduce `per_device_train_batch_size` or enable gradient checkpointing |
| Spot VM preempted | Checkpoints auto-sync to GCS every 200 steps. Just restart the VM. |
| `model_config is required` | Ensure your training config has `model_config: "configs/model/..."` |
| HF gated model access denied | Run `huggingface-cli login` with token that has Gemma access |
| Tests fail with `ModuleNotFoundError` | Run `pip install -e ".[dev]"` from project root |
| NaN in loss | Reduce learning rate, check data quality, disable tf32 |
| Slow data loading | Increase `dataloader_num_workers` (4-8 for A100) |

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
