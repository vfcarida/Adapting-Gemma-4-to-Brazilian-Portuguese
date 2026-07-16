# 🇧🇷 Adapting Gemma 4 to Brazilian Portuguese

> **Production-grade pipeline for computationally adapting Google Gemma 4 to Portuguese (pt-BR) via the Aurora-PT corpus (331B tokens).**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Framework: HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests: Passing](https://img.shields.io/badge/tests-223_passed-success.svg)](#)

---

## 📋 Scientific Overview

This repository implements a rigorous **five-stage adaptation pipeline** intended to produce state-of-the-art results for Brazilian Portuguese, moving beyond simple instruction-tuning to proper language adaptation. 

Our strategy strictly separates **Language Adaptation (CPT)** from **Instruction Alignment (SFT/DPO)**, preventing the catastrophic forgetting often seen when continuously pretraining on instruction-tuned models.

```mermaid
graph TD
    subgraph Pre-flight & Validation
        PF[gemma4pt preflight<br>Check Env / CUDA / Storage]
        DC[gemma4pt contamination-check<br>MinHash LSH / Exact / Ngram]
        TA[gemma4pt tokenizer-audit<br>Fertility Analysis]
    end

    subgraph Data Pipeline
        AP[Aurora-PT Dataset<br>331B Tokens] --> LB[AuroraLoader<br>Sequence Packing / Mix Replay]
    end

    subgraph Training Stage
        GM[Gemma-4 Base] --> CPT[train-cpt<br>Continued Pretraining]
        LB --> CPT
    end

    subgraph Alignment & Merging
        CPT --> AM[Weight Merging]
        GM --> AM
        IT[Gemma-4 IT] --> AM
        
        subgraph Merging Algorithms
            AM --> |Task Arithmetic| TA_M[cpt + alpha * residual]
            AM --> |TIES-Merge| TI_M[Trim + Elect Sign + Merge]
            AM --> |DARE-TIES| DT_M[Drop + Rescale + TIES]
        end
    end

    subgraph Evaluation
        TA_M & TI_M & DT_M --> EV[gemma4pt eval<br>Single-Load Memory Cache]
        EV --> RB[gemma4pt report<br>Findings & Stats]
    end

    style PF fill:#e1f5fe,stroke:#01579b
    style DC fill:#e1f5fe,stroke:#01579b
    style TA fill:#e1f5fe,stroke:#01579b
    style LB fill:#fff3e0,stroke:#e65100
    style CPT fill:#fff3e0,stroke:#e65100
    style AM fill:#e8f5e9,stroke:#1b5e20
    style TA_M fill:#f3e5f5,stroke:#4a148c
    style TI_M fill:#f3e5f5,stroke:#4a148c
    style DT_M fill:#f3e5f5,stroke:#4a148c
    style EV fill:#ffe0b2,stroke:#e65100
    style RB fill:#ffe0b2,stroke:#e65100
```

### 🔬 Core Methodology & Golden Rules
1. **The Golden Rule**: Aurora-PT is an unstructured corpus and is **never** used inside an `SFTTrainer`. It is processed strictly via `CausalLM` next-token prediction with packed sequences.
2. **Replay Mix Strategy**: To preserve emergent downstream capabilities and coding skills, our CPT stage utilizes probabilistic dataset interleaving. We mix Portuguese (Aurora-PT) with high-quality English (e.g., FineWeb-Edu) and optional Code (e.g., StarCoder).
3. **LoRA Safety Validation**: Gemma 4 utilizes `Gemma4ClippableLinear` in its vision and audio towers. We explicitly restrict our LoRA `target_modules` to language projections to prevent architectural crashes.
4. **Think Mode Isolation**: Evaluations are strictly isolated. We run all benchmarks in both `think_on` and `think_off` parametric modes to decouple native language improvements from chain-of-thought reasoning artifacts.
5. **Multi-tier Decontamination**: We run MinHash LSH and Exact/Normalized overlap checks against all benchmark datasets prior to training to ensure clean data validation.

## 📚 The Aurora-PT Dataset

**Aurora-PT** is a foundational, massive-scale dataset comprising **331B tokens** of high-quality Portuguese text. It is designed to act as the ultimate pretraining resource for adapting LLMs to the Portuguese language.

- **Size**: ~331 Billion tokens.
- **Role in Gemma 4**: It is strictly used for the **CPT (Continued Pretraining)** stage to inject profound linguistic representations.
- **Structure**: Unstructured text, processed entirely via causal modeling (next-token prediction), safely separating linguistic syntax from instruction-following behaviors.

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/vfcarida/Adapting-Gemma-4-to-Brazilian-Portuguese
cd Adapting-Gemma-4-to-Brazilian-Portuguese

# 2. Install (Development / CPU mode)
pip install -e ".[dev]"

# 3. (Optional) Install GPU and Training dependencies
pip install -e ".[gpu]"    # or ".[all]" for full stack
```

# 2. Validate environment
gemma4pt preflight

# 3. Run tests
pytest tests/ -q

# 4. End-to-end Smoke test (CPU)
gemma4pt smoke

# 5. Train (when GPU is available)
gemma4pt train-cpt configs/train/cpt_pilot.yaml
```

## 🛠️ CLI (`gemma4pt`)

The project now includes a powerful CLI to manage the entire pipeline:
```bash
gemma4pt preflight          # Valida ambiente
gemma4pt smoke              # Smoke test E2E
gemma4pt data-validate      # Valida dados
gemma4pt contamination-check # Verifica contaminação
gemma4pt tokenizer-audit    # Fertilidade tokenizer
gemma4pt train-cpt CONFIG   # Continued Pretraining
gemma4pt train-sft CONFIG   # SFT
gemma4pt merge              # Residual merge
gemma4pt eval               # Avaliação benchmarks
gemma4pt report             # Gera relatórios
gemma4pt manifest           # Manifesto reprodutibilidade
gemma4pt run-all            # Pipeline completo
```
*(All operations support `--dry-run`, `--tiny`, and `--cpu-only` flags).*

---

## 🧬 Advanced Weight Merging

To recover instruction capability after CPT, this pipeline implements state-of-the-art parameter merging algorithms inside `src/train/residual_merge.py`. These prevent weight interference and sign conflicts:

*   **Task Arithmetic (Linear):** Direct addition of the instruction task vector: `CPT + alpha * (IT - Base)`.
*   **TIES-Merging:** Trims small parameter deltas, elects consensus signs across task vectors, and merges only agreeing updates (averaging the updates).
*   **DARE-Linear:** Prunes weights randomly using a Bernoulli drop mask and rescales remaining weights by `1 / density` to preserve expectation, followed by a linear merge.
*   **DARE-TIES:** Combines random drop-and-rescale (DARE) with consensus sign election and disjoint merging (TIES).

### Merging CLI Usage
```bash
# Run TIES-Merge with density 0.6 and alpha 0.8
gemma4pt merge \
  --base-model google/gemma-4-E4B \
  --instruct-model google/gemma-4-E4B-it \
  --cpt-model outputs/cpt_main/final \
  --alpha 0.8 \
  --method ties \
  --density 0.6 \
  --output-dir outputs/residual_merge

# Run DARE-TIES merge sweep over multiple alpha values
gemma4pt merge \
  --base-model google/gemma-4-E4B \
  --instruct-model google/gemma-4-E4B-it \
  --cpt-model outputs/cpt_main/final \
  --alpha 0.6 0.8 1.0 1.2 \
  --method dare_ties \
  --density 0.5 \
  --output-dir outputs/residual_merge
```

---

## 📚 Documentation

Deep operational guides are located in the `docs/` folder:
| Document | Content |
|----------|---------|
| `docs/GEMMA4_COMPLIANCE.md` | Conformidade Gemma 4, thinking, multi-turn |
| `docs/TRAIN_READY.md` | Checklist de prontidão para treino |
| `docs/EVAL_PROTOCOL.md` | Protocolo de avaliação, métricas, CI |
| `docs/SMOKE_TESTS.md` | Smoke tests e validação |
| `docs/EXPERIMENT_PLAN.md` | Plano experimental 11 passos |
| `docs/ARCHITECTURE.md` | Design do sistema |
| `docs/DATA_PIPELINE.md` | Pipeline de dados |
| `docs/TRAINING_GUIDE.md` | Guia de treinamento |

---

## 📊 Evaluation Benchmarks

We utilize a layered evaluation suite to prevent saturation on easy or highly-translated English benchmarks. All models are evaluated generatively (`temperature=0.0`).

| Benchmark | Domain | Metric | 
|-----------|--------|--------|
| **ENEM** | Education (National Exam) | Approval Rate |
| **BluEx** | Education (University Entrance) | Approval Rate |
| **OAB-Bench** | Legal (Bar Exam) | Approval Rate |
| **ASSIN2-RTE** | NLI (Textual Entailment) | macro-F1 |
| **ASSIN2-STS** | Semantic Similarity | Pearson r / Spearman ρ |
| **HateBR** | Hate Speech Detection | macro-F1 |
| **TweetSentBR** | Sentiment Analysis | macro-F1 |
| **COPA-PT** | Causal Reasoning | Accuracy |
| **BRoverbs** | Cultural (Proverb Completion) | Accuracy |
| **MRPC-PT** | Paraphrase Detection | macro-F1 |
| **RTE-PT** | Textual Entailment | Accuracy |
| **DoNotAnswer-PT** | Safety / Refusal | Refusal Rate |
| **TugueSICE-PT** | Language Understanding | Accuracy |
| **XLSum-PT** | Long-context Summarization | ROUGE (opt) / Gen |

---

## 📁 Repository Architecture

```
.
├── ablations/                 # Automated hypothesis test outputs
├── configs/                   # YAML configurations for CPT, SFT, DPO, Merge, Eval
├── src/                       # CLI, Preflight, Data, Train, Eval, Utils
├── docs/                      # Extensive operational guides
├── tests/                     # 198+ Unit, integration, smoke, and golden tests
├── scripts/                   # Legacy end-to-end bash execution scripts
└── reports/                   # Markdown generation (summary.md, findings_for_paper.md)
```
## 📝 Requirements
- Python ≥ 3.10
- HuggingFace account with access to `google/gemma-4` variants and `Itau-Unibanco/Aurora-PT`.

## 🤝 Contributing
Contributions are welcome! Please run `gemma4pt preflight` and ensure `pytest` and `ruff check` pass before submitting a Pull Request. Check out the issues tab for open tasks.

## 📜 License
Apache 2.0

## 📖 Citation
If you use this repository or methodology in your academic work, please cite it as:
```bibtex
@software{gemma4ptbr2026,
  author = {Caridá, Vinícius and Team},
  title = {Adapting Gemma 4 to Brazilian Portuguese},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\\url{https://github.com/vfcarida/Adapting-Gemma-4-to-Brazilian-Portuguese}}
}
```
