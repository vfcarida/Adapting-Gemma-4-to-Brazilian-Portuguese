# Data Pipeline Guide

## Overview

The data pipeline transforms raw Aurora-PT corpus into training-ready packed sequences,
while ensuring no contamination with evaluation benchmarks.

## Stage 1: Raw Loading

The Aurora-PT corpus (`Itau-Unibanco/Aurora-PT` — ~331B GPT-2 tokens, ~855GB,
**CC-BY-NC-SA-4.0**, non-commercial license) is a large Portuguese text
collection hosted on HuggingFace, built by aggregating 9 sources (CC100,
mOSCAR, Aya, FineWeb-2, Blogset-br, Aroeira, mC4, Wikipedia, HPLT 2.0) — see
[the NorBERTo paper](https://arxiv.org/abs/2605.00086). We load it with
optional streaming for memory efficiency.

```python
from src.data.aurora_loader import AuroraLoader
from src.utils.config_utils import load_config

config = load_config("configs/data/aurora_pt.yaml")
loader = AuroraLoader(config)
dataset = loader.load_raw(streaming=False)
```

## Stage 2: Preprocessing

There are two DISTINCT preprocessing paths — don't confuse them:

- **Live CPT path** (`AuroraLoader.preprocess`, always runs as part of
  `gemma4pt train-cpt`): reads the `preprocessing:` section of
  `configs/data/aurora_pt.yaml`. Applies:
  1. **Length filter**: Remove docs < 100 chars (noise) or > 500K chars (dumps)
  2. **Whitespace normalization**: Collapse multiple spaces, limit blank lines
  3. **Email redaction**: Replace email addresses with `[EMAIL]` token
- **Offline QC/dedup path** (`src/data/make_splits.py`, run explicitly via
  `bash scripts/run_data_qc.sh` — NOT automatically part of `train-cpt`):
  reads `quality_control:`/`deduplication:` in the same YAML. Applies
  language-ID, PII, and toxicity filtering (`quality_manifest.py`) plus
  MinHash near-duplicate clustering (`cluster_dedup.py`), and writes
  `outputs/data_splits/split_indices.json`. **This path's output is
  informational/for offline QC only — `train-cpt` does not currently read
  `split_indices.json` and re-derives its own hash-based train/val split
  independently (see Stage 3 below).** Run `run_data_qc.sh` to audit corpus
  quality before a real training run, but know that it doesn't gate what
  `train-cpt` actually trains on.

## Stage 3: Train/Validation Split

We use **document-level hashing** for deterministic splitting:

```python
hash_value = MD5(first_500_chars) / 0xFFFFFFFF  # Uniform in [0, 1]
split = "val" if hash_value < 0.005 else "train"
```

This ensures:
- Same document always goes to same split (idempotent)
- No sentence-level leakage between splits
- ~0.5% validation (tunable)

## Stage 4: Data Mixtures

To prevent catastrophic forgetting of English, we mix Portuguese with replay data.
Mixture names below are the exact keys under `mixtures:` in
`configs/data/aurora_pt.yaml` — used via `data_mixture: "<name>"` in a
training config (there is no `pt_en_replay` key; that name doesn't exist):

| Mixture | Portuguese | English | Code |
|---------|-----------|---------|------|
| `pt_only` | 100% | 0% | 0% |
| `pt_en_5` | 95% | 5% | 0% |
| `pt_en_10` | 90% | 10% | 0% |
| `pt_en_15` | 85% | 15% | 0% |
| `pt_en_code` | 80% | 12% | 8% |

English replay uses FineWeb-Edu (high quality). Code uses StarCoder data
(gated — requires HF auth with dataset access granted).

**Replay ratio guidance** (Ibrahim et al. 2024, see
`docs/CPT_BEST_PRACTICES_RESEARCH.md`): ~5% replay is enough for a
same-domain shift, but a full language shift (English→Portuguese, the case
here) benefits from more — their closest analog (English→German) found
~25% replay gave the best English-retention/Portuguese-adaptation tradeoff.
`pt_en_15` is a reasonable middle ground; consider `pt_en_5`+a custom
`pt_en_25` mixture if English retention is a hard requirement for your run.

## Stage 5: Tokenization and Packing

For CPT, we pack multiple documents into fixed-length sequences:

```
[Doc1_tokens][Doc2_tokens][Doc3_partial...] → seq_length = 8192
[...Doc3_remaining][Doc4_tokens][...]       → seq_length = 8192
```

An EOS token is inserted between documents by default
(`packing.eos_separator: true` in a training config — see
`configs/train/cpt_pilot.yaml`) so the model learns document boundaries;
this can be disabled for an ablation (`configs/train/ablation_packing.yaml`'s
F1 variant). Labels = input_ids (causal LM objective: predict next token),
optionally with `-100` at the EOS boundary
(`packing.mask_cross_doc_labels: true`) to exclude the boundary-crossing
prediction from the loss. The collator that feeds the Trainer
(`PackedSequenceCollator` in `src/train/cpt_trainer.py`) preserves these
precomputed labels as-is — it does NOT use
`DataCollatorForLanguageModeling`, which would silently recompute labels
from `input_ids` and discard this masking.

## Stage 6: Contamination Checks

Before training, we verify no benchmark data appears in the training corpus:

1. **Exact match**: SHA-256 hash comparison after normalization
2. **Normalized overlap**: Lowercased, stripped punctuation comparison
3. **Fuzzy match**: MinHash LSH with Jaccard threshold ≥ 0.7
4. **N-gram overlap**: 10-gram set intersection ratio ≥ 0.5

Any document exceeding thresholds is flagged and can be removed.

## Tokenizer Audit

We measure Gemma 4's tokenizer efficiency on Portuguese:

- **Tokens per word**: How many subwords per Portuguese word (lower = better)
- **Tokens per character**: Byte-level efficiency
- **Average token length**: In characters (longer = more efficient)

This quantifies the "fertility gap" vs. a Portuguese-native tokenizer.

## Configuration Reference

See `configs/data/aurora_pt.yaml` for all options:

```yaml
dataset:
  hub_id: "Itau-Unibanco/Aurora-PT"
  val_ratio: 0.005
  seed: 42

# Read by AuroraLoader.preprocess (the live CPT path)
preprocessing:
  min_chars: 100
  max_chars: 500000
  remove_emails: true
  normalize_whitespace: true

# Read by src/data/make_splits.py (the offline QC/dedup path — see Stage 2)
quality_control:
  min_quality_score: 0.3
  # ...

mixtures:
  pt_en_15:
    aurora_pt: 0.85
    english_replay: 0.15

# packing.enabled/eos_separator/mask_cross_doc_labels live in the TRAINING
# config (e.g. configs/train/cpt_pilot.yaml), not here — see Stage 5.
```

---

## Running on a single GPU (Google Colab)

For a demonstration/pilot run on a single GPU without any of the above
requiring manual tuning, see [`colab/README.md`](../colab/README.md) and
[`configs/train/cpt_colab_pilot.yaml`](../configs/train/cpt_colab_pilot.yaml)
— a pre-sized QLoRA CPT config (Gemma 4 E2B, shorter sequences, small step
budget) that fits a Colab T4/L4/A100 session.
