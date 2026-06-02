# Data Pipeline Guide

## Overview

The data pipeline transforms raw Aurora-PT corpus into training-ready packed sequences,
while ensuring no contamination with evaluation benchmarks.

## Stage 1: Raw Loading

The Aurora-PT corpus (`dominguesm/aurora-pt`) is a large Portuguese text collection
hosted on HuggingFace. We load it with optional streaming for memory efficiency.

```python
from src.data.aurora_loader import AuroraLoader
from src.utils.config_utils import load_config

config = load_config("configs/data/aurora_pt.yaml")
loader = AuroraLoader(config)
dataset = loader.load_raw(streaming=False)
```

## Stage 2: Preprocessing

Documents are filtered and cleaned:

1. **Length filter**: Remove docs < 100 chars (noise) or > 500K chars (dumps)
2. **Whitespace normalization**: Collapse multiple spaces, limit blank lines
3. **Email redaction**: Replace email addresses with `[EMAIL]` token
4. **Deduplication** (optional): MinHash-based near-duplicate removal

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

To prevent catastrophic forgetting of English, we mix Portuguese with replay data:

| Mixture | Portuguese | English | Code |
|---------|-----------|---------|------|
| `pt_only` | 100% | 0% | 0% |
| `pt_en_replay` | 85% | 15% | 0% |
| `pt_en_code` | 80% | 12% | 8% |

English replay uses FineWeb-Edu (high quality). Code uses StarCoder data.

## Stage 5: Tokenization and Packing

For CPT, we pack multiple documents into fixed-length sequences:

```
[Doc1_tokens][Doc2_tokens][Doc3_partial...] → seq_length = 8192
[...Doc3_remaining][Doc4_tokens][...]       → seq_length = 8192
```

No separator tokens between documents (standard for CPT). Labels = input_ids
(causal LM objective: predict next token).

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
  hub_id: "dominguesm/aurora-pt"
  val_ratio: 0.005
  seed: 42

preprocessing:
  min_chars: 100
  max_chars: 500000
  dedup_method: "minhash"

mixtures:
  pt_en_replay:
    aurora_pt: 0.85
    english_replay: 0.15

packing:
  enabled: true
  max_seq_length: 8192
```
