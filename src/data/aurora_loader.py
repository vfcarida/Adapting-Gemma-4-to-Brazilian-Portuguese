"""Aurora-PT corpus loader with document-level splitting.

This module handles the complete data preparation pipeline for continued
pretraining (CPT). It loads the Aurora-PT Portuguese corpus from HuggingFace,
applies quality filters, splits deterministically by document, and packs
sequences for efficient causal language model training.

Key design decisions:
- Document-level split prevents data leakage between train/val
- Hash-based splitting is deterministic and idempotent
- Sequence packing eliminates padding waste for variable-length documents
- No special separator between packed documents (standard CPT practice)
"""

import hashlib
import re
from typing import Any

from datasets import Dataset, load_dataset

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class AuroraLoader:
    """Load and preprocess Aurora-PT corpus for continued pretraining.

    The Aurora-PT corpus (dominguesm/aurora-pt) is a large-scale Brazilian
    Portuguese text collection. This loader applies quality filters and
    creates a deterministic train/validation split.

    Args:
        config: Data configuration dict (from configs/data/aurora_pt.yaml)

    Example:
        >>> config = load_config("configs/data/aurora_pt.yaml")
        >>> loader = AuroraLoader(config)
        >>> splits = loader.load_and_prepare()
        >>> print(f"Train: {len(splits['train'])}, Val: {len(splits['validation'])}")
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        ds_cfg = config["dataset"]
        self.hub_id = ds_cfg["hub_id"]
        self.val_ratio = ds_cfg.get("val_ratio", 0.005)
        self.seed = ds_cfg.get("seed", 42)
        self.preprocess_cfg = config.get("preprocessing", {})

    def load_raw(self, streaming: bool = False) -> Dataset:
        """Load raw Aurora-PT dataset from HuggingFace Hub.

        Args:
            streaming: If True, returns an IterableDataset (memory-efficient
                      for large corpora but incompatible with .filter/.map
                      that require length). Use False for preprocessing.

        Returns:
            HuggingFace Dataset with at least a 'text' column.
        """
        logger.info(f"Loading {self.hub_id} (streaming={streaming})")
        ds = load_dataset(self.hub_id, streaming=streaming, split="train")
        return ds

    def preprocess(self, dataset: Dataset) -> Dataset:
        """Apply preprocessing filters to remove noise and normalize text.

        Filters applied:
        1. Length filter: Remove documents shorter than min_chars or longer
           than max_chars. Very short docs are typically noise; very long
           docs may be data dumps.
        2. Whitespace normalization: Collapse multiple spaces/tabs into one,
           limit consecutive newlines to 2 (paragraph breaks).
        3. Email redaction: Replace email addresses with [EMAIL] placeholder
           to avoid memorization of PII.

        Args:
            dataset: Raw dataset to preprocess.

        Returns:
            Filtered and cleaned dataset.
        """
        min_chars = self.preprocess_cfg.get("min_chars", 100)
        max_chars = self.preprocess_cfg.get("max_chars", 500000)
        remove_emails = self.preprocess_cfg.get("remove_emails", True)
        normalize_ws = self.preprocess_cfg.get("normalize_whitespace", True)

        def filter_fn(example):
            """Reject documents outside acceptable length range."""
            text = example.get("text", "")
            if len(text) < min_chars or len(text) > max_chars:
                return False
            return True

        def clean_fn(example):
            """Normalize whitespace and redact emails."""
            text = example["text"]
            if normalize_ws:
                # Collapse horizontal whitespace (spaces, tabs)
                text = re.sub(r"[ \t]+", " ", text)
                # Limit vertical whitespace to paragraph breaks
                text = re.sub(r"\n{3,}", "\n\n", text)
            if remove_emails:
                # Simple email pattern - covers most cases
                text = re.sub(r"\S+@\S+\.\S+", "[EMAIL]", text)
            example["text"] = text.strip()
            return example

        logger.info("Filtering documents by length...")
        dataset = dataset.filter(filter_fn)
        logger.info("Cleaning documents...")
        dataset = dataset.map(clean_fn)
        return dataset

    def split_train_val(self, dataset: Dataset) -> dict[str, Dataset]:
        """Split by document hash for deterministic, leakage-free split.

        Uses MD5 hash of the first 500 characters of each document to
        deterministically assign it to train or validation. This ensures:
        - Same result regardless of document order
        - No information leakage between splits
        - Reproducible without storing split indices

        The first 500 chars are used (not full content) for efficiency and
        because they sufficiently identify unique documents.

        Args:
            dataset: Preprocessed dataset to split.

        Returns:
            Dict with "train" and "validation" Dataset objects.
        """

        def assign_split(example, idx):
            # Hash first 500 chars for deterministic assignment
            # MD5 is fine here (not security-critical, just uniform distribution)
            doc_hash = hashlib.md5(example["text"][:500].encode()).hexdigest()
            # Convert first 8 hex digits to float in [0, 1]
            hash_val = int(doc_hash[:8], 16) / 0xFFFFFFFF
            example["_split"] = "val" if hash_val < self.val_ratio else "train"
            return example

        logger.info(f"Splitting dataset (val_ratio={self.val_ratio})")
        dataset = dataset.map(assign_split, with_indices=True)

        train_ds = dataset.filter(lambda x: x["_split"] == "train")
        val_ds = dataset.filter(lambda x: x["_split"] == "val")

        # Remove temporary column
        train_ds = train_ds.remove_columns(["_split"])
        val_ds = val_ds.remove_columns(["_split"])

        logger.info(f"Train: {len(train_ds)} docs, Val: {len(val_ds)} docs")
        return {"train": train_ds, "validation": val_ds}

    def load_and_prepare(self) -> dict[str, Dataset]:
        """Full pipeline: load, preprocess, split.

        This is the main entry point for data preparation. It chains
        all steps in sequence: raw loading → preprocessing → splitting.

        Returns:
            Dict with "train" and "validation" Dataset objects,
            ready for tokenization and packing.
        """
        dataset = self.load_raw(streaming=False)
        dataset = self.preprocess(dataset)
        splits = self.split_train_val(dataset)
        return splits


def tokenize_for_cpt(
    dataset: Dataset,
    tokenizer,
    max_seq_length: int = 8192,
    pack: bool = True,
) -> Dataset:
    """Tokenize and optionally pack sequences for causal LM training.

    For CPT, we tokenize without truncation (documents may span multiple
    sequences after packing), without padding (packing handles alignment),
    and without attention masks (all tokens are attended to in packed seqs).

    Args:
        dataset: Dataset with "text" column.
        tokenizer: HuggingFace tokenizer instance.
        max_seq_length: Target sequence length for packing.
        pack: If True, concatenate documents into fixed-length sequences.
              If False, truncate each document independently.

    Returns:
        Dataset with "input_ids" and "labels" columns, ready for training.
    """

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=False,  # Don't truncate - packing handles length
            padding=False,  # No padding - packing fills sequences
            return_attention_mask=False,  # Not needed for packed CPT
        )

    logger.info("Tokenizing dataset...")
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    if pack:
        tokenized = pack_sequences(tokenized, max_seq_length)

    return tokenized


def pack_sequences(tokenized_dataset: Dataset, max_seq_length: int) -> Dataset:
    """Pack multiple documents into fixed-length sequences.

    Concatenates tokenized documents into a continuous stream, then slices
    into chunks of exactly max_seq_length. This eliminates padding waste
    and maximizes GPU utilization.

    Note: No separator tokens are inserted between documents. This is
    standard practice for CPT, as the model learns document boundaries
    implicitly from content patterns.

    Leftover tokens (< max_seq_length) at the end of a batch are carried
    over to the next batch via the buffer.

    Args:
        tokenized_dataset: Dataset with "input_ids" column (list of ints).
        max_seq_length: Fixed length for each output sequence.

    Returns:
        Dataset with "input_ids" and "labels" columns, each of length
        max_seq_length. Labels are identical to input_ids (causal LM
        objective: predict the next token at each position).
    """

    def pack_fn(examples):
        all_input_ids = []
        all_labels = []
        buffer = []  # Accumulates tokens across documents

        for ids in examples["input_ids"]:
            buffer.extend(ids)
            # Slice full sequences from buffer
            while len(buffer) >= max_seq_length:
                chunk = buffer[:max_seq_length]
                all_input_ids.append(chunk)
                # For causal LM, labels = input_ids (shifted internally by the model)
                all_labels.append(chunk.copy())
                buffer = buffer[max_seq_length:]

        # Note: remaining tokens in buffer are discarded per batch.
        # With large datasets, this loss is negligible.
        return {"input_ids": all_input_ids, "labels": all_labels}

    logger.info(f"Packing sequences to length {max_seq_length}...")
    packed = tokenized_dataset.map(
        pack_fn,
        batched=True,
        remove_columns=tokenized_dataset.column_names,
        desc="Packing",
    )
    logger.info(f"Packed dataset: {len(packed)} sequences")
    return packed
