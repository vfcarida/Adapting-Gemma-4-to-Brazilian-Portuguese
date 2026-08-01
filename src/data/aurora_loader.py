"""Aurora-PT corpus loader with document-level splitting.

This module handles the complete data preparation pipeline for continued
pretraining (CPT). It loads the Aurora-PT Portuguese corpus from HuggingFace,
applies quality filters, splits deterministically by document, and packs
sequences for efficient causal language model training.

Key design decisions:
- Document-level split prevents data leakage between train/val
- Hash-based splitting is deterministic and idempotent
- Sequence packing eliminates padding waste for variable-length documents
- EOS token inserted between packed documents to signal boundaries
- Optional curriculum sorting (shorter/cleaner docs first)
"""

import re
from typing import Any

from datasets import Dataset, load_dataset

from src.utils.hashing import deterministic_split_value
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class AuroraLoader:
    """Load and preprocess Aurora-PT corpus for continued pretraining.

    The Aurora-PT corpus (Itau-Unibanco/Aurora-PT) is a large-scale Brazilian
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
            # Shared with make_splits.py's hash-split fallback so the same
            # document lands in the same split under either code path.
            hash_val = deterministic_split_value(example["text"])
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
    curriculum_sort: bool = False,
    mask_cross_doc_labels: bool = False,
    use_eos_separator: bool = True,
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
        curriculum_sort: If True, sort documents by length (shorter first)
            before packing. This implements a simple curriculum learning
            strategy where shorter (typically cleaner) documents are seen
            first during training.
        mask_cross_doc_labels: If True, set labels to -100 at document
            boundaries in packed sequences. Prevents cross-document loss.
        use_eos_separator: If True (default), insert the tokenizer's EOS
            token between packed documents. If False, documents are
            concatenated with no separator (legacy/ablation behavior — see
            configs/train/ablation_packing.yaml's F1 variant).

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

    if curriculum_sort:
        logger.info("Curriculum sort: ordering documents by length (shorter first)...")
        dataset = dataset.map(lambda x: {"_len": len(x["text"])}, desc="Computing lengths")
        dataset = dataset.sort("_len")
        dataset = dataset.remove_columns(["_len"])

    logger.info("Tokenizing dataset...")
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    if pack:
        eos_token_id = tokenizer.eos_token_id if use_eos_separator else None
        tokenized = pack_sequences(
            tokenized,
            max_seq_length,
            eos_token_id=eos_token_id,
            mask_cross_doc_labels=mask_cross_doc_labels,
        )

    return tokenized


def pack_sequences(
    tokenized_dataset: Dataset,
    max_seq_length: int,
    eos_token_id: int | None = None,
    mask_cross_doc_labels: bool = False,
) -> Dataset:
    """Pack multiple documents into fixed-length sequences.

    Concatenates tokenized documents into a continuous stream, then slices
    into chunks of exactly max_seq_length. This eliminates padding waste
    and maximizes GPU utilization.

    An EOS token is inserted between documents to signal document boundaries.
    Without this separator, the model may learn spurious cross-document
    associations (e.g., predicting tokens from document B given context from
    document A's ending).

    Optionally, labels at document boundaries can be set to -100 (ignored by
    CrossEntropyLoss), preventing the model from being penalized for failing
    to predict the first token of a new document given unrelated context.
    This is a lightweight alternative to full document attention masking.

    Leftover tokens (< max_seq_length) at the end of a batch are discarded.
    With large datasets, this loss is negligible.

    Args:
        tokenized_dataset: Dataset with "input_ids" column (list of ints).
        max_seq_length: Fixed length for each output sequence.
        eos_token_id: Token ID to insert between documents. If None,
            no separator is inserted (legacy behavior).
        mask_cross_doc_labels: If True, set labels to -100 at positions
            immediately after EOS separators (first token of each new
            document in a packed sequence). This prevents cross-document
            loss while keeping the simple packing approach.

    Returns:
        Dataset with "input_ids" and "labels" columns, each of length
        max_seq_length. Labels are identical to input_ids (causal LM
        objective: predict the next token at each position), except at
        masked boundary positions when mask_cross_doc_labels=True.
    """
    IGNORE_INDEX = -100

    def pack_fn(examples):
        all_input_ids = []
        all_labels = []
        buffer = []  # Accumulates tokens across documents
        # Track positions of EOS separators for label masking
        boundary_positions = []

        for ids in examples["input_ids"]:
            # Insert EOS separator between documents (not before the first)
            if buffer and eos_token_id is not None:
                boundary_positions.append(len(buffer))
                buffer.append(eos_token_id)
            buffer.extend(ids)
            # Slice full sequences from buffer
            while len(buffer) >= max_seq_length:
                chunk = buffer[:max_seq_length]
                labels = chunk.copy()

                # Mask labels at cross-document boundaries
                if mask_cross_doc_labels:
                    for pos in boundary_positions:
                        if pos < max_seq_length:
                            # Mask the EOS and the token after it
                            labels[pos] = IGNORE_INDEX
                            if pos + 1 < max_seq_length:
                                labels[pos + 1] = IGNORE_INDEX

                all_input_ids.append(chunk)
                all_labels.append(labels)
                buffer = buffer[max_seq_length:]
                # Adjust boundary positions for consumed tokens
                boundary_positions = [
                    p - max_seq_length for p in boundary_positions if p >= max_seq_length
                ]

        # Remaining tokens in buffer are discarded per batch.
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
