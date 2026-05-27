"""
src/data/aurora_loader.py
─────────────────────────
Streaming loader for the Aurora-PT corpus with packed-sequence dataset
construction.  This module implements CausalLM-style data preparation
(next-token prediction) — it must NEVER be used with ``SFTTrainer``.

Golden Rule: Aurora-PT is unstructured text.  Build packed sequences
with CausalLM / next-token prediction only.
"""

from __future__ import annotations

import itertools
from typing import Any, Iterator

import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class AuroraLoader:
    """Streaming loader for the Aurora-PT dataset on Hugging Face.

    Yields raw text documents from ``Itau-Unibanco/Aurora-PT`` using HF
    datasets streaming mode with optional sharding for multi-GPU or
    multi-node setups.

    Parameters
    ----------
    dataset_id : str
        HuggingFace dataset identifier.
    split : str
        Dataset split to load.
    text_column : str
        Column name containing the raw text.
    num_shards : int
        Total number of data shards (for distributed training).
    shard_index : int
        Index of the current shard (0-based).
    cache_dir : str | None
        Override for the HF cache directory.
    hf_token : str | None
        HuggingFace token for gated dataset access.

    Usage
    -----
    >>> loader = AuroraLoader(dataset_id="Itau-Unibanco/Aurora-PT")
    >>> for text in itertools.islice(loader.stream(), 100):
    ...     print(text[:80])  # first 80 chars of each document
    """

    def __init__(
        self,
        dataset_id: str = "Itau-Unibanco/Aurora-PT",
        split: str = "train",
        text_column: str = "text",
        num_shards: int = 1,
        shard_index: int = 0,
        cache_dir: str | None = None,
        hf_token: str | None = None,
    ) -> None:
        self.dataset_id = dataset_id
        self.split = split
        self.text_column = text_column
        self.num_shards = num_shards
        self.shard_index = shard_index
        self.cache_dir = cache_dir
        self.hf_token = hf_token

    def stream(self) -> Iterator[str]:
        """Yield text documents one at a time in streaming mode.

        .. note::
            **BLOCKING**: First call may take time to download the
            dataset metadata.  Requires ``HF_TOKEN`` to be set for
            gated datasets.
        """
        logger.info(
            "Streaming %s (split=%s, shard %d/%d)",
            self.dataset_id,
            self.split,
            self.shard_index + 1,
            self.num_shards,
        )
        # ⚠️ REQUIRES HF_TOKEN — Aurora-PT is gated
        ds = load_dataset(
            self.dataset_id,
            split=self.split,
            streaming=True,
            cache_dir=self.cache_dir,
            token=self.hf_token,
        )

        # Shard for distributed setups
        if self.num_shards > 1:
            ds = ds.shard(num_shards=self.num_shards, index=self.shard_index)

        for example in ds:
            text = example.get(self.text_column, "")
            if text and text.strip():
                yield text.strip()


class PackedSequenceDataset(IterableDataset):
    """Pack multiple documents into fixed-length sequences for CausalLM.

    Concatenates tokenized documents separated by EOS tokens, then
    chunks the stream into non-overlapping windows of ``max_seq_len``
    tokens.  This is the standard approach for continued pretraining
    without wasting compute on padding.

    Parameters
    ----------
    text_iterator : Iterator[str]
        An iterator yielding raw text documents.
    tokenizer : PreTrainedTokenizerBase
        The model tokenizer.
    max_seq_len : int
        Sequence length for each packed sample.

    Yields
    ------
    dict[str, torch.Tensor]
        ``{"input_ids": Tensor, "labels": Tensor, "attention_mask": Tensor}``
        where labels are a copy of input_ids (standard CausalLM).

    Example
    -------
    >>> loader = AuroraLoader()
    >>> packed = PackedSequenceDataset(loader.stream(), tokenizer, max_seq_len=4096)
    >>> for batch in DataLoader(packed, batch_size=2):
    ...     print(batch["input_ids"].shape)  # (2, 4096)
    """

    def __init__(
        self,
        text_iterator: Iterator[str],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int = 4096,
    ) -> None:
        self.text_iterator = text_iterator
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.eos_token_id = tokenizer.eos_token_id

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Yield packed sequences of exactly ``max_seq_len`` tokens."""
        buffer: list[int] = []

        for text in self.text_iterator:
            # Tokenize without special tokens — we add EOS manually
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            # Append EOS as document separator
            token_ids.append(self.eos_token_id)
            buffer.extend(token_ids)

            # Emit complete chunks from the buffer
            while len(buffer) >= self.max_seq_len:
                chunk = buffer[: self.max_seq_len]
                buffer = buffer[self.max_seq_len :]

                input_ids = torch.tensor(chunk, dtype=torch.long)
                yield {
                    "input_ids": input_ids,
                    "labels": input_ids.clone(),
                    "attention_mask": torch.ones_like(input_ids),
                }

        # Discard the final partial chunk — avoid padding waste
        if buffer:
            logger.debug(
                "Discarding final partial chunk (%d tokens < %d)",
                len(buffer),
                self.max_seq_len,
            )


def build_aurora_dataset(
    config: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> PackedSequenceDataset:
    """Factory function: build a packed Aurora-PT dataset from config.

    Parameters
    ----------
    config : dict
        Data section of the YAML config (expects keys: dataset_id,
        max_seq_len, num_shards, shard_index, etc.).
    tokenizer : PreTrainedTokenizerBase
        Model tokenizer.

    Returns
    -------
    PackedSequenceDataset
        Ready to pass to a DataLoader.
    """
    data_cfg = config.get("data", config)
    loader = AuroraLoader(
        dataset_id=data_cfg.get("dataset_id", "Itau-Unibanco/Aurora-PT"),
        text_column=data_cfg.get("text_column", "text"),
        num_shards=data_cfg.get("num_shards", 1),
        shard_index=data_cfg.get("shard_index", 0),
        cache_dir=data_cfg.get("cache_dir"),
    )
    return PackedSequenceDataset(
        text_iterator=loader.stream(),
        tokenizer=tokenizer,
        max_seq_len=data_cfg.get("max_seq_len", 4096),
    )
