"""
src/data/instruction_data_builder.py
────────────────────────────────────
Format instruction datasets into Gemma 4 official chat template.

Uses ``tokenizer.apply_chat_template()`` with proper turn tokens and
applies label masking so loss is computed only on assistant turns.
Supports ``enable_thinking=True/False`` for think-mode variants.

.. important::
    This module is for **structured instruction data only** — never
    for raw Aurora-PT text (which uses CausalLM packed sequences).
"""

from __future__ import annotations

from typing import Any

import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Label ignore index for CrossEntropy (standard PyTorch convention)
IGNORE_INDEX = -100


class InstructionDataBuilder:
    """Build tokenized instruction datasets with label masking.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        Gemma 4 tokenizer (must have ``apply_chat_template``).
    max_seq_len : int
        Maximum sequence length.
    enable_thinking : bool
        If ``True``, enables thinking mode in the chat template.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int = 4096,
        enable_thinking: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.enable_thinking = enable_thinking

    def format_conversation(
        self,
        messages: list[dict[str, str]],
    ) -> dict[str, torch.Tensor]:
        """Tokenize a single conversation with label masking.

        Parameters
        ----------
        messages : list[dict[str, str]]
            Chat messages in the format:
            ``[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]``

        Returns
        -------
        dict[str, torch.Tensor]
            ``{"input_ids": ..., "labels": ..., "attention_mask": ...}``
            Labels are set to ``IGNORE_INDEX`` for non-assistant tokens.
        """
        # Full conversation — tokenized with special tokens
        full_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=self.enable_thinking,
            return_tensors="pt",
            max_length=self.max_seq_len,
            truncation=True,
        ).squeeze(0)

        # Build labels: mask everything except assistant responses
        labels = full_ids.clone()

        # Tokenize only the user/system parts to find what to mask
        # Strategy: tokenize without the last assistant turn to find
        # the boundary, then repeat for each turn pair.
        labels = self._mask_non_assistant_tokens(messages, labels)

        attention_mask = torch.ones_like(full_ids)

        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def _mask_non_assistant_tokens(
        self,
        messages: list[dict[str, str]],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Mask non-assistant tokens in labels with IGNORE_INDEX.

        Uses incremental tokenization to find turn boundaries.
        """
        # Build prefix for each turn to find boundaries
        current_prefix: list[dict[str, str]] = []

        for msg in messages:
            if msg["role"] != "assistant":
                # This is a user/system turn — we need to mask its tokens
                current_prefix.append(msg)
                prefix_ids = self.tokenizer.apply_chat_template(
                    current_prefix,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                    max_length=self.max_seq_len,
                    truncation=True,
                )
                # Mask up to this point
                mask_len = min(len(prefix_ids), len(labels))
                labels[:mask_len] = IGNORE_INDEX
            else:
                current_prefix.append(msg)

        return labels

    def build_dataset(
        self,
        dataset_id: str,
        split: str = "train",
        messages_column: str | None = "messages",
        human_column: str | None = None,
        assistant_column: str | None = None,
        system_column: str | None = None,
        max_samples: int | None = None,
    ) -> Dataset:
        """Load and tokenize an instruction dataset.

        Parameters
        ----------
        dataset_id : str
            HuggingFace dataset ID or local path.
        split : str
            Dataset split.
        messages_column : str | None
            Column containing pre-formatted messages list.
        human_column : str | None
            Column with human/user instructions (alternative format).
        assistant_column : str | None
            Column with assistant responses (alternative format).
        system_column : str | None
            Optional column with system prompts.
        max_samples : int | None
            Limit number of samples.

        Returns
        -------
        Dataset
            Tokenized dataset ready for SFTTrainer.
        """
        logger.info("Loading instruction dataset: %s [%s]", dataset_id, split)
        ds = load_dataset(dataset_id, split=split)

        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))

        def _process(example: dict[str, Any]) -> dict[str, Any]:
            if messages_column and messages_column in example:
                messages = example[messages_column]
            elif human_column and assistant_column:
                messages = []
                if system_column and example.get(system_column):
                    messages.append({"role": "system", "content": example[system_column]})
                messages.append({"role": "user", "content": example[human_column]})
                messages.append({"role": "assistant", "content": example[assistant_column]})
            else:
                raise ValueError(
                    f"Cannot extract messages from example. "
                    f"Provide messages_column or human_column + assistant_column."
                )

            result = self.format_conversation(messages)
            return {
                "input_ids": result["input_ids"].tolist(),
                "labels": result["labels"].tolist(),
                "attention_mask": result["attention_mask"].tolist(),
            }

        logger.info("Tokenizing %d examples …", len(ds))
        ds = ds.map(_process, remove_columns=ds.column_names)
        return ds
