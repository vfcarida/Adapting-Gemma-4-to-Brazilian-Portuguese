"""Build instruction datasets for SFT with proper chat formatting."""

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Gemma 4 chat template
GEMMA4_USER_PREFIX = "<start_of_turn>user\n"
GEMMA4_USER_SUFFIX = "<end_of_turn>\n"
GEMMA4_MODEL_PREFIX = "<start_of_turn>model\n"
GEMMA4_MODEL_SUFFIX = "<end_of_turn>\n"


def format_gemma4_chat(
    messages: list[dict[str, str]],
    add_generation_prompt: bool = False,
    use_think: bool = False,
) -> str:
    """Format messages using Gemma 4 chat template.

    Args:
        messages: List of {"role": "user"|"model", "content": "..."} dicts
        add_generation_prompt: Whether to add model turn prefix at end
        use_think: Whether to add <think> token after model prefix
    """
    formatted = "<bos>"
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            formatted += f"{GEMMA4_USER_PREFIX}{content}{GEMMA4_USER_SUFFIX}"
        elif role in ("model", "assistant"):
            if use_think:
                formatted += f"{GEMMA4_MODEL_PREFIX}<think>\n{content}\n</think>\n{GEMMA4_MODEL_SUFFIX}"
            else:
                formatted += f"{GEMMA4_MODEL_PREFIX}{content}{GEMMA4_MODEL_SUFFIX}"

    if add_generation_prompt:
        if use_think:
            formatted += f"{GEMMA4_MODEL_PREFIX}<think>\n"
        else:
            formatted += GEMMA4_MODEL_PREFIX

    return formatted


class InstructionDataBuilder:
    """Build instruction-tuning datasets from various sources."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.sft_cfg = config.get("sft", {})
        self.use_think = self.sft_cfg.get("use_think_tokens", False)
        self.max_seq_length = self.sft_cfg.get("max_seq_length", 4096)

    def load_datasets(self) -> Dataset:
        """Load and merge all configured instruction datasets."""
        datasets_cfg = self.sft_cfg.get("datasets", [])
        all_datasets = []

        for ds_cfg in datasets_cfg:
            hub_id = ds_cfg.get("hub_id")
            local_path = ds_cfg.get("path")
            weight = ds_cfg.get("weight", 1.0)

            if hub_id:
                ds = self._load_from_hub(hub_id)
            elif local_path:
                ds = self._load_from_local(local_path)
            else:
                logger.warning("Dataset config has no hub_id or path, skipping")
                continue

            if ds is not None and len(ds) > 0:
                all_datasets.append((ds, weight))

        if not all_datasets:
            raise ValueError("No instruction datasets could be loaded")

        return self._weighted_merge(all_datasets)

    def _load_from_hub(self, hub_id: str) -> Dataset | None:
        """Load dataset from HuggingFace Hub."""
        try:
            logger.info(f"Loading instruction data from {hub_id}")
            ds = load_dataset(hub_id, split="train")
            return self._normalize_columns(ds)
        except Exception as e:
            logger.warning(f"Failed to load {hub_id}: {e}")
            return None

    def _load_from_local(self, path: str) -> Dataset | None:
        """Load dataset from local JSONL file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Local dataset not found: {path}")
            return None
        try:
            ds = load_dataset("json", data_files=str(path), split="train")
            return self._normalize_columns(ds)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return None

    def _normalize_columns(self, ds: Dataset) -> Dataset:
        """Normalize dataset columns to standard format."""
        # Handle various column naming conventions
        col_map = {}
        columns = ds.column_names

        if "instruction" in columns and "output" in columns:
            # Alpaca format
            def convert_alpaca(example):
                user_msg = example["instruction"]
                if example.get("input"):
                    user_msg += f"\n\n{example['input']}"
                return {
                    "messages": [
                        {"role": "user", "content": user_msg},
                        {"role": "model", "content": example["output"]},
                    ]
                }
            return ds.map(convert_alpaca, remove_columns=columns)

        elif "conversations" in columns:
            # ShareGPT format
            def convert_sharegpt(example):
                messages = []
                for turn in example["conversations"]:
                    role = "user" if turn["from"] in ("human", "user") else "model"
                    messages.append({"role": role, "content": turn["value"]})
                return {"messages": messages}
            return ds.map(convert_sharegpt, remove_columns=columns)

        elif "messages" in columns:
            # Already in messages format
            return ds

        elif "prompt" in columns and "response" in columns:
            def convert_prompt_response(example):
                return {
                    "messages": [
                        {"role": "user", "content": example["prompt"]},
                        {"role": "model", "content": example["response"]},
                    ]
                }
            return ds.map(convert_prompt_response, remove_columns=columns)

        else:
            logger.warning(f"Unknown column format: {columns}")
            return ds

    def _weighted_merge(self, datasets_weights: list[tuple[Dataset, float]]) -> Dataset:
        """Merge datasets with weighting via sampling."""
        total_samples = sum(len(ds) for ds, _ in datasets_weights)
        merged = []

        for ds, weight in datasets_weights:
            n_samples = int(len(ds) * weight)
            if n_samples > len(ds):
                n_samples = len(ds)
            indices = list(range(n_samples))
            merged.append(ds.select(indices))

        result = concatenate_datasets(merged)
        result = result.shuffle(seed=42)
        logger.info(f"Merged instruction dataset: {len(result)} samples")
        return result

    def format_for_training(self, dataset: Dataset, tokenizer) -> Dataset:
        """Format dataset with chat template and tokenize."""

        def format_and_tokenize(example):
            messages = example["messages"]
            # Format full conversation
            full_text = format_gemma4_chat(messages, use_think=self.use_think)

            # Tokenize
            tokenized = tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_seq_length,
                padding=False,
            )

            # Create labels with masking for prompt tokens
            input_ids = tokenized["input_ids"]
            labels = input_ids.copy()

            if self.sft_cfg.get("train_on_completions_only", True):
                # Mask everything before the model response
                response_template = self.sft_cfg.get(
                    "response_template", GEMMA4_MODEL_PREFIX
                )
                response_token_ids = tokenizer.encode(
                    response_template, add_special_tokens=False
                )

                # Find response start positions and mask prefix
                labels = self._mask_prompt_tokens(
                    input_ids, labels, response_token_ids
                )

            tokenized["labels"] = labels
            return tokenized

        formatted = dataset.map(
            format_and_tokenize,
            remove_columns=dataset.column_names,
            desc="Formatting for SFT",
        )
        return formatted

    def _mask_prompt_tokens(
        self,
        input_ids: list[int],
        labels: list[int],
        response_token_ids: list[int],
    ) -> list[int]:
        """Mask prompt tokens in labels (set to -100).

        Finds all model response sections (after response_template) and
        only trains on those tokens. Everything else (system prompt, user
        turns, template tokens) is masked.

        Strategy: unmask from each response_template end until the start
        of the next response_template (which marks the beginning of a new
        user turn in multi-turn conversations).
        """
        IGNORE_INDEX = -100
        masked_labels = [IGNORE_INDEX] * len(labels)

        # Find all occurrences of response template (model turn starts)
        template_len = len(response_token_ids)
        template_starts = []  # positions where template begins

        for i in range(len(input_ids) - template_len + 1):
            if input_ids[i : i + template_len] == response_token_ids:
                template_starts.append(i)

        if not template_starts:
            # If we can't find the template, train on everything
            return labels

        # For each model response: unmask from end of template to start of next template
        for idx, tstart in enumerate(template_starts):
            response_begin = tstart + template_len  # first token of model response

            if idx + 1 < len(template_starts):
                # End before the next template (which is user turn + next model prefix)
                response_end = template_starts[idx + 1]
            else:
                # Last response: goes until end of sequence
                response_end = len(input_ids)

            # Unmask model response tokens (includes end_of_turn as part of response)
            for j in range(response_begin, response_end):
                masked_labels[j] = labels[j]

        return masked_labels
