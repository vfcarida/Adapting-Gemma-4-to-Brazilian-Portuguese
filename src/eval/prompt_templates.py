"""
src/eval/prompt_templates.py
────────────────────────────
Gemma 4 official chat template wrapper for evaluation prompts.

Handles:
  • ``enable_thinking=True/False`` via ``apply_chat_template``
  • Few-shot prompt construction
  • Stripping ``<|channel|>thought`` blocks from multi-turn history
"""

from __future__ import annotations

import re
from typing import Any

from transformers import PreTrainedTokenizerBase

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Regex to strip Gemma 4 thought channels from conversation history
_THOUGHT_CHANNEL_RE = re.compile(
    r"<\|channel\|>thought\n.*?<channel\|>",
    re.DOTALL,
)


class Gemma4PromptFormatter:
    """Format prompts using the Gemma 4 official chat template.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        Gemma 4 tokenizer with ``apply_chat_template`` support.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def format_prompt(
        self,
        messages: list[dict[str, str]],
        enable_thinking: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        """Format a conversation into a Gemma 4 prompt string.

        Parameters
        ----------
        messages : list[dict[str, str]]
            Messages in ``[{"role": "user/system/assistant", "content": "..."}]`` format.
        enable_thinking : bool
            If ``True``, inserts ``<|think|>`` token for reasoning mode.
        add_generation_prompt : bool
            If ``True``, appends the generation prompt for the model.

        Returns
        -------
        str
            Formatted prompt string.
        """
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    def format_fewshot(
        self,
        examples: list[dict[str, str]],
        query: str,
        system_prompt: str | None = None,
        enable_thinking: bool = False,
    ) -> str:
        """Build a few-shot prompt with examples.

        Parameters
        ----------
        examples : list[dict[str, str]]
            List of ``{"input": "...", "output": "..."}`` examples.
        query : str
            The query to answer.
        system_prompt : str | None
            Optional system instruction.
        enable_thinking : bool
            Whether to enable thinking mode.

        Returns
        -------
        str
            Few-shot formatted prompt.
        """
        messages: list[dict[str, str]] = []

        # System prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Few-shot examples as user/assistant turns
        for ex in examples:
            messages.append({"role": "user", "content": ex["input"]})
            messages.append({"role": "assistant", "content": ex["output"]})

        # Query (user turn with generation prompt)
        messages.append({"role": "user", "content": query})

        return self.format_prompt(
            messages,
            enable_thinking=enable_thinking,
            add_generation_prompt=True,
        )

    @staticmethod
    def strip_thought_channels(text: str) -> str:
        """Remove ``<|channel|>thought ... <channel|>`` blocks.

        Used when passing multi-turn history back to the model to
        avoid feeding previous reasoning into the next turn.

        Parameters
        ----------
        text : str
            Generated text potentially containing thought channels.

        Returns
        -------
        str
            Text with thought channels removed.
        """
        return _THOUGHT_CHANNEL_RE.sub("", text).strip()

    @staticmethod
    def extract_answer(text: str) -> str:
        """Extract the final answer from model output.

        Strips any thought channels and returns the clean response.

        Parameters
        ----------
        text : str
            Raw model output.

        Returns
        -------
        str
            Clean answer text.
        """
        cleaned = Gemma4PromptFormatter.strip_thought_channels(text)
        # Remove any remaining control tokens
        cleaned = re.sub(r"<\|[^|]+\|>", "", cleaned)
        return cleaned.strip()
