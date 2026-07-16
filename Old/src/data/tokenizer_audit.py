"""Tokenizer fertility and efficiency audit for Portuguese text.

This module measures how efficiently a tokenizer handles Portuguese text
compared to English. Tokenizer "fertility" is the ratio of tokens produced
per word or character — lower is better (more efficient encoding).

Why this matters:
- Tokenizers trained primarily on English may over-segment Portuguese words,
  producing more tokens for the same content (higher cost, longer sequences).
- Portuguese has rich morphology (inflections, diminutives, augmentatives)
  that poorly-trained tokenizers fragment excessively.
- The fertility ratio between tokenizers helps quantify the efficiency gap
  and justify whether tokenizer adaptation is needed.

Metrics computed:
- tokens_per_word: Average number of tokens per whitespace-separated word.
  English-optimized tokenizers typically show 1.3-1.5 for English but
  1.8-2.5 for Portuguese.
- tokens_per_char: Average tokens per character (captures subword granularity).
- avg_token_length_chars: Average decoded length of individual tokens.
  Longer tokens = more efficient encoding.

Usage:
    from transformers import AutoTokenizer
    from src.data.tokenizer_audit import TokenizerAudit

    tok = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it")
    audit = TokenizerAudit(tok, name="gemma4")
    results = audit.compute_fertility(portuguese_texts)
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
from tqdm import tqdm

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TokenizerAudit:
    """Audit tokenizer efficiency for Portuguese text.

    Computes fertility metrics that quantify how many tokens a tokenizer
    produces per word/character on a given corpus. Supports comparison
    between two tokenizers on the same texts.

    Args:
        tokenizer: HuggingFace tokenizer instance.
        name: Human-readable name for this tokenizer (used in reports).
    """

    def __init__(self, tokenizer, name: str = "default"):
        self.tokenizer = tokenizer
        self.name = name

    def compute_fertility(self, texts: list[str]) -> dict[str, float]:
        """Compute tokenizer fertility metrics over a sample of texts.

        Processes each text independently, computing per-text ratios
        and then aggregating with mean/std. This gives both the central
        tendency and the variation across different text types.

        Args:
            texts: List of raw text strings to tokenize.

        Returns:
            Dict with fertility metrics:
            - tokens_per_char_mean/std: Tokens per character
            - tokens_per_word_mean/std: Tokens per whitespace word
            - avg_token_length_chars: Mean decoded token length
            - median_token_length_chars: Median decoded token length
            - num_samples: Number of texts processed
        """
        tokens_per_char = []
        tokens_per_word = []
        token_lengths = []

        for text in tqdm(texts, desc=f"Auditing {self.name}"):
            # Tokenize without special tokens to measure content encoding only
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            n_tokens = len(tokens)
            n_chars = len(text)
            n_words = len(text.split())

            if n_chars > 0:
                tokens_per_char.append(n_tokens / n_chars)
            if n_words > 0:
                tokens_per_word.append(n_tokens / n_words)

            # Measure individual token lengths (how many chars each token represents)
            # Sample first 1000 tokens to avoid O(n^2) on very long texts
            for tok_id in tokens[:1000]:
                decoded = self.tokenizer.decode([tok_id])
                token_lengths.append(len(decoded))

        return {
            "tokenizer": self.name,
            "tokens_per_char_mean": float(np.mean(tokens_per_char)),
            "tokens_per_char_std": float(np.std(tokens_per_char)),
            "tokens_per_word_mean": float(np.mean(tokens_per_word)),
            "tokens_per_word_std": float(np.std(tokens_per_word)),
            "avg_token_length_chars": float(np.mean(token_lengths)),
            "median_token_length_chars": float(np.median(token_lengths)),
            "num_samples": len(texts),
        }

    def compare_with(self, other_tokenizer, other_name: str, texts: list[str]) -> dict[str, Any]:
        """Compare this tokenizer against another on the same texts.

        Useful for measuring how much less efficient Gemma 4's tokenizer is
        on Portuguese compared to a Portuguese-specific tokenizer (like
        BERTimbau's or Sabia's).

        Args:
            other_tokenizer: Reference tokenizer to compare against.
            other_name: Name for the reference tokenizer.
            texts: Shared text corpus for fair comparison.

        Returns:
            Dict with both tokenizers' metrics plus:
            - fertility_ratio: primary/reference tokens_per_word ratio
              (>1 means primary is less efficient)
            - efficiency_gap_pct: Percentage difference in tokens_per_word
              (positive = primary uses more tokens)
        """
        self_metrics = self.compute_fertility(texts)
        other_audit = TokenizerAudit(other_tokenizer, other_name)
        other_metrics = other_audit.compute_fertility(texts)

        comparison = {
            "primary": self_metrics,
            "reference": other_metrics,
            # Ratio > 1.0 means primary tokenizer is less efficient
            "fertility_ratio": (
                self_metrics["tokens_per_word_mean"]
                / max(other_metrics["tokens_per_word_mean"], 1e-8)
            ),
            # Percentage gap (positive = primary uses more tokens per word)
            "efficiency_gap_pct": (
                (self_metrics["tokens_per_word_mean"] - other_metrics["tokens_per_word_mean"])
                / max(other_metrics["tokens_per_word_mean"], 1e-8)
                * 100
            ),
        }
        return comparison


def run_tokenizer_audit(
    tokenizer,
    dataset: Dataset,
    sample_size: int = 5000,
    output_path: str | Path = "outputs/tokenizer_audit.json",
    reference_tokenizer=None,
    reference_name: str = "reference",
) -> dict[str, Any]:
    """Run full tokenizer audit pipeline and save results.

    Convenience function that samples from a dataset, computes fertility,
    optionally compares with a reference tokenizer, and saves results to JSON.

    Args:
        tokenizer: Primary tokenizer to audit (typically Gemma 4).
        dataset: HuggingFace Dataset with a "text" column.
        sample_size: Number of texts to sample (default 5000 for speed).
        output_path: Where to save the JSON results.
        reference_tokenizer: Optional second tokenizer for comparison.
        reference_name: Name for the reference tokenizer in reports.

    Returns:
        Dict with all fertility metrics (and comparison if reference provided).
    """
    # Random sample for efficiency (full corpus audit would be too slow)
    if len(dataset) > sample_size:
        indices = np.random.choice(len(dataset), sample_size, replace=False)
        texts = [dataset[int(i)]["text"] for i in indices]
    else:
        texts = [dataset[i]["text"] for i in range(len(dataset))]

    audit = TokenizerAudit(tokenizer, name="gemma4")
    results = audit.compute_fertility(texts)

    # Optional: compare against a reference tokenizer
    if reference_tokenizer:
        comparison = audit.compare_with(reference_tokenizer, reference_name, texts)
        results["comparison"] = comparison

    # Persist results as JSON for downstream reporting
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Tokenizer audit saved to {output_path}")
    logger.info(f"  Tokens/word: {results['tokens_per_word_mean']:.3f}")
    logger.info(f"  Tokens/char: {results['tokens_per_char_mean']:.4f}")
    logger.info(f"  Avg token length: {results['avg_token_length_chars']:.2f} chars")

    return results
