"""
src/data/tokenizer_audit.py
───────────────────────────
Tokenizer fertility analysis for Gemma 4 vs Portuguese text.

Computes:
  • tokens / character  — expansion ratio
  • tokens / word       — subword fragmentation degree

Compares Gemma 4 tokenizer against a reference baseline and outputs
a structured JSON report to ``reports/``.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Iterator

from transformers import AutoTokenizer

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TokenizerAuditor:
    """Analyze tokenizer fertility on Portuguese text samples.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID whose tokenizer to audit.
    reference_tokenizer_id : str | None
        Optional reference tokenizer for comparison (e.g. a BPE model
        trained on Portuguese).  If ``None``, only the primary tokenizer
        is audited.
    """

    def __init__(
        self,
        model_id: str = "google/gemma-4-E4B",
        reference_tokenizer_id: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.reference_tokenizer_id = reference_tokenizer_id

        logger.info("Loading primary tokenizer: %s", model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        self.ref_tokenizer = None
        if reference_tokenizer_id:
            logger.info("Loading reference tokenizer: %s", reference_tokenizer_id)
            self.ref_tokenizer = AutoTokenizer.from_pretrained(
                reference_tokenizer_id, trust_remote_code=True
            )

    def audit(
        self,
        text_iterator: Iterator[str],
        num_samples: int = 1000,
        output_path: str | Path = "reports/tokenizer_audit.json",
    ) -> dict[str, Any]:
        """Run the fertility audit on sampled documents.

        Parameters
        ----------
        text_iterator : Iterator[str]
            An iterator yielding text documents (e.g. from AuroraLoader).
        num_samples : int
            Number of documents to sample for the audit.
        output_path : str | Path
            Where to save the JSON report.

        Returns
        -------
        dict[str, Any]
            Audit results including per-metric statistics.
        """
        primary_stats = _FertilityAccumulator(self.model_id)
        ref_stats = (
            _FertilityAccumulator(self.reference_tokenizer_id)
            if self.ref_tokenizer
            else None
        )

        count = 0
        for text in text_iterator:
            if count >= num_samples:
                break
            if not text.strip():
                continue

            # Primary tokenizer
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            primary_stats.update(text, tokens)

            # Reference tokenizer
            if self.ref_tokenizer and ref_stats:
                ref_tokens = self.ref_tokenizer.encode(text, add_special_tokens=False)
                ref_stats.update(text, ref_tokens)

            count += 1

        report = {
            "num_samples": count,
            "primary_tokenizer": {
                "model_id": self.model_id,
                "vocab_size": len(self.tokenizer),
                **primary_stats.summarize(),
            },
        }
        if ref_stats:
            report["reference_tokenizer"] = {
                "model_id": self.reference_tokenizer_id,
                "vocab_size": len(self.ref_tokenizer),
                **ref_stats.summarize(),
            }

        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Tokenizer audit saved → %s", output_path)

        return report


class _FertilityAccumulator:
    """Accumulate token fertility statistics."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.tokens_per_char: list[float] = []
        self.tokens_per_word: list[float] = []
        self.total_tokens: int = 0
        self.total_chars: int = 0
        self.total_words: int = 0

    def update(self, text: str, token_ids: list[int]) -> None:
        n_tokens = len(token_ids)
        n_chars = len(text)
        n_words = len(text.split())

        self.total_tokens += n_tokens
        self.total_chars += n_chars
        self.total_words += n_words

        if n_chars > 0:
            self.tokens_per_char.append(n_tokens / n_chars)
        if n_words > 0:
            self.tokens_per_word.append(n_tokens / n_words)

    def summarize(self) -> dict[str, Any]:
        return {
            "total_tokens": self.total_tokens,
            "total_chars": self.total_chars,
            "total_words": self.total_words,
            "global_tokens_per_char": (
                self.total_tokens / self.total_chars if self.total_chars else 0
            ),
            "global_tokens_per_word": (
                self.total_tokens / self.total_words if self.total_words else 0
            ),
            "tokens_per_char": {
                "mean": statistics.mean(self.tokens_per_char) if self.tokens_per_char else 0,
                "median": statistics.median(self.tokens_per_char) if self.tokens_per_char else 0,
                "stdev": (
                    statistics.stdev(self.tokens_per_char)
                    if len(self.tokens_per_char) > 1
                    else 0
                ),
            },
            "tokens_per_word": {
                "mean": statistics.mean(self.tokens_per_word) if self.tokens_per_word else 0,
                "median": statistics.median(self.tokens_per_word) if self.tokens_per_word else 0,
                "stdev": (
                    statistics.stdev(self.tokens_per_word)
                    if len(self.tokens_per_word) > 1
                    else 0
                ),
            },
        }


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    from src.data.aurora_loader import AuroraLoader

    parser = argparse.ArgumentParser(description="Tokenizer fertility audit")
    parser.add_argument("--model_id", default="google/gemma-4-E4B")
    parser.add_argument("--reference", default=None, help="Reference tokenizer model ID")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--output", default="reports/tokenizer_audit.json")
    parser.add_argument("--dataset_id", default="Itau-Unibanco/Aurora-PT")
    args = parser.parse_args()

    # ⚠️ REQUIRES HF_TOKEN for gated dataset access
    loader = AuroraLoader(dataset_id=args.dataset_id)
    auditor = TokenizerAuditor(
        model_id=args.model_id,
        reference_tokenizer_id=args.reference,
    )
    results = auditor.audit(
        text_iterator=loader.stream(),
        num_samples=args.num_samples,
        output_path=args.output,
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
