"""
src/data/contamination_checks.py
────────────────────────────────
Three-tier decontamination pipeline:
  1. Exact match    — SHA-256 hash comparison of normalized text
  2. Normalized     — lowercase + strip punctuation + collapse whitespace
  3. Fuzzy match    — MinHash LSH (datasketch) with Jaccard threshold

Builds a contamination index from evaluation test sets, then filters
the Aurora-PT stream to produce a clean training shard.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterator

from datasketch import MinHash, MinHashLSH

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Text normalization
# ──────────────────────────────────────────────────────────────────────

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Lowercase, strip accents, remove punctuation, collapse whitespace."""
    text = text.lower()
    # Strip accents (NFD decomposition + remove combining chars)
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    # Remove punctuation
    text = _PUNCT_RE.sub(" ", text)
    # Collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def sha256_hash(text: str) -> str:
    """SHA-256 hex digest of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ──────────────────────────────────────────────────────────────────────
# MinHash builder
# ──────────────────────────────────────────────────────────────────────

_DEFAULT_NUM_PERM = 128


def build_minhash(text: str, num_perm: int = _DEFAULT_NUM_PERM) -> MinHash:
    """Build a MinHash signature from word-level shingles (3-grams)."""
    words = text.split()
    m = MinHash(num_perm=num_perm)
    for i in range(len(words) - 2):
        shingle = " ".join(words[i : i + 3])
        m.update(shingle.encode("utf-8"))
    return m


# ──────────────────────────────────────────────────────────────────────
# Contamination Checker
# ──────────────────────────────────────────────────────────────────────


class ContaminationChecker:
    """Three-tier decontamination checker.

    Build the index from evaluation test-set samples, then call
    :meth:`is_contaminated` on each training document to decide
    whether it should be kept or discarded.

    Parameters
    ----------
    jaccard_threshold : float
        MinHash LSH Jaccard similarity threshold for fuzzy matching.
    num_perm : int
        Number of permutations for MinHash signatures.
    """

    def __init__(
        self,
        jaccard_threshold: float = 0.8,
        num_perm: int = _DEFAULT_NUM_PERM,
    ) -> None:
        self.jaccard_threshold = jaccard_threshold
        self.num_perm = num_perm

        # Tier 1: exact hashes
        self._exact_hashes: set[str] = set()
        # Tier 2: normalized hashes
        self._norm_hashes: set[str] = set()
        # Tier 3: MinHash LSH index
        self._lsh = MinHashLSH(threshold=jaccard_threshold, num_perm=num_perm)
        self._lsh_counter = 0

        # Statistics
        self.stats = {
            "total_checked": 0,
            "exact_matches": 0,
            "normalized_matches": 0,
            "fuzzy_matches": 0,
            "clean": 0,
        }

    def add_reference(self, text: str) -> None:
        """Add a single reference text (from eval test set) to the index.

        Parameters
        ----------
        text : str
            A test-set sample to protect from contamination.
        """
        # Tier 1 — exact
        self._exact_hashes.add(sha256_hash(text))

        # Tier 2 — normalized
        norm = normalize_text(text)
        self._norm_hashes.add(sha256_hash(norm))

        # Tier 3 — MinHash
        mh = build_minhash(norm, num_perm=self.num_perm)
        self._lsh.insert(f"ref_{self._lsh_counter}", mh)
        self._lsh_counter += 1

    def add_references_from_dataset(
        self,
        dataset_id: str,
        split: str = "test",
        text_column: str = "text",
        max_samples: int | None = None,
    ) -> int:
        """Load a HuggingFace dataset split and add all samples to the index.

        Parameters
        ----------
        dataset_id : str
            HuggingFace dataset identifier.
        split : str
            Dataset split (typically ``"test"`` or ``"validation"``).
        text_column : str
            Column containing the text to index.
        max_samples : int | None
            Limit samples (for debugging).

        Returns
        -------
        int
            Number of samples added.
        """
        from datasets import load_dataset

        logger.info("Loading reference dataset: %s [%s]", dataset_id, split)
        try:
            ds = load_dataset(dataset_id, split=split)
        except Exception as e:
            logger.warning("Failed to load %s: %s — skipping.", dataset_id, e)
            return 0

        count = 0
        for example in ds:
            text = example.get(text_column, "")
            if not text or not text.strip():
                continue
            self.add_reference(text.strip())
            count += 1
            if max_samples and count >= max_samples:
                break

        logger.info("Added %d reference samples from %s", count, dataset_id)
        return count

    def is_contaminated(self, text: str) -> tuple[bool, str]:
        """Check whether a training document is contaminated.

        Returns
        -------
        tuple[bool, str]
            ``(is_contaminated, reason)`` where *reason* is one of
            ``"exact"``, ``"normalized"``, ``"fuzzy"``, or ``"clean"``.
        """
        self.stats["total_checked"] += 1

        # Tier 1 — Exact match
        if sha256_hash(text) in self._exact_hashes:
            self.stats["exact_matches"] += 1
            return True, "exact"

        # Tier 2 — Normalized match
        norm = normalize_text(text)
        if sha256_hash(norm) in self._norm_hashes:
            self.stats["normalized_matches"] += 1
            return True, "normalized"

        # Tier 3 — Fuzzy match (MinHash LSH)
        mh = build_minhash(norm, num_perm=self.num_perm)
        candidates = self._lsh.query(mh)
        if candidates:
            self.stats["fuzzy_matches"] += 1
            return True, "fuzzy"

        self.stats["clean"] += 1
        return False, "clean"

    def filter_stream(
        self,
        text_iterator: Iterator[str],
    ) -> Iterator[str]:
        """Yield only clean (non-contaminated) documents.

        Parameters
        ----------
        text_iterator : Iterator[str]
            Stream of training documents to filter.

        Yields
        ------
        str
            Clean documents.
        """
        for text in text_iterator:
            contaminated, reason = self.is_contaminated(text)
            if not contaminated:
                yield text
            else:
                logger.debug("Contaminated (%s): %.80s…", reason, text)

    def save_report(self, output_path: str | Path = "reports/contamination_report.json") -> None:
        """Save contamination statistics as JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "jaccard_threshold": self.jaccard_threshold,
                    "num_perm": self.num_perm,
                    "num_references": self._lsh_counter,
                    "stats": self.stats,
                },
                f,
                indent=2,
            )
        logger.info("Contamination report saved → %s", output_path)


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    from src.data.aurora_loader import AuroraLoader

    parser = argparse.ArgumentParser(description="Contamination check pipeline")
    parser.add_argument("--dataset_id", default="Itau-Unibanco/Aurora-PT")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--output", default="reports/contamination_report.json")
    parser.add_argument(
        "--ref_datasets",
        nargs="+",
        default=[
            "eduagarcia/enem_challenge:test:alternatives",
            "eduagarcia-temp/BLUEX:test:alternatives",
            "nilc-nlp/assin2:test:hypothesis",
        ],
        help="Reference datasets in format dataset_id:split:text_column",
    )
    args = parser.parse_args()

    checker = ContaminationChecker(jaccard_threshold=args.threshold)

    # Build contamination index from evaluation test sets
    for ref_spec in args.ref_datasets:
        parts = ref_spec.split(":")
        ds_id = parts[0]
        split = parts[1] if len(parts) > 1 else "test"
        col = parts[2] if len(parts) > 2 else "text"
        checker.add_references_from_dataset(ds_id, split=split, text_column=col)

    # Filter Aurora-PT stream
    # ⚠️ REQUIRES HF_TOKEN for gated dataset access
    loader = AuroraLoader(dataset_id=args.dataset_id)
    clean_count = 0
    for text in checker.filter_stream(loader.stream()):
        clean_count += 1
        if clean_count >= args.num_samples:
            break

    checker.save_report(args.output)
    print(f"Clean documents: {clean_count}")
    print(json.dumps(checker.stats, indent=2))
