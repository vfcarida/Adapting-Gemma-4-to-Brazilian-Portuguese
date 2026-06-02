"""Contamination detection between training data and evaluation benchmarks."""

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from tqdm import tqdm

try:
    from datasketch import MinHash, MinHashLSH
    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join([c for c in text if not unicodedata.combining(c)])
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def compute_hash(text: str) -> str:
    """Compute SHA-256 hash of normalized text."""
    return hashlib.sha256(normalize_text(text).encode()).hexdigest()


def ngrams(text: str, n: int = 5) -> set[str]:
    """Extract word n-grams from text."""
    words = text.split()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


class ContaminationChecker:
    """Check for data contamination between training corpus and benchmarks."""

    def __init__(self, benchmark_texts: list[str], benchmark_name: str):
        self.benchmark_name = benchmark_name
        self.benchmark_texts = benchmark_texts
        self.benchmark_normalized = [normalize_text(t) for t in benchmark_texts]
        self.benchmark_hashes = {compute_hash(t) for t in benchmark_texts}
        self._build_minhash_index()

    def _build_minhash_index(self, num_perm: int = 128):
        """Build MinHash LSH index for fuzzy matching."""
        self.num_perm = num_perm
        self.minhashes = {}

        if not HAS_DATASKETCH:
            self.lsh = None
            return

        self.lsh = MinHashLSH(threshold=0.5, num_perm=num_perm)

        for i, text in enumerate(self.benchmark_normalized):
            mh = MinHash(num_perm=num_perm)
            for gram in ngrams(text):
                mh.update(gram.encode())
            self.minhashes[i] = mh
            try:
                self.lsh.insert(f"bench_{i}", mh)
            except ValueError:
                pass  # Duplicate

    def check_exact(self, train_texts: list[str]) -> dict[str, Any]:
        """Check exact string overlap."""
        matches = []
        for i, text in enumerate(tqdm(train_texts, desc="Exact match")):
            h = compute_hash(text)
            if h in self.benchmark_hashes:
                matches.append({"train_idx": i, "hash": h})
        return {
            "method": "exact",
            "matches": len(matches),
            "total_checked": len(train_texts),
            "contamination_rate": len(matches) / max(len(train_texts), 1),
            "details": matches[:100],
        }

    def check_normalized(self, train_texts: list[str]) -> dict[str, Any]:
        """Check normalized text overlap."""
        bench_set = set(self.benchmark_normalized)
        matches = []
        for i, text in enumerate(tqdm(train_texts, desc="Normalized match")):
            norm = normalize_text(text)
            if norm in bench_set:
                matches.append({"train_idx": i})
        return {
            "method": "normalized",
            "matches": len(matches),
            "total_checked": len(train_texts),
            "contamination_rate": len(matches) / max(len(train_texts), 1),
            "details": matches[:100],
        }

    def check_fuzzy(self, train_texts: list[str], threshold: float = 0.7) -> dict[str, Any]:
        """Check fuzzy overlap using MinHash LSH."""
        if not HAS_DATASKETCH or self.lsh is None:
            return {
                "method": "fuzzy",
                "threshold": threshold,
                "matches": 0,
                "total_checked": len(train_texts),
                "contamination_rate": 0.0,
                "details": [],
                "warning": "datasketch not installed — fuzzy check skipped",
            }

        matches = []
        for i, text in enumerate(tqdm(train_texts, desc="Fuzzy match")):
            norm = normalize_text(text)
            mh = MinHash(num_perm=self.num_perm)
            for gram in ngrams(norm):
                mh.update(gram.encode())

            candidates = self.lsh.query(mh)
            for cand in candidates:
                bench_idx = int(cand.split("_")[1])
                # Compute actual Jaccard
                jaccard = mh.jaccard(self.minhashes[bench_idx])
                if jaccard >= threshold:
                    matches.append({
                        "train_idx": i,
                        "bench_idx": bench_idx,
                        "jaccard": float(jaccard),
                    })
                    break  # One match is enough

        return {
            "method": "fuzzy",
            "threshold": threshold,
            "matches": len(matches),
            "total_checked": len(train_texts),
            "contamination_rate": len(matches) / max(len(train_texts), 1),
            "details": matches[:100],
        }

    def check_ngram_overlap(
        self, train_texts: list[str], n: int = 10, threshold: float = 0.5
    ) -> dict[str, Any]:
        """Check n-gram overlap ratio."""
        # Build benchmark n-gram set
        bench_ngrams: set[str] = set()
        for text in self.benchmark_normalized:
            bench_ngrams.update(ngrams(text, n))

        matches = []
        for i, text in enumerate(tqdm(train_texts, desc=f"{n}-gram overlap")):
            norm = normalize_text(text)
            train_ng = ngrams(norm, n)
            if not train_ng:
                continue
            overlap = len(train_ng & bench_ngrams) / len(train_ng)
            if overlap >= threshold:
                matches.append({"train_idx": i, "overlap_ratio": float(overlap)})

        return {
            "method": f"ngram_{n}",
            "threshold": threshold,
            "matches": len(matches),
            "total_checked": len(train_texts),
            "contamination_rate": len(matches) / max(len(train_texts), 1),
            "details": matches[:100],
        }

    def run_all_checks(self, train_texts: list[str]) -> dict[str, Any]:
        """Run all contamination checks."""
        results = {
            "benchmark": self.benchmark_name,
            "benchmark_size": len(self.benchmark_texts),
            "train_size": len(train_texts),
            "checks": {
                "exact": self.check_exact(train_texts),
                "normalized": self.check_normalized(train_texts),
                "fuzzy": self.check_fuzzy(train_texts),
                "ngram_10": self.check_ngram_overlap(train_texts, n=10),
            },
        }
        return results


def run_contamination_report(
    train_texts: list[str],
    benchmarks: dict[str, list[str]],
    output_dir: str | Path = "outputs/contamination",
) -> dict[str, Any]:
    """Run contamination checks against all benchmarks and save report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_report = {"benchmarks": {}}

    for name, bench_texts in benchmarks.items():
        logger.info(f"Checking contamination against: {name} ({len(bench_texts)} samples)")
        checker = ContaminationChecker(bench_texts, name)
        result = checker.run_all_checks(train_texts)
        full_report["benchmarks"][name] = result

        # Save per-benchmark
        with open(output_dir / f"{name}.json", "w") as f:
            json.dump(result, f, indent=2)

    # Summary
    summary = {}
    for name, result in full_report["benchmarks"].items():
        summary[name] = {
            method: check["contamination_rate"]
            for method, check in result["checks"].items()
        }
    full_report["summary"] = summary

    with open(output_dir / "full_report.json", "w") as f:
        json.dump(full_report, f, indent=2)

    logger.info(f"Contamination report saved to {output_dir}")
    return full_report
