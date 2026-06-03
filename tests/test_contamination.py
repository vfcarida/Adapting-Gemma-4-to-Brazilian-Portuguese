"""Tests for contamination detection module.

Pure-function tests (normalize, hash, ngrams) run without datasketch.
Integration tests require datasketch and are skipped if unavailable.
"""

import hashlib
import re
import sys
import unicodedata
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# --- Re-implement pure functions locally to avoid datasketch import ---


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def compute_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode()).hexdigest()


def ngrams(text: str, n: int = 5) -> set[str]:
    words = text.split()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


# Check if datasketch is available for integration tests
try:
    from datasketch import MinHash  # noqa: F401

    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False


class TestNormalizeText:
    """Test text normalization for contamination comparison."""

    def test_lowercases(self):
        assert "hello world" in normalize_text("Hello World")

    def test_removes_punctuation(self):
        result = normalize_text("Hello, world! How are you?")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_collapses_whitespace(self):
        result = normalize_text("hello   world\t\ttab")
        assert "   " not in result
        assert "\t" not in result

    def test_unicode_normalization(self):
        # NFKD decomposes accented characters
        result = normalize_text("cafe\u0301")  # café with combining accent
        assert "e" in result

    def test_strips_edges(self):
        result = normalize_text("  hello  ")
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_only_punctuation(self):
        result = normalize_text("!!!???...")
        assert result == ""


class TestComputeHash:
    """Test hash computation for exact match."""

    def test_deterministic(self):
        h1 = compute_hash("test text")
        h2 = compute_hash("test text")
        assert h1 == h2

    def test_different_texts_different_hashes(self):
        h1 = compute_hash("text one")
        h2 = compute_hash("text two")
        assert h1 != h2

    def test_normalization_applied(self):
        # Same text with different formatting should produce same hash
        h1 = compute_hash("Hello World")
        h2 = compute_hash("hello world")
        assert h1 == h2

    def test_punctuation_invariant(self):
        h1 = compute_hash("hello world")
        h2 = compute_hash("hello, world!")
        assert h1 == h2

    def test_returns_hex_string(self):
        h = compute_hash("test")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex length


class TestNgrams:
    """Test n-gram extraction."""

    def test_basic_ngrams(self):
        result = ngrams("a b c d e", n=3)
        assert "a b c" in result
        assert "b c d" in result
        assert "c d e" in result
        assert len(result) == 3

    def test_text_shorter_than_n(self):
        result = ngrams("a b", n=5)
        # Should return the full text as single gram
        assert "a b" in result
        assert len(result) == 1

    def test_exact_length_n(self):
        result = ngrams("a b c", n=3)
        assert "a b c" in result
        assert len(result) == 1

    def test_single_word(self):
        result = ngrams("hello", n=3)
        assert "hello" in result

    def test_n_equals_1(self):
        result = ngrams("a b c", n=1)
        assert "a" in result
        assert "b" in result
        assert "c" in result
        assert len(result) == 3

    def test_large_n(self):
        # 20 distinct words: "w0 w1 w2 ... w19"
        words = [f"w{i}" for i in range(20)]
        text = " ".join(words)
        result = ngrams(text, n=10)
        assert len(result) == 11  # 20 - 10 + 1

    def test_empty_string(self):
        result = ngrams("", n=5)
        # Empty string split -> [''], len < n -> returns {""}
        assert len(result) == 1


@pytest.mark.skipif(not HAS_DATASKETCH, reason="datasketch not installed")
class TestContaminationIntegration:
    """Integration tests for ContaminationChecker (requires datasketch)."""

    def test_exact_match_found(self):
        """Two identical texts should be detected as exact match."""
        from src.data.contamination_checks import ContaminationChecker

        benchmark_texts = ["This is a test sentence for the benchmark"]
        train_texts = ["This is a test sentence for the benchmark"]

        checker = ContaminationChecker(benchmark_texts, "test_bench")
        result = checker.check_exact(train_texts)

        assert result["matches"] == 1
        assert result["contamination_rate"] == 1.0

    @pytest.mark.skipif(not HAS_DATASKETCH, reason="datasketch not installed")
    def test_exact_match_not_found(self):
        """Different texts should not match."""
        from src.data.contamination_checks import ContaminationChecker

        benchmark_texts = ["Benchmark text about science"]
        train_texts = ["Completely different training text"]

        checker = ContaminationChecker(benchmark_texts, "test_bench")
        result = checker.check_exact(train_texts)

        assert result["matches"] == 0
        assert result["contamination_rate"] == 0.0

    @pytest.mark.skipif(not HAS_DATASKETCH, reason="datasketch not installed")
    def test_normalized_match(self):
        """Texts differing only in case/punctuation should match when normalized."""
        from src.data.contamination_checks import ContaminationChecker

        benchmark_texts = ["Hello, World! This is a test."]
        train_texts = ["hello world this is a test"]

        checker = ContaminationChecker(benchmark_texts, "test_bench")
        result = checker.check_normalized(train_texts)

        assert result["matches"] == 1

    @pytest.mark.skipif(not HAS_DATASKETCH, reason="datasketch not installed")
    def test_no_contamination_empty_benchmark(self):
        """Empty benchmark should find no matches."""
        from src.data.contamination_checks import ContaminationChecker

        benchmark_texts = []
        # Should handle gracefully
        checker = ContaminationChecker(benchmark_texts, "empty")
        result = checker.check_exact(["some training text"])
        assert result["matches"] == 0

    @pytest.mark.skipif(not HAS_DATASKETCH, reason="datasketch not installed")
    def test_multiple_benchmarks(self):
        """run_contamination_report should handle multiple benchmarks."""
        import tempfile

        from src.data.contamination_checks import run_contamination_report

        benchmarks = {
            "bench_a": ["text A for benchmark"],
            "bench_b": ["text B for benchmark"],
        }
        train_texts = ["text A for benchmark", "unrelated text"]

        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_contamination_report(train_texts, benchmarks, tmpdir)
            assert "bench_a" in report["benchmarks"]
            assert "bench_b" in report["benchmarks"]
            assert "summary" in report
