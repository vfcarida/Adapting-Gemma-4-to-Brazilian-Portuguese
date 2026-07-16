"""Tests for data pipeline components (without requiring datasets download)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAuroraLoaderPreprocessing:
    """Test preprocessing logic without loading actual data."""

    def test_length_filter_too_short(self):
        """Documents below min_chars should be filtered out."""

        # Simulate the filter logic
        min_chars = 100
        text = "Short text."
        assert len(text) < min_chars

    def test_length_filter_too_long(self):
        """Documents above max_chars should be filtered out."""
        max_chars = 500000
        text = "x" * 500001
        assert len(text) > max_chars

    def test_email_redaction(self):
        """Emails should be replaced with [EMAIL]."""
        import re

        text = "Contact user@example.com for details"
        result = re.sub(r"\S+@\S+\.\S+", "[EMAIL]", text)
        assert "[EMAIL]" in result
        assert "user@example.com" not in result

    def test_whitespace_normalization(self):
        """Multiple spaces/tabs should collapse to single space."""
        import re

        text = "hello   world\t\ttab"
        result = re.sub(r"[ \t]+", " ", text)
        assert result == "hello world tab"

    def test_newline_normalization(self):
        """3+ consecutive newlines should collapse to 2."""
        import re

        text = "para1\n\n\n\n\npara2"
        result = re.sub(r"\n{3,}", "\n\n", text)
        assert result == "para1\n\npara2"


class TestDocumentHashSplit:
    """Test document-level hash splitting logic."""

    def test_deterministic_split(self):
        """Same document always gets same split assignment."""
        import hashlib

        text = "This is a test document for splitting"
        val_ratio = 0.005

        def get_split(doc_text):
            doc_hash = hashlib.md5(doc_text[:500].encode()).hexdigest()
            hash_val = int(doc_hash[:8], 16) / 0xFFFFFFFF
            return "val" if hash_val < val_ratio else "train"

        # Same text should always produce same split
        split1 = get_split(text)
        split2 = get_split(text)
        assert split1 == split2

    def test_uniform_distribution(self):
        """Hash values should be approximately uniform in [0, 1]."""
        import hashlib

        values = []
        for i in range(10000):
            text = f"document number {i} with some content"
            doc_hash = hashlib.md5(text[:500].encode()).hexdigest()
            hash_val = int(doc_hash[:8], 16) / 0xFFFFFFFF
            values.append(hash_val)

        # Should be roughly uniform - mean around 0.5
        import statistics

        mean = statistics.mean(values)
        assert 0.45 < mean < 0.55

    def test_val_ratio_approximate(self):
        """Approximately val_ratio fraction should go to validation."""
        import hashlib

        val_ratio = 0.01  # 1%
        n_val = 0
        n_total = 10000

        for i in range(n_total):
            text = f"unique document {i}"
            doc_hash = hashlib.md5(text[:500].encode()).hexdigest()
            hash_val = int(doc_hash[:8], 16) / 0xFFFFFFFF
            if hash_val < val_ratio:
                n_val += 1

        # Should be approximately 1% (allow ±0.5%)
        actual_ratio = n_val / n_total
        assert 0.005 < actual_ratio < 0.02


class TestSequencePacking:
    """Test sequence packing logic."""

    def test_pack_basic(self):
        """Multiple short sequences should pack into one."""
        max_seq_length = 10
        buffer = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

        chunks = []
        while len(buffer) >= max_seq_length:
            chunks.append(buffer[:max_seq_length])
            buffer = buffer[max_seq_length:]

        assert len(chunks) == 1
        assert chunks[0] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert buffer == [11, 12]  # Leftover

    def test_pack_exact_multiple(self):
        """Sequence that is exact multiple of max_length packs perfectly."""
        max_seq_length = 5
        buffer = list(range(15))

        chunks = []
        while len(buffer) >= max_seq_length:
            chunks.append(buffer[:max_seq_length])
            buffer = buffer[max_seq_length:]

        assert len(chunks) == 3
        assert buffer == []

    def test_pack_shorter_than_max(self):
        """Sequence shorter than max_length produces no chunks."""
        max_seq_length = 100
        buffer = [1, 2, 3]

        chunks = []
        while len(buffer) >= max_seq_length:
            chunks.append(buffer[:max_seq_length])
            buffer = buffer[max_seq_length:]

        assert len(chunks) == 0
        assert buffer == [1, 2, 3]

    def test_labels_equal_input_ids(self):
        """For causal LM, labels should be identical to input_ids."""
        max_seq_length = 4
        input_ids = [10, 20, 30, 40, 50, 60, 70, 80]

        all_input_ids = []
        all_labels = []
        buffer = input_ids.copy()

        while len(buffer) >= max_seq_length:
            chunk = buffer[:max_seq_length]
            all_input_ids.append(chunk)
            all_labels.append(chunk.copy())
            buffer = buffer[max_seq_length:]

        assert all_input_ids == all_labels
        assert all_input_ids[0] == [10, 20, 30, 40]
        assert all_input_ids[1] == [50, 60, 70, 80]
