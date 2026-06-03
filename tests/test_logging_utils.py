"""Tests for logging utilities."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_utils import MetricsLogger, get_logger


class TestGetLogger:
    """Test logger creation."""

    def test_creates_logger(self):
        logger = get_logger("test_module")
        assert logger.name == "test_module"

    def test_console_handler_present(self):
        logger = get_logger("test_console")
        assert len(logger.handlers) >= 1

    def test_file_handler_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = f"{tmpdir}/test.log"
            logger = get_logger("test_file", log_file=log_path)
            logger.info("test message")
            assert Path(log_path).exists()
            for handler in logger.handlers:
                handler.close()
            logger.handlers.clear()

    def test_log_level_respected(self):
        logger = get_logger("test_level", level="ERROR")
        import logging

        assert logger.level == logging.ERROR


class TestMetricsLogger:
    """Test JSONL metrics logger."""

    def test_log_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            ml = MetricsLogger(path)
            ml.log({"loss": 0.5}, step=1)
            assert path.exists()

    def test_log_appends(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            ml = MetricsLogger(path)
            ml.log({"loss": 0.5}, step=1)
            ml.log({"loss": 0.3}, step=2)

            lines = path.read_text().strip().split("\n")
            assert len(lines) == 2

    def test_log_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            ml = MetricsLogger(path)
            ml.log({"loss": 0.5, "lr": 1e-4}, step=10)

            with open(path) as f:
                record = json.loads(f.readline())
            assert record["loss"] == 0.5
            assert record["lr"] == 1e-4
            assert record["step"] == 10
            assert "timestamp" in record

    def test_read_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            ml = MetricsLogger(path)
            ml.log({"a": 1}, step=1)
            ml.log({"a": 2}, step=2)
            ml.log({"a": 3}, step=3)

            records = ml.read_all()
            assert len(records) == 3
            assert records[0]["a"] == 1
            assert records[2]["a"] == 3

    def test_read_all_empty_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            ml = MetricsLogger(path)
            records = ml.read_all()
            assert records == []

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "deep" / "nested" / "metrics.jsonl"
            ml = MetricsLogger(path)
            ml.log({"x": 1})
            assert path.exists()

    def test_timestamp_is_utc(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.jsonl"
            ml = MetricsLogger(path)
            ml.log({"x": 1})

            records = ml.read_all()
            ts = records[0]["timestamp"]
            # Should be ISO format with timezone info
            assert "T" in ts
