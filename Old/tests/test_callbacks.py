"""Tests for training callbacks (requires torch)."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytest.importorskip("torch")

from src.train.callbacks import (
    EarlyStoppingOnPlateau,
    GPUMemoryCallback,
    LocalMetricsCallback,
    ThroughputCallback,
)


class MockMetricsLogger:
    """Mock MetricsLogger that records all logged entries."""

    def __init__(self):
        self.entries = []

    def log(self, data, step=None):
        self.entries.append({"data": data, "step": step})


class MockArgs:
    """Mock TrainingArguments with common fields."""

    def __init__(self, batch_size=4, grad_accum=2, logging_steps=10, max_seq_length=8192):
        self.per_device_train_batch_size = batch_size
        self.gradient_accumulation_steps = grad_accum
        self.logging_steps = logging_steps
        self.max_seq_length = max_seq_length


class MockState:
    """Mock TrainerState."""

    def __init__(self, global_step=0):
        self.global_step = global_step


class MockControl:
    """Mock TrainerControl."""

    def __init__(self):
        self.should_training_stop = False


class TestThroughputCallback:
    """Test throughput tracking callback."""

    def test_logs_at_logging_step(self):
        logger = MockMetricsLogger()
        cb = ThroughputCallback(logger)
        args = MockArgs(batch_size=4, grad_accum=2, logging_steps=10)
        state = MockState(global_step=10)
        control = MockControl()

        cb.on_step_begin(args, state, control)
        cb.on_step_end(args, state, control)

        assert len(logger.entries) == 1
        entry = logger.entries[0]["data"]
        assert "throughput_tokens_per_sec" in entry
        assert "step_time_sec" in entry
        assert "total_tokens_processed" in entry

    def test_skips_non_logging_step(self):
        logger = MockMetricsLogger()
        cb = ThroughputCallback(logger)
        args = MockArgs(logging_steps=10)
        state = MockState(global_step=7)  # Not a multiple of 10
        control = MockControl()

        cb.on_step_begin(args, state, control)
        cb.on_step_end(args, state, control)

        assert len(logger.entries) == 0

    def test_accumulates_total_tokens(self):
        logger = MockMetricsLogger()
        cb = ThroughputCallback(logger)
        args = MockArgs(batch_size=2, grad_accum=1, logging_steps=1, max_seq_length=1024)

        for step in range(1, 4):
            state = MockState(global_step=step)
            control = MockControl()
            cb.on_step_begin(args, state, control)
            cb.on_step_end(args, state, control)

        # 3 steps * 2 batch * 1024 seq_len = 6144
        last_entry = logger.entries[-1]["data"]
        assert last_entry["total_tokens_processed"] == 6144

    def test_handles_no_step_begin(self):
        """on_step_end without on_step_begin should not crash."""
        logger = MockMetricsLogger()
        cb = ThroughputCallback(logger)
        args = MockArgs(logging_steps=1)
        state = MockState(global_step=1)
        control = MockControl()

        cb.on_step_end(args, state, control)
        assert len(logger.entries) == 0


class TestLocalMetricsCallback:
    """Test local metrics logging callback."""

    def test_logs_numeric_metrics(self):
        logger = MockMetricsLogger()
        cb = LocalMetricsCallback(logger)
        args = MockArgs()
        state = MockState(global_step=5)
        control = MockControl()

        logs = {"loss": 2.5, "learning_rate": 1e-4, "epoch": "1"}
        cb.on_log(args, state, control, logs=logs)

        assert len(logger.entries) == 1
        data = logger.entries[0]["data"]
        assert "loss" in data
        assert "learning_rate" in data
        # String values should be filtered out
        assert "epoch" not in data

    def test_handles_none_logs(self):
        logger = MockMetricsLogger()
        cb = LocalMetricsCallback(logger)
        cb.on_log(MockArgs(), MockState(), MockControl(), logs=None)
        assert len(logger.entries) == 0

    def test_on_evaluate_logs_eval_metrics(self):
        logger = MockMetricsLogger()
        cb = LocalMetricsCallback(logger)
        args = MockArgs()
        state = MockState(global_step=100)
        control = MockControl()

        metrics = {"eval_loss": 1.8, "eval_accuracy": 0.75}
        cb.on_evaluate(args, state, control, metrics=metrics)

        assert len(logger.entries) == 1
        data = logger.entries[0]["data"]
        assert data["event"] == "eval"
        assert data["eval_loss"] == 1.8

    def test_on_evaluate_handles_none(self):
        logger = MockMetricsLogger()
        cb = LocalMetricsCallback(logger)
        cb.on_evaluate(MockArgs(), MockState(), MockControl(), metrics=None)
        assert len(logger.entries) == 0


class TestEarlyStoppingOnPlateau:
    """Test early stopping callback."""

    def test_does_not_stop_when_improving(self):
        cb = EarlyStoppingOnPlateau(patience=3, threshold=0.001)
        control = MockControl()
        args = MockArgs()
        state = MockState()

        # Each eval shows improvement
        for loss in [2.0, 1.5, 1.0, 0.5]:
            cb.on_evaluate(args, state, control, metrics={"eval_loss": loss})
            assert control.should_training_stop is False

    def test_stops_after_patience_exhausted(self):
        cb = EarlyStoppingOnPlateau(patience=3, threshold=0.001)
        control = MockControl()
        args = MockArgs()
        state = MockState()

        # Initial good loss
        cb.on_evaluate(args, state, control, metrics={"eval_loss": 1.0})
        assert control.should_training_stop is False

        # No improvement for 3 evals (patience=3)
        cb.on_evaluate(args, state, control, metrics={"eval_loss": 1.0})
        assert control.should_training_stop is False
        cb.on_evaluate(args, state, control, metrics={"eval_loss": 1.001})
        assert control.should_training_stop is False
        cb.on_evaluate(args, state, control, metrics={"eval_loss": 1.002})
        assert control.should_training_stop is True

    def test_resets_patience_on_improvement(self):
        cb = EarlyStoppingOnPlateau(patience=2, threshold=0.001)
        control = MockControl()
        args = MockArgs()
        state = MockState()

        cb.on_evaluate(args, state, control, metrics={"eval_loss": 2.0})
        cb.on_evaluate(args, state, control, metrics={"eval_loss": 2.0})  # wait=1
        # Improvement resets counter
        cb.on_evaluate(args, state, control, metrics={"eval_loss": 1.5})
        assert cb.wait == 0
        assert control.should_training_stop is False

    def test_handles_no_eval_loss(self):
        cb = EarlyStoppingOnPlateau(patience=2)
        control = MockControl()
        args = MockArgs()
        state = MockState()

        # No eval_loss key in metrics
        cb.on_evaluate(args, state, control, metrics={"other_metric": 0.5})
        assert control.should_training_stop is False

    def test_handles_none_metrics(self):
        cb = EarlyStoppingOnPlateau(patience=2)
        control = MockControl()
        cb.on_evaluate(MockArgs(), MockState(), control, metrics=None)
        assert control.should_training_stop is False

    def test_threshold_sensitivity(self):
        """Improvement must exceed threshold to count."""
        cb = EarlyStoppingOnPlateau(patience=2, threshold=0.1)
        control = MockControl()
        args = MockArgs()
        state = MockState()

        cb.on_evaluate(args, state, control, metrics={"eval_loss": 1.0})
        # Tiny improvement (0.05) is below threshold (0.1)
        cb.on_evaluate(args, state, control, metrics={"eval_loss": 0.95})
        assert cb.wait == 1  # Not counted as improvement


class TestGPUMemoryCallback:
    """Test GPU memory logging callback."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=5e9)
    @patch("torch.cuda.memory_reserved", return_value=8e9)
    def test_logs_memory_at_interval(self, mock_reserved, mock_allocated, mock_avail):
        logger = MockMetricsLogger()
        cb = GPUMemoryCallback(logger)
        args = MockArgs(logging_steps=10)
        # global_step must be multiple of logging_steps * 10 = 100
        state = MockState(global_step=100)
        control = MockControl()

        cb.on_step_end(args, state, control)

        assert len(logger.entries) == 1
        data = logger.entries[0]["data"]
        assert data["gpu_memory_allocated_gb"] == pytest.approx(5.0)
        assert data["gpu_memory_reserved_gb"] == pytest.approx(8.0)

    @patch("torch.cuda.is_available", return_value=True)
    def test_skips_non_interval_steps(self, mock_avail):
        logger = MockMetricsLogger()
        cb = GPUMemoryCallback(logger)
        args = MockArgs(logging_steps=10)
        state = MockState(global_step=50)  # Not a multiple of 100
        control = MockControl()

        cb.on_step_end(args, state, control)
        assert len(logger.entries) == 0

    @patch("torch.cuda.is_available", return_value=False)
    def test_skips_when_no_cuda(self, mock_avail):
        logger = MockMetricsLogger()
        cb = GPUMemoryCallback(logger)
        args = MockArgs(logging_steps=10)
        state = MockState(global_step=100)
        control = MockControl()

        cb.on_step_end(args, state, control)
        assert len(logger.entries) == 0
