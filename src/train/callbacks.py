"""Custom training callbacks for logging, monitoring, and early stopping.

This module provides HuggingFace TrainerCallback implementations that
integrate with the project's local logging system. These callbacks run
during training to track:

- Throughput: tokens/sec, step timing (ThroughputCallback)
- Metrics: loss, learning rate, gradients (LocalMetricsCallback)
- Memory: GPU VRAM usage over time (GPUMemoryCallback)
- Convergence: early stopping on loss plateau (EarlyStoppingOnPlateau)

All numeric metrics are logged to a local JSONL file via MetricsLogger,
providing a W&B-free alternative for experiment tracking. The JSONL format
is append-only and easily parseable for post-hoc analysis.

Usage:
    from src.train.callbacks import ThroughputCallback, LocalMetricsCallback
    from src.utils.logging_utils import MetricsLogger

    logger = MetricsLogger("outputs/train_log.jsonl")
    trainer = Trainer(
        ...,
        callbacks=[ThroughputCallback(logger), LocalMetricsCallback(logger)],
    )
"""

import time

import torch
from transformers import TrainerCallback

from src.utils.logging_utils import MetricsLogger, get_logger

logger = get_logger(__name__)


class ThroughputCallback(TrainerCallback):
    """Track training throughput (tokens/sec, samples/sec).

    Measures wall-clock time per training step and estimates token
    throughput based on batch size and sequence length. Logs every
    `logging_steps` to avoid I/O overhead on every step.

    This is essential for:
    - Comparing hardware configurations (A100 vs H100)
    - Detecting I/O bottlenecks (throughput drops)
    - Estimating total training time

    Args:
        metrics_logger: MetricsLogger instance for persisting metrics.
    """

    def __init__(self, metrics_logger: MetricsLogger):
        self.metrics_logger = metrics_logger
        self.step_start_time = None
        self.total_tokens = 0

    def on_step_begin(self, args, state, control, **kwargs):
        """Record step start time for elapsed computation."""
        self.step_start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        """Compute and log throughput at logging intervals."""
        if self.step_start_time is None:
            return

        elapsed = time.time() - self.step_start_time
        # Effective batch size includes gradient accumulation
        batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        # Estimate tokens: assumes packed sequences fill max_seq_length
        seq_length = getattr(args, "max_seq_length", 8192)
        tokens_per_step = batch_size * seq_length
        self.total_tokens += tokens_per_step

        # Only log at intervals to reduce I/O overhead
        if state.global_step % args.logging_steps == 0:
            throughput = tokens_per_step / max(elapsed, 1e-6)
            self.metrics_logger.log(
                {
                    "throughput_tokens_per_sec": throughput,
                    "step_time_sec": elapsed,
                    "total_tokens_processed": self.total_tokens,
                },
                step=state.global_step,
            )


class LocalMetricsCallback(TrainerCallback):
    """Log all training metrics to local JSONL file.

    Captures every metric emitted by the Trainer (loss, learning rate,
    gradient norm, etc.) and persists them locally. This provides a
    complete training record independent of external services like W&B.

    Also captures evaluation metrics when on_evaluate fires.

    Args:
        metrics_logger: MetricsLogger instance for persisting metrics.
    """

    def __init__(self, metrics_logger: MetricsLogger):
        self.metrics_logger = metrics_logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Persist numeric training metrics from each log event."""
        if logs:
            # Filter to numeric values only (skip strings like "epoch")
            self.metrics_logger.log(
                {k: v for k, v in logs.items() if isinstance(v, (int, float))},
                step=state.global_step,
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Persist evaluation metrics with an 'eval' event marker."""
        if metrics:
            self.metrics_logger.log(
                {"event": "eval", **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}},
                step=state.global_step,
            )


class EarlyStoppingOnPlateau(TrainerCallback):
    """Early stopping when validation loss stops improving.

    Monitors eval_loss and stops training if no improvement is seen
    for `patience` consecutive evaluations. "Improvement" is defined
    as a decrease greater than `threshold` from the best observed loss.

    This prevents wasting compute on training that has converged or
    is beginning to overfit.

    Args:
        patience: Number of evaluations to wait before stopping.
        threshold: Minimum improvement to reset patience counter.

    Example:
        trainer = Trainer(
            ...,
            callbacks=[EarlyStoppingOnPlateau(patience=5, threshold=0.001)],
        )
    """

    def __init__(self, patience: int = 5, threshold: float = 0.001):
        self.patience = patience
        self.threshold = threshold
        self.best_loss = float("inf")
        self.wait = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Check if loss has improved; stop training if patience exhausted."""
        if metrics is None:
            return

        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return

        # Check for meaningful improvement (must beat best by threshold)
        if eval_loss < self.best_loss - self.threshold:
            self.best_loss = eval_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                logger.info(
                    f"Early stopping triggered: no improvement for {self.patience} evals. "
                    f"Best loss: {self.best_loss:.4f}"
                )
                control.should_training_stop = True


class GPUMemoryCallback(TrainerCallback):
    """Log GPU memory usage periodically during training.

    Tracks both allocated memory (actively used by tensors) and reserved
    memory (held by the CUDA allocator). This helps identify:
    - Memory leaks (monotonically increasing allocation)
    - OOM risk (approaching GPU capacity)
    - Optimal batch size tuning

    Logs every 10 * logging_steps to avoid excessive overhead from
    CUDA memory queries.

    Args:
        metrics_logger: MetricsLogger instance for persisting metrics.
    """

    def __init__(self, metrics_logger: MetricsLogger):
        self.metrics_logger = metrics_logger

    def on_step_end(self, args, state, control, **kwargs):
        """Log GPU memory at reduced frequency (every 10x logging_steps)."""
        if state.global_step % (args.logging_steps * 10) == 0 and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9  # Convert bytes to GB
            reserved = torch.cuda.memory_reserved() / 1e9
            self.metrics_logger.log(
                {
                    "gpu_memory_allocated_gb": allocated,
                    "gpu_memory_reserved_gb": reserved,
                },
                step=state.global_step,
            )
