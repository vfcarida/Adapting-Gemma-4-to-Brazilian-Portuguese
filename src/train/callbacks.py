"""Custom training callbacks for logging, monitoring, and early stopping.

This module provides HuggingFace TrainerCallback implementations that
integrate with the project's local logging system. These callbacks run
during training to track:

- Throughput: tokens/sec, step timing (ThroughputCallback)
- Metrics: loss, learning rate, gradients (LocalMetricsCallback)
- Memory: GPU VRAM usage over time (GPUMemoryCallback)
- Convergence: early stopping on loss plateau (EarlyStoppingOnPlateau)
- W&B: remote dashboard logging via Weights & Biases (WandBCallback)

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

import os
import time
from typing import Any

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

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


class ForgettingMonitorCallback(TrainerCallback):
    """Monitor catastrophic forgetting by tracking English perplexity during training.

    Periodically computes perplexity on a small English validation set (held constant).
    This is the most sensitive early-warning signal for forgetting — if English perplexity
    starts rising, the replay ratio may need adjustment.

    The monitor evaluates every `eval_interval_steps` and logs:
    - en_perplexity: Perplexity on English validation set
    - en_loss: Cross-entropy loss on English validation set
    - forgetting_delta: Change from initial English perplexity

    Args:
        en_eval_texts: List of English text samples for perplexity computation.
        eval_interval_steps: How often to evaluate (default: every 500 steps).
        max_eval_samples: Maximum samples to use per evaluation (for speed).
        metrics_logger: MetricsLogger for persisting results.
    """

    def __init__(
        self,
        en_eval_texts: list[str] | None = None,
        eval_interval_steps: int = 500,
        max_eval_samples: int = 100,
        metrics_logger: MetricsLogger | None = None,
    ):
        self.en_eval_texts = en_eval_texts
        self.eval_interval_steps = eval_interval_steps
        self.max_eval_samples = max_eval_samples
        self.metrics_logger = metrics_logger
        self.initial_perplexity: float | None = None
        self._tokenizer = None
        self._eval_encodings = None

    def on_train_begin(self, args, state, control, model=None, processing_class=None, **kwargs):
        """Prepare English evaluation data at training start."""
        if not self.en_eval_texts:
            self._load_default_en_data()

        if not self.en_eval_texts:
            logger.warning("ForgettingMonitor: No English eval data available, disabling")
            return

        # Get tokenizer from trainer
        self._tokenizer = processing_class
        if self._tokenizer is None:
            logger.warning("ForgettingMonitor: No tokenizer available, disabling")
            return

        # Pre-tokenize English eval set
        texts = self.en_eval_texts[:self.max_eval_samples]
        self._eval_encodings = self._tokenizer(
            texts, truncation=True, max_length=512, padding=True, return_tensors="pt"
        )

        # Compute initial perplexity (baseline before any training)
        if model is not None:
            self.initial_perplexity = self._compute_perplexity(model)
            logger.info(f"ForgettingMonitor: Initial EN perplexity = {self.initial_perplexity:.2f}")
            if self.metrics_logger:
                self.metrics_logger.log({
                    "en_perplexity": self.initial_perplexity,
                    "en_forgetting_delta": 0.0,
                    "event": "forgetting_baseline",
                }, step=0)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Evaluate English perplexity at regular intervals."""
        if self._eval_encodings is None or model is None:
            return

        if state.global_step % self.eval_interval_steps != 0:
            return
        if state.global_step == 0:
            return

        ppl = self._compute_perplexity(model)
        delta = ppl - self.initial_perplexity if self.initial_perplexity else 0.0

        logger.info(
            f"ForgettingMonitor [step {state.global_step}]: "
            f"EN ppl={ppl:.2f} (Δ={delta:+.2f} from baseline)"
        )

        if self.metrics_logger:
            self.metrics_logger.log({
                "en_perplexity": ppl,
                "en_forgetting_delta": delta,
                "event": "forgetting_check",
            }, step=state.global_step)

        # Warn if forgetting is significant (>20% increase)
        if self.initial_perplexity and ppl > self.initial_perplexity * 1.2:
            logger.warning(
                f"FORGETTING ALERT: EN perplexity increased by "
                f"{(ppl/self.initial_perplexity - 1)*100:.1f}% — consider increasing replay ratio"
            )

    def _compute_perplexity(self, model) -> float:
        """Compute perplexity on the English eval set."""
        import math
        model.eval()
        device = next(model.parameters()).device

        input_ids = self._eval_encodings["input_ids"].to(device)
        attention_mask = self._eval_encodings["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss.item()

        model.train()
        return math.exp(min(loss, 100))  # Cap to avoid overflow

    def _load_default_en_data(self):
        """Load a small English validation set from FineWeb-Edu."""
        try:
            from datasets import load_dataset
            logger.info("ForgettingMonitor: Loading English eval data from FineWeb-Edu...")
            ds = load_dataset(
                "HuggingFaceFW/fineweb-edu", "sample-10BT",
                split="train", streaming=True
            )
            samples = []
            for i, ex in enumerate(ds):
                if i >= self.max_eval_samples:
                    break
                text = ex.get("text", "")
                if len(text) > 100:
                    samples.append(text[:512])  # Truncate for efficiency
            self.en_eval_texts = samples
            logger.info(f"ForgettingMonitor: Loaded {len(samples)} English samples")
        except Exception as e:
            logger.warning(f"ForgettingMonitor: Could not load English data: {e}")
            self.en_eval_texts = []


class WandBCallback(TrainerCallback):
    """Weights & Biases integration for remote experiment tracking.

    Initializes a W&B run on training start and logs all metrics.
    Only activates if WANDB_API_KEY is set in environment or wandb
    is already logged in. Falls back gracefully if wandb is not installed.

    Features:
    - Logs all training metrics (loss, lr, grad_norm, throughput)
    - Logs evaluation metrics with "eval/" prefix
    - Saves config as W&B run config for filtering/grouping
    - Logs system metrics (GPU utilization, memory) automatically

    Args:
        project: W&B project name.
        config: Full experiment config dict (saved as run config).
        tags: Optional list of tags for filtering runs.
    """

    def __init__(
        self,
        project: str = "gemma4-pt-br-adaptation",
        config: dict | None = None,
        tags: list[str] | None = None,
    ):
        self.project = project
        self.config = config or {}
        self.tags = tags or []
        self._wandb = None
        self._run = None

    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize W&B run at training start."""
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            logger.warning("wandb not installed — WandBCallback disabled. pip install wandb")
            return

        # Only init if API key is available
        if not (os.environ.get("WANDB_API_KEY") or wandb.api.api_key):
            logger.warning("WANDB_API_KEY not set — WandBCallback disabled")
            self._wandb = None
            return

        run_name = self.config.get("experiment", {}).get("name", "train")
        group = self.config.get("experiment", {}).get("trilha", None)

        self._run = wandb.init(
            project=self.project,
            name=run_name,
            group=group,
            config=self.config,
            tags=self.tags,
            resume="allow",
        )
        logger.info(f"W&B run initialized: {self._run.url}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics to W&B."""
        if not self._wandb or not self._run or not logs:
            return
        # Filter to numeric and log with step
        numeric_logs = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
        if numeric_logs:
            self._wandb.log(numeric_logs, step=state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log evaluation metrics to W&B with eval/ prefix."""
        if not self._wandb or not self._run or not metrics:
            return
        eval_metrics = {
            f"eval/{k}" if not k.startswith("eval_") else k: v
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        }
        if eval_metrics:
            self._wandb.log(eval_metrics, step=state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        """Finish W&B run cleanly."""
        if self._run:
            self._run.finish()
            logger.info("W&B run finished")
