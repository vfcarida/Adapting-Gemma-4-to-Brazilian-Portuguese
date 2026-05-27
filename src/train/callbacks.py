"""
src/train/callbacks.py
──────────────────────
Custom HuggingFace ``TrainerCallback`` subclasses for enhanced
training observability.

Callbacks:
  • JSONLLoggingCallback    — writes metrics to local JSONL file
  • PerplexityCallback      — computes validation perplexity
  • EarlyStoppingWithPatience — stop on val loss plateau
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from src.utils.logging_utils import JSONLWriter, get_logger

logger = get_logger(__name__)


class JSONLLoggingCallback(TrainerCallback):
    """Write training metrics to a local JSONL file.

    Each log event is appended as a JSON line with all metrics from
    the Trainer's log history.

    Parameters
    ----------
    log_dir : str | Path
        Directory for the JSONL output file.
    filename : str
        Name of the JSONL file.
    """

    def __init__(self, log_dir: str | Path, filename: str = "train_metrics.jsonl") -> None:
        self._writer = JSONLWriter(log_dir, filename)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if logs is None:
            return
        self._writer.write(
            step=state.global_step,
            epoch=state.epoch,
            **{k: v for k, v in logs.items() if isinstance(v, (int, float, str))},
        )


class PerplexityCallback(TrainerCallback):
    """Compute and log validation perplexity at each evaluation step.

    Perplexity is computed as ``exp(eval_loss)`` from the Trainer's
    reported evaluation loss.

    Parameters
    ----------
    log_dir : str | Path
        Directory for perplexity JSONL log.
    """

    def __init__(self, log_dir: str | Path | None = None) -> None:
        self._writer = JSONLWriter(log_dir, "perplexity.jsonl") if log_dir else None

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if metrics is None:
            return

        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None and not math.isnan(eval_loss):
            perplexity = math.exp(min(eval_loss, 100))  # clamp to avoid overflow
            metrics["eval_perplexity"] = perplexity
            logger.info(
                "Step %d — eval_loss=%.4f  perplexity=%.2f",
                state.global_step,
                eval_loss,
                perplexity,
            )
            if self._writer:
                self._writer.write(
                    step=state.global_step,
                    epoch=state.epoch,
                    eval_loss=eval_loss,
                    perplexity=perplexity,
                )


class EarlyStoppingWithPatience(TrainerCallback):
    """Early stopping based on validation loss plateau.

    Stops training if the validation loss does not improve for
    ``patience`` consecutive evaluation steps.

    Parameters
    ----------
    patience : int
        Number of evaluation steps to wait for improvement.
    min_delta : float
        Minimum absolute improvement to consider as progress.
    metric : str
        Metric to monitor (default: ``eval_loss``).
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        metric: str = "eval_loss",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self._best_value: float | None = None
        self._wait: int = 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if metrics is None:
            return

        current = metrics.get(self.metric)
        if current is None:
            return

        if self._best_value is None or current < self._best_value - self.min_delta:
            self._best_value = current
            self._wait = 0
            logger.info(
                "EarlyStopping: %s improved to %.6f (patience reset).",
                self.metric,
                current,
            )
        else:
            self._wait += 1
            logger.info(
                "EarlyStopping: no improvement for %d/%d steps (%s=%.6f, best=%.6f).",
                self._wait,
                self.patience,
                self.metric,
                current,
                self._best_value,
            )
            if self._wait >= self.patience:
                logger.warning(
                    "EarlyStopping: patience exhausted (%d steps). Stopping training.",
                    self.patience,
                )
                control.should_training_stop = True
