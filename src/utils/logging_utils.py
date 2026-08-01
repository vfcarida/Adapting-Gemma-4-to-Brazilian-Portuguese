"""Logging utilities with JSON and console output."""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def get_logger(name: str, log_file: str | None = None, level: str = "INFO") -> logging.Logger:
    """Create a logger with console and optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    # Avoid double-printed lines when a root handler is also configured
    # (e.g. by a library or the Colab/Jupyter default logging setup).
    logger.propagate = False

    if not logger.handlers:
        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        logger.addHandler(console)

        # File handler
        if log_file:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(path)
            fh.setFormatter(
                logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
            )
            logger.addHandler(fh)

    return logger


def resolve_report_to(requested: list[str] | None) -> list[str]:
    """Filter a `report_to` list down to backends that are actually usable.

    transformers raises at Trainer-construction time if `report_to` names an
    integration whose package isn't installed (or isn't a real integration
    name at all, e.g. `"json"` — a common config typo in this repo, since
    metrics are already persisted separately via `MetricsLogger`/JSONL).
    This makes `report_to` degrade gracefully instead of crashing training:
    unknown/uninstalled backends are dropped with a warning, so a config can
    freely request `["tensorboard", "wandb"]` and still run on a machine
    that has neither installed.

    Args:
        requested: Backend names as they'd be passed to
            `TrainingArguments(report_to=...)`, or None.

    Returns:
        The subset of `requested` that is safe to pass to `TrainingArguments`
        (or `["none"]` if nothing requested is usable).
    """
    if not requested:
        return ["none"]

    logger = get_logger(__name__)
    usable = []
    for backend in requested:
        name = backend.lower().strip()
        if name in ("none", "all"):
            usable.append(name)
            continue
        if name == "wandb":
            try:
                import wandb  # noqa: F401

                usable.append(name)
            except ImportError:
                logger.warning("report_to=wandb requested but wandb is not installed; dropping.")
        elif name == "tensorboard":
            try:
                import tensorboard  # noqa: F401

                usable.append(name)
            except ImportError:
                logger.warning(
                    "report_to=tensorboard requested but tensorboard is not installed; dropping."
                )
        elif name in (
            "comet_ml",
            "mlflow",
            "clearml",
            "codecarbon",
            "dagshub",
            "neptune",
            "dvclive",
        ):
            # Recognized integrations we don't special-case; let transformers
            # validate them (best-effort passthrough).
            usable.append(name)
        else:
            logger.warning(
                f"report_to={backend!r} is not a valid transformers integration; dropping."
            )

    return usable or ["none"]


class MetricsLogger:
    """Append-only JSON lines logger for metrics."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log a metrics dict as a JSON line."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step,
            **metrics,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def read_all(self) -> list[dict]:
        """Read all logged records."""
        if not self.path.exists():
            return []
        records = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
