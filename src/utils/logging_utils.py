"""
src/utils/logging_utils.py
──────────────────────────
Structured logging setup with console, file, and JSONL output.
Integrates with Weights & Biases when available.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ──────────────────────────────────────────────────────────────────────
# Logger factory
# ──────────────────────────────────────────────────────────────────────

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a named logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def setup_logging(
    log_dir: str | Path | None = None,
    level: int = logging.INFO,
    log_filename: str = "run.log",
) -> logging.Logger:
    """Configure root logger with console + optional file handler.

    Parameters
    ----------
    log_dir : str | Path | None
        Directory for the log file.  ``None`` disables file logging.
    level : int
        Logging level.
    log_filename : str
        Name of the log file inside *log_dir*.

    Returns
    -------
    logging.Logger
        The configured root logger.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(console)

    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / log_filename, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(fh)

    return root


# ──────────────────────────────────────────────────────────────────────
# JSONL structured metric writer
# ──────────────────────────────────────────────────────────────────────


class JSONLWriter:
    """Append-only JSONL writer for training metrics.

    Each line is a JSON object with a ``timestamp`` field and any
    additional key-value pairs passed to :meth:`write`.

    Usage::

        writer = JSONLWriter("reports/training_logs/cpt_pilot")
        writer.write(step=100, loss=2.31, lr=1e-4)
    """

    def __init__(self, log_dir: str | Path, filename: str = "metrics.jsonl") -> None:
        self._path = Path(log_dir) / filename
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **kwargs: Any) -> None:
        """Write a single JSONL record with automatic timestamp."""
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), **kwargs}
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @property
    def path(self) -> Path:
        return self._path


# ──────────────────────────────────────────────────────────────────────
# W&B helpers
# ──────────────────────────────────────────────────────────────────────


def init_wandb(
    project: str,
    run_name: str,
    config: dict[str, Any] | None = None,
    entity: str | None = None,
    mode: str = "online",
) -> Any:
    """Initialise a Weights & Biases run.

    Returns the ``wandb.Run`` object, or ``None`` if W&B is disabled or
    unavailable.

    Parameters
    ----------
    project : str
        W&B project name.
    run_name : str
        Display name for the run.
    config : dict
        Hyperparameters to log.
    entity : str | None
        W&B team / user entity.
    mode : str
        ``"online"`` | ``"offline"`` | ``"disabled"``.
    """
    try:
        import wandb  # noqa: F811
    except ImportError:
        get_logger(__name__).warning("wandb not installed — skipping W&B logging.")
        return None

    return wandb.init(
        project=project,
        name=run_name,
        config=config,
        entity=entity,
        mode=mode,
        reinit=True,
    )
