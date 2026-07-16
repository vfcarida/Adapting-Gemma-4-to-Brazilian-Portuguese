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
