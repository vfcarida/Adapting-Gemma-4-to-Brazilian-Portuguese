"""Checkpoint management utilities."""

import json
from pathlib import Path
from typing import Any

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Find the latest checkpoint in an output directory."""
    output_dir = Path(output_dir)
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else 0,
    )
    if checkpoints:
        logger.info(f"Found latest checkpoint: {checkpoints[-1]}")
        return checkpoints[-1]
    return None


def save_training_state(output_dir: str | Path, state: dict[str, Any]) -> None:
    """Save training state metadata alongside checkpoint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "training_state.json"
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2, default=str)


def load_training_state(output_dir: str | Path) -> dict[str, Any] | None:
    """Load training state from checkpoint directory."""
    state_path = Path(output_dir) / "training_state.json"
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return None
