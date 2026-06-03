"""Base class for evaluation tasks."""

import re
from abc import ABC, abstractmethod
from typing import Any

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class BaseTask(ABC):
    """Abstract base class for evaluation tasks."""

    @abstractmethod
    def load_data(self, config: dict[str, Any]) -> list[dict]:
        """Load task data from hub or local path."""
        ...

    @abstractmethod
    def get_gold_label(self, example: dict) -> Any:
        """Extract gold label from an example."""
        ...

    def parse_prediction(self, raw_prediction: str) -> str:
        """Parse model output into a standardized prediction."""
        # Default: extract first letter/word
        text = raw_prediction.strip()
        return text

    def _extract_letter(self, text: str) -> str:
        """Extract a single letter answer (A-E) from model output.

        Hierarquia de parsing:
        1. Letra isolada
        2. Padrão "A)" ou "A."
        3. Padrão "Resposta: X" / "alternativa X"
        4. Letra em parênteses "(X)"
        5. Primeira letra isolada A-E com word boundary
        6. Fallback: primeiro caractere
        """
        text = text.strip()
        if not text:
            return ""

        # 1. Exact single letter
        if len(text) == 1 and text.upper() in "ABCDE":
            return text.upper()

        # 2. "A)" or "A." at start (but NOT "A " followed by word)
        match = re.match(r"^([A-Ea-e])[)\.]", text)
        if match:
            return match.group(1).upper()

        # 3. "Resposta: X" / "Answer: X" / "alternativa X" / "é X"
        match = re.search(
            r"(?:resposta|answer|alternativa|é)[^A-Ea-e]*([A-Ea-e])\b",
            text, re.IGNORECASE
        )
        if match:
            return match.group(1).upper()

        # 4. Letter in parentheses "(X)"
        match = re.search(r"\(([A-Ea-e])\)", text)
        if match:
            return match.group(1).upper()

        # 5. Standalone letter at start (letter + end or letter + non-alpha)
        match = re.match(r"^([A-Ea-e])(?:\s*$|[^a-zA-Z])", text)
        if match:
            return match.group(1).upper()

        # 6. Last resort: find any isolated letter A-E
        match = re.search(r"(?<![a-zA-Z])([A-Ea-e])(?![a-zA-Z])", text)
        if match:
            return match.group(1).upper()

        # 7. Final fallback
        if text[0].upper() in "ABCDE":
            return text[0].upper()
        return text[:1].upper()

    def _extract_number(self, text: str) -> str:
        """Extract a number from text."""
        match = re.search(r"(\d+\.?\d*)", text.strip())
        return match.group(1) if match else ""

    def _load_from_hub(self, hub_id: str, subset: str | None = None, split: str = "test") -> list[dict]:
        """Load data from HuggingFace Hub."""
        from datasets import load_dataset
        kwargs = {"split": split}
        if subset:
            kwargs["name"] = subset
        try:
            ds = load_dataset(hub_id, **kwargs)
            return [dict(ex) for ex in ds]
        except Exception as e:
            logger.warning(f"Failed to load split '{split}' from {hub_id}: {e}")
            # Try other splits
            for fallback_split in ["validation", "train"]:
                try:
                    kwargs["split"] = fallback_split
                    ds = load_dataset(hub_id, **kwargs)
                    logger.info(f"Loaded fallback split '{fallback_split}' for {hub_id}")
                    return [dict(ex) for ex in ds]
                except Exception as fallback_e:
                    logger.debug(f"Fallback split '{fallback_split}' failed: {fallback_e}")
                    continue
        logger.error(f"Could not load any split from {hub_id}")
        return []

    def _load_from_local(self, path: str) -> list[dict]:
        """Load data from local JSONL file."""
        import json
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return []
        data = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
