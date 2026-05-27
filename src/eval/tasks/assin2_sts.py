"""
src/eval/tasks/assin2_sts.py
────────────────────────────
ASSIN2-STS — Semantic Textual Similarity (Portuguese).
Regression task, measured by Pearson correlation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskConfig:
    task_name: str = "assin2_sts"
    dataset_id: str = "nilc-nlp/assin2"
    dataset_split: str = "test"
    metric: str = "pearson"
    num_fewshot: int = 15
    description: str = "ASSIN2-STS — Semantic Textual Similarity"

    premise_column: str = "premise"
    hypothesis_column: str = "hypothesis"
    score_column: str = "relatedness_score"
    score_range: tuple[float, float] = (1.0, 5.0)

    output_type: str = "generate_until"
    doc_to_text: str = (
        "Sentença 1: {{premise}}\n"
        "Sentença 2: {{hypothesis}}\n"
        "Similaridade semântica (1-5):"
    )
    doc_to_target: str = "{{relatedness_score}}"

    metadata: dict[str, Any] = field(default_factory=lambda: {
        "version": "1.0",
        "source": "https://huggingface.co/datasets/nilc-nlp/assin2",
        "language": "pt-BR",
        "domain": "STS",
    })


def build_task() -> TaskConfig:
    return TaskConfig()
