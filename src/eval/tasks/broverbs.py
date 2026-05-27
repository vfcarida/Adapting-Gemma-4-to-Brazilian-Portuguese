"""
src/eval/tasks/broverbs.py
──────────────────────────
BRoverbs — Brazilian Portuguese Proverb Completion.
Multiple-choice, measured by Accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskConfig:
    task_name: str = "broverbs"
    dataset_id: str = "Se7eN/broverbs"
    dataset_split: str = "test"
    metric: str = "accuracy"
    num_fewshot: int = 5
    description: str = "BRoverbs — Brazilian Portuguese Proverb Completion"

    text_column: str = "context"
    choices_column: str = "options"
    answer_column: str = "answer"

    output_type: str = "multiple_choice"
    doc_to_text: str = "Complete o provérbio:\n{{context}}\n\nOpções:\n{{options}}\n\nResposta:"
    doc_to_target: str = "{{answer}}"

    metadata: dict[str, Any] = field(default_factory=lambda: {
        "version": "1.0",
        "source": "https://huggingface.co/datasets/Se7eN/broverbs",
        "language": "pt-BR",
        "domain": "language_understanding",
    })


def build_task() -> TaskConfig:
    return TaskConfig()
