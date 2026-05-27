"""
src/eval/tasks/tuguesice_pt.py
──────────────────────────────
TugueSICE-PT — Portuguese Language Understanding.
Multiple-choice, measured by Accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskConfig:
    task_name: str = "tuguesice_pt"
    dataset_id: str = "Se7eN/tuguesice_pt"
    dataset_split: str = "test"
    metric: str = "accuracy"
    num_fewshot: int = 5
    description: str = "TugueSICE-PT — Portuguese Language Understanding"

    question_column: str = "question"
    choices_column: str = "options"
    answer_column: str = "answer"

    output_type: str = "multiple_choice"
    doc_to_text: str = (
        "{{question}}\n\n"
        "Opções:\n{{options}}\n\n"
        "Resposta:"
    )
    doc_to_target: str = "{{answer}}"

    metadata: dict[str, Any] = field(default_factory=lambda: {
        "version": "1.0",
        "source": "https://huggingface.co/datasets/Se7eN/tuguesice_pt",
        "language": "pt",
        "domain": "language_understanding",
    })


def build_task() -> TaskConfig:
    return TaskConfig()
