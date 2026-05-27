"""
src/eval/tasks/bluex.py
───────────────────────
BluEx — University Entrance Exam (Vestibular).
Multiple-choice, measured by Approval Rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskConfig:
    task_name: str = "bluex"
    dataset_id: str = "eduagarcia-temp/BLUEX"
    dataset_split: str = "test"
    metric: str = "approval_rate"
    num_fewshot: int = 3
    description: str = "BluEx — University Entrance Exam (Vestibular)"

    question_column: str = "question"
    choices_column: str = "alternatives"
    answer_column: str = "answer"

    output_type: str = "multiple_choice"
    doc_to_text: str = "{{question}}\n\nAlternativas:\n{{alternatives}}\n\nResposta:"
    doc_to_target: str = "{{answer}}"

    metadata: dict[str, Any] = field(default_factory=lambda: {
        "version": "1.0",
        "source": "https://huggingface.co/datasets/eduagarcia-temp/BLUEX",
        "language": "pt-BR",
        "domain": "education",
    })


def build_task() -> TaskConfig:
    return TaskConfig()
