"""
src/eval/tasks/oab_bench.py
───────────────────────────
OAB-Bench — Brazilian Bar Exam (Ordem dos Advogados do Brasil).
Multiple-choice legal exam, measured by Approval Rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskConfig:
    task_name: str = "oab_bench"
    dataset_id: str = "eduagarcia/oab_exams"
    dataset_split: str = "test"
    metric: str = "approval_rate"
    num_fewshot: int = 3
    description: str = "OAB-Bench — Brazilian Bar Exam"

    question_column: str = "question"
    choices_column: str = "alternatives"
    answer_column: str = "answer"

    output_type: str = "multiple_choice"
    doc_to_text: str = "{{question}}\n\nAlternativas:\n{{alternatives}}\n\nResposta:"
    doc_to_target: str = "{{answer}}"

    # OAB-specific: compute approval rate as percentage of correct answers
    # The official passing threshold varies by edition, but we report raw rate
    approval_threshold: float = 0.5  # informational only

    metadata: dict[str, Any] = field(default_factory=lambda: {
        "version": "1.0",
        "source": "https://huggingface.co/datasets/eduagarcia/oab_exams",
        "language": "pt-BR",
        "domain": "legal",
    })


def build_task() -> TaskConfig:
    return TaskConfig()
