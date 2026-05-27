"""
src/eval/tasks/enem.py
──────────────────────
ENEM — Brazilian High School National Exam.
Multiple-choice questions, measured by Approval Rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskConfig:
    """ENEM task configuration for lm-evaluation-harness."""

    task_name: str = "enem"
    dataset_id: str = "eduagarcia/enem_challenge"
    dataset_split: str = "test"
    metric: str = "approval_rate"
    num_fewshot: int = 3
    description: str = "ENEM — Brazilian High School National Exam (Exame Nacional do Ensino Médio)"

    # Column mapping
    question_column: str = "question"
    choices_column: str = "alternatives"
    answer_column: str = "answer"

    # Harness-specific
    output_type: str = "multiple_choice"
    doc_to_text: str = "{{question}}\n\nAlternativas:\n{{alternatives}}\n\nResposta:"
    doc_to_target: str = "{{answer}}"
    target_delimiter: str = " "
    fewshot_delimiter: str = "\n\n"

    metadata: dict[str, Any] = field(default_factory=lambda: {
        "version": "1.0",
        "source": "https://huggingface.co/datasets/eduagarcia/enem_challenge",
        "language": "pt-BR",
        "domain": "education",
        "license": "CC-BY-4.0",
    })


def build_task() -> TaskConfig:
    """Factory function returning the ENEM task configuration."""
    return TaskConfig()
