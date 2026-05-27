"""
src/eval/tasks/copa_pt.py
─────────────────────────
COPA-PT — Choice of Plausible Alternatives (Portuguese).
Causal reasoning, measured by Accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskConfig:
    task_name: str = "copa_pt"
    dataset_id: str = "community-datasets/xcopa"
    dataset_config: str = "pt"
    dataset_split: str = "test"
    metric: str = "accuracy"
    num_fewshot: int = 0
    description: str = "COPA-PT — Causal Reasoning (Choice of Plausible Alternatives)"

    premise_column: str = "premise"
    choice1_column: str = "choice1"
    choice2_column: str = "choice2"
    label_column: str = "label"
    question_column: str = "question"  # "cause" or "effect"

    output_type: str = "multiple_choice"
    doc_to_text: str = (
        "Premissa: {{premise}}\n"
        "Pergunta: Qual é a {{question}}?\n"
        "A) {{choice1}}\n"
        "B) {{choice2}}\n"
        "Resposta:"
    )
    doc_to_target: str = "{{label}}"

    metadata: dict[str, Any] = field(default_factory=lambda: {
        "version": "1.0",
        "source": "https://huggingface.co/datasets/community-datasets/xcopa",
        "language": "pt",
        "domain": "reasoning",
    })


def build_task() -> TaskConfig:
    return TaskConfig()
