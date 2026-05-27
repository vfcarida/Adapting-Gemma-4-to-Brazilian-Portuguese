"""
src/eval/tasks/rte_pt.py
────────────────────────
RTE-PT — Recognizing Textual Entailment (Portuguese translation).
Binary classification, measured by Accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskConfig:
    task_name: str = "rte_pt"
    dataset_id: str = "Se7eN/rte_pt"
    dataset_split: str = "test"
    metric: str = "accuracy"
    num_fewshot: int = 15
    description: str = "RTE-PT — Recognizing Textual Entailment (Portuguese)"

    premise_column: str = "premise"
    hypothesis_column: str = "hypothesis"
    label_column: str = "label"
    label_map: dict[int, str] = field(default_factory=lambda: {
        0: "Implicação",    # Entailment
        1: "Não implicação", # Not entailment
    })

    output_type: str = "multiple_choice"
    doc_to_text: str = (
        "Premissa: {{premise}}\n"
        "Hipótese: {{hypothesis}}\n"
        "A premissa implica a hipótese?"
    )
    doc_to_target: str = "{{label}}"

    metadata: dict[str, Any] = field(default_factory=lambda: {
        "version": "1.0",
        "source": "https://huggingface.co/datasets/Se7eN/rte_pt",
        "language": "pt-BR",
        "domain": "NLI",
    })


def build_task() -> TaskConfig:
    return TaskConfig()
