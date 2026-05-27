"""
src/eval/tasks/assin2_rte.py
────────────────────────────
ASSIN2-RTE — Recognizing Textual Entailment (Portuguese).
Classification task, measured by macro-F1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskConfig:
    task_name: str = "assin2_rte"
    dataset_id: str = "nilc-nlp/assin2"
    dataset_split: str = "test"
    metric: str = "macro_f1"
    num_fewshot: int = 15
    description: str = "ASSIN2-RTE — Recognizing Textual Entailment"

    premise_column: str = "premise"
    hypothesis_column: str = "hypothesis"
    label_column: str = "entailment_judgment"
    label_map: dict[int, str] = field(default_factory=lambda: {
        0: "Nenhuma",       # None
        1: "Implicação",    # Entailment
        2: "Paráfrase",     # Paraphrase
    })

    output_type: str = "multiple_choice"
    doc_to_text: str = (
        "Premissa: {{premise}}\n"
        "Hipótese: {{hypothesis}}\n"
        "A relação entre a premissa e a hipótese é:"
    )
    doc_to_target: str = "{{entailment_judgment}}"

    metadata: dict[str, Any] = field(default_factory=lambda: {
        "version": "1.0",
        "source": "https://huggingface.co/datasets/nilc-nlp/assin2",
        "language": "pt-BR",
        "domain": "NLI",
    })


def build_task() -> TaskConfig:
    return TaskConfig()
