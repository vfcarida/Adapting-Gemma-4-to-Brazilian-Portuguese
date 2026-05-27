"""
src/eval/tasks/mrpc_pt.py
─────────────────────────
MRPC-PT — Paraphrase Detection (Portuguese translation).
Binary classification, measured by macro-F1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskConfig:
    task_name: str = "mrpc_pt"
    dataset_id: str = "Se7eN/mrpc_pt"
    dataset_split: str = "test"
    metric: str = "macro_f1"
    num_fewshot: int = 5
    description: str = "MRPC-PT — Paraphrase Detection (Portuguese)"

    sentence1_column: str = "sentence1"
    sentence2_column: str = "sentence2"
    label_column: str = "label"
    label_map: dict[int, str] = field(default_factory=lambda: {
        0: "Não",
        1: "Sim",
    })

    output_type: str = "multiple_choice"
    doc_to_text: str = (
        "Sentença 1: {{sentence1}}\n"
        "Sentença 2: {{sentence2}}\n"
        "As sentenças são paráfrases uma da outra?"
    )
    doc_to_target: str = "{{label}}"

    metadata: dict[str, Any] = field(default_factory=lambda: {
        "version": "1.0",
        "source": "https://huggingface.co/datasets/Se7eN/mrpc_pt",
        "language": "pt-BR",
        "domain": "paraphrase",
    })


def build_task() -> TaskConfig:
    return TaskConfig()
