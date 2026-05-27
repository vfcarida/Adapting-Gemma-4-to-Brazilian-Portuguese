"""
src/eval/tasks/hatebr.py
────────────────────────
HateBR — Hate Speech Detection in Brazilian Portuguese.
Binary classification, measured by macro-F1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskConfig:
    task_name: str = "hatebr"
    dataset_id: str = "ruanchaves/hatebr"
    dataset_split: str = "test"
    metric: str = "macro_f1"
    num_fewshot: int = 25
    description: str = "HateBR — Hate Speech Detection in Brazilian Portuguese"

    text_column: str = "instagram_comment"
    label_column: str = "offensive_language"
    label_map: dict[int, str] = field(default_factory=lambda: {
        0: "Não ofensivo",
        1: "Ofensivo",
    })

    output_type: str = "multiple_choice"
    doc_to_text: str = (
        "Texto: {{instagram_comment}}\n"
        "Este texto contém discurso de ódio?"
    )
    doc_to_target: str = "{{offensive_language}}"

    metadata: dict[str, Any] = field(default_factory=lambda: {
        "version": "1.0",
        "source": "https://huggingface.co/datasets/ruanchaves/hatebr",
        "language": "pt-BR",
        "domain": "social_media",
    })


def build_task() -> TaskConfig:
    return TaskConfig()
