"""
src/eval/tasks/xlsum_pt.py
──────────────────────────
XLSum-PT — Portuguese Abstractive Summarization.
Generative task (optional appendage).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskConfig:
    task_name: str = "xlsum_pt"
    dataset_id: str = "csebuetnlp/xlsum"
    dataset_config: str = "portuguese"
    dataset_split: str = "test"
    metric: str = "rougeL"  # Will require ROUGE metric if used heavily, but we can default to rough gen output
    num_fewshot: int = 1
    description: str = "XLSum-PT — Portuguese Abstractive Summarization"

    text_column: str = "text"
    summary_column: str = "summary"

    output_type: str = "generate_until"
    doc_to_text: str = (
        "Resuma o seguinte texto em português:\n\n"
        "{{text}}\n\n"
        "Resumo:"
    )
    doc_to_target: str = "{{summary}}"
    
    generation_kwargs: dict[str, Any] = field(default_factory=lambda: {
        "max_new_tokens": 128,
        "temperature": 0.0,
        "do_sample": False,
    })

    metadata: dict[str, Any] = field(default_factory=lambda: {
        "version": "1.0",
        "source": "https://huggingface.co/datasets/csebuetnlp/xlsum",
        "language": "pt",
        "domain": "summarization",
    })


def build_task() -> TaskConfig:
    return TaskConfig()
