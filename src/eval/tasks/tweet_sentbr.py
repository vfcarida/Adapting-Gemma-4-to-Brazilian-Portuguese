"""
src/eval/tasks/tweet_sentbr.py
──────────────────────────────
TweetSentBR — Tweet Sentiment Analysis in Brazilian Portuguese.
Multi-class sentiment classification, measured by macro-F1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskConfig:
    task_name: str = "tweet_sentbr"
    dataset_id: str = "ruanchaves/tweetsentbr"
    dataset_split: str = "test"
    metric: str = "macro_f1"
    num_fewshot: int = 25
    description: str = "TweetSentBR — Tweet Sentiment Analysis"

    text_column: str = "tweet_text"
    label_column: str = "label"
    label_map: dict[int, str] = field(default_factory=lambda: {
        0: "Negativo",
        1: "Neutro",
        2: "Positivo",
    })

    output_type: str = "multiple_choice"
    doc_to_text: str = (
        "Tweet: {{tweet_text}}\n"
        "Sentimento (Negativo/Neutro/Positivo):"
    )
    doc_to_target: str = "{{label}}"

    metadata: dict[str, Any] = field(default_factory=lambda: {
        "version": "1.0",
        "source": "https://huggingface.co/datasets/ruanchaves/tweetsentbr",
        "language": "pt-BR",
        "domain": "social_media",
    })


def build_task() -> TaskConfig:
    return TaskConfig()
