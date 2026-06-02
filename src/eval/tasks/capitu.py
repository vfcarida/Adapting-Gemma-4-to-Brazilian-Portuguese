"""Tarefa CAPITU — Benchmark de compreensão cultural brasileira.

Formato: questões sobre literatura, cultura e história brasileira
Métrica: accuracy
TODO: Identificar fonte oficial do dataset CAPITU
"""

from src.eval.tasks.base_task import BaseTask


class Capitu(BaseTask):
    """Benchmark CAPITU de compreensão cultural brasileira."""

    task_name = "capitu"

    def parse_prediction(self, raw_output: str) -> str:
        """Extrai letra da alternativa."""
        text = raw_output.strip().upper()
        for char in text:
            if char in "ABCDE":
                return char
        return text[:1] if text else ""

    def get_gold_label(self, example: dict) -> str:
        """Extrai resposta correta."""
        return str(example.get("answer", example.get("label", ""))).strip().upper()
