"""Tarefa LegalBench-BR — Benchmark jurídico brasileiro.

Formato: questões de múltipla escolha sobre direito brasileiro
Métrica: accuracy
TODO: Identificar fonte oficial do dataset e hub_id
"""

from src.eval.tasks.base_task import BaseTask


class LegalBenchBR(BaseTask):
    """Benchmark de competência jurídica brasileira."""

    task_name = "legalbench_br"

    def parse_prediction(self, raw_output: str) -> str:
        """Extrai letra da alternativa escolhida."""
        text = raw_output.strip().upper()
        for char in text:
            if char in "ABCDE":
                return char
        return text[:1] if text else ""

    def get_gold_label(self, example: dict) -> str:
        """Extrai resposta correta."""
        return str(example.get("answer", example.get("label", ""))).strip().upper()
