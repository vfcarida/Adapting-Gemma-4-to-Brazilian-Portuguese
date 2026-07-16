"""Tarefa PublicHearingBR — Classificação de audiências públicas.

Formato: texto de audiência pública → classificação temática
Métrica: macro_f1
TODO: Identificar fonte oficial do dataset PublicHearingBR
"""

from src.eval.tasks.base_task import BaseTask


class PublicHearingBR(BaseTask):
    """Classificação de transcrições de audiências públicas brasileiras."""

    task_name = "publichearing_br"

    def parse_prediction(self, raw_output: str) -> str:
        """Extrai classificação temática."""
        return raw_output.strip().lower()

    def get_gold_label(self, example: dict) -> str:
        """Extrai label gold."""
        return str(example.get("label", example.get("category", ""))).strip().lower()
