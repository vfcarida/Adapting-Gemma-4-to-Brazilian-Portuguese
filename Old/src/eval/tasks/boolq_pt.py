"""Tarefa BoolQ-PT — Perguntas booleanas em português.

Formato: pergunta + passagem → sim/não
Métrica: boolq_accuracy (normaliza sim/não/yes/no)
"""

from src.eval.tasks.base_task import BaseTask


class BoolQPT(BaseTask):
    """BoolQ traduzido para português."""

    task_name = "boolq_pt"

    def parse_prediction(self, raw_output: str) -> str:
        """Normaliza resposta para 'sim' ou 'nao'."""
        text = raw_output.strip().lower()
        # Normalizar variações
        if any(w in text for w in ["sim", "yes", "verdadeiro", "true"]):
            return "sim"
        if any(w in text for w in ["não", "nao", "no", "falso", "false"]):
            return "nao"
        # Fallback: primeira palavra
        first = text.split()[0] if text else ""
        return "sim" if "sim" in first else "nao"

    def get_gold_label(self, example: dict) -> str:
        """Extrai label gold normalizado."""
        label = example.get("answer", example.get("label", ""))
        if isinstance(label, bool):
            return "sim" if label else "nao"
        return "sim" if str(label).lower() in ["sim", "yes", "true", "1"] else "nao"
