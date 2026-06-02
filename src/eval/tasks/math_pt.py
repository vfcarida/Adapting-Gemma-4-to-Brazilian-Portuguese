"""Tarefa Math-PT — Problemas matemáticos em português.

Formato: problema de matemática → resposta numérica ou alternativa
Métrica: accuracy (normaliza respostas numéricas)
TODO: Identificar fonte oficial do dataset Math-PT
"""

import re

from src.eval.tasks.base_task import BaseTask


class MathPT(BaseTask):
    """Problemas matemáticos em português brasileiro."""

    task_name = "math_pt"

    def parse_prediction(self, raw_output: str) -> str:
        """Extrai resposta do problema matemático.

        Tenta extrair: letra de alternativa OU valor numérico final.
        """
        text = raw_output.strip()

        # Tentar extrair letra (se múltipla escolha)
        upper = text.upper()
        for char in upper:
            if char in "ABCDE":
                return char

        # Tentar extrair número final
        numbers = re.findall(r"-?\d+(?:[.,]\d+)?", text)
        if numbers:
            # Retorna o último número encontrado (tipicamente a resposta final)
            return numbers[-1].replace(",", ".")

        return text[:20]  # Fallback: primeiros 20 chars

    def get_gold_label(self, example: dict) -> str:
        """Extrai resposta correta."""
        answer = example.get("answer", example.get("label", ""))
        return str(answer).strip()
