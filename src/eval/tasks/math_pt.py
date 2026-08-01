"""Tarefa Math-PT — Problemas matemáticos em português.

Formato: problema de matemática → resposta numérica ou alternativa
Métrica: accuracy (normaliza respostas numéricas)

Nenhum benchmark verificado de matemática em português brasileiro foi
encontrado no HF Hub (Se7enB/math_pt era fabricado). Mantido desabilitado
por padrão em configs/eval/benchmarks.yaml até que uma fonte real seja
identificada.
"""

import re

from src.eval.tasks.base_task import BaseTask


class MathPT(BaseTask):
    """Problemas matemáticos em português brasileiro."""

    task_name = "math_pt"

    def parse_prediction(self, raw_output: str) -> str:
        """Extrai resposta do problema matemático.

        Tenta extrair: valor numérico final OU letra de alternativa.

        Prioriza número sobre letra (a maioria das respostas matemáticas é
        numérica). Anteriormente o código escaneava por qualquer caractere
        A-E ANTES de tentar números, o que dava falso positivo em qualquer
        texto contendo essas letras (ex.: "RESULTADO: 42" retornava "E" —
        a primeira letra A-E encontrada em "RESULTADO" — em vez de "42").
        Usa o parser robusto de letras da BaseTask (`_extract_letter`) como
        fallback apenas quando não há número na resposta.
        """
        text = raw_output.strip()

        numbers = re.findall(r"-?\d+(?:[.,]\d+)?", text)
        if numbers:
            # Retorna o último número encontrado (tipicamente a resposta final)
            return numbers[-1].replace(",", ".")

        letter = self._extract_letter(text)
        if letter:
            return letter

        return text[:20]  # Fallback: primeiros 20 chars

    def get_gold_label(self, example: dict) -> str:
        """Extrai resposta correta."""
        answer = example.get("answer", example.get("label", ""))
        return str(answer).strip()
