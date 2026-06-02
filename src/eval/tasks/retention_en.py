"""Tarefas de retenção em inglês — MMLU, HellaSwag, ARC.

Estas tarefas medem catastrophic forgetting: quanto de capacidade
em inglês o modelo perdeu após CPT em português.

Cada tarefa usa uma amostra (500 exemplos) do benchmark original
para avaliação rápida mas representativa.
"""

from src.eval.tasks.base_task import BaseTask


class MMLUEn(BaseTask):
    """MMLU (Massive Multitask Language Understanding) — amostra EN.

    Questões de múltipla escolha cobrindo 57 áreas do conhecimento.
    """

    task_name = "mmlu_en"

    def parse_prediction(self, raw_output: str) -> str:
        text = raw_output.strip().upper()
        for char in text:
            if char in "ABCD":
                return char
        return text[:1] if text else ""

    def get_gold_label(self, example: dict) -> str:
        answer = example.get("answer", example.get("label", ""))
        # MMLU pode usar índice numérico (0-3) ou letra
        if isinstance(answer, int):
            return chr(65 + answer)  # 0→A, 1→B, etc.
        return str(answer).strip().upper()


class HellaSwagEn(BaseTask):
    """HellaSwag — completar sentenças com senso comum (EN)."""

    task_name = "hellaswag_en"

    def parse_prediction(self, raw_output: str) -> str:
        text = raw_output.strip()
        # Aceitar número (0-3) ou letra (A-D)
        for char in text.upper():
            if char in "ABCD":
                return char
        for char in text:
            if char in "0123":
                return chr(65 + int(char))  # 0→A
        return text[:1].upper() if text else ""

    def get_gold_label(self, example: dict) -> str:
        label = example.get("label", example.get("answer", ""))
        if isinstance(label, int):
            return chr(65 + label)
        return str(label).strip().upper()


class ARCEn(BaseTask):
    """ARC-Challenge — raciocínio científico (EN)."""

    task_name = "arc_en"

    def parse_prediction(self, raw_output: str) -> str:
        text = raw_output.strip().upper()
        for char in text:
            if char in "ABCDE":
                return char
        return text[:1] if text else ""

    def get_gold_label(self, example: dict) -> str:
        return str(example.get("answerKey", example.get("answer", ""))).strip().upper()
