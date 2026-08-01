"""Tarefas de retenção em inglês — MMLU, HellaSwag, ARC.

Estas tarefas medem catastrophic forgetting: quanto de capacidade
em inglês o modelo perdeu após CPT em português.

Cada tarefa usa uma amostra (500 exemplos) do benchmark original
para avaliação rápida mas representativa.

Todas as três usam datasets reais (cais/mmlu subset "all", Rowan/hellaswag,
allenai/ai2_arc subset "ARC-Challenge"), mas anteriormente não tinham
`load_data` próprio: os campos brutos do HF passavam sem normalização, e o
`_default_formatter` genérico (prompt_templates.py) procura uma chave
"options" que nenhum dos três datasets possui nativamente. `load_data` foi
adicionado a cada classe para produzir esse formato.
"""

import re
from typing import Any

from src.eval.tasks.base_task import BaseTask


class MMLUEn(BaseTask):
    """MMLU (Massive Multitask Language Understanding) — amostra EN.

    Questões de múltipla escolha cobrindo 57 áreas do conhecimento.
    """

    task_name = "mmlu_en"

    def load_data(self, config: dict[str, Any]) -> list[dict]:
        hub_id = config.get("hub_id", "cais/mmlu")
        subset = config.get("subset", "all")
        data = self._apply_max_samples(
            self._load_from_hub(hub_id, subset=subset, split="test"), config
        )

        # Real schema: question, subject, choices (list[str]), answer (int index).
        examples = []
        for item in data:
            examples.append(
                {
                    "question": item.get("question", ""),
                    "options": item.get("choices", []),
                    "answer": item.get("answer", 0),  # int; converted to letter in get_gold_label
                }
            )
        return examples

    def parse_prediction(self, raw_output: str) -> str:
        # Substitui o scan ingênuo por caractere (falso positivo em texto
        # livre) pelo parser robusto e já testado da BaseTask.
        return self._extract_letter(raw_output)

    def get_gold_label(self, example: dict) -> str:
        answer = example.get("answer", example.get("label", ""))
        # MMLU pode usar índice numérico (0-3) ou letra
        if isinstance(answer, int):
            return chr(65 + answer)  # 0→A, 1→B, etc.
        return str(answer).strip().upper()


class HellaSwagEn(BaseTask):
    """HellaSwag — completar sentenças com senso comum (EN)."""

    task_name = "hellaswag_en"

    def load_data(self, config: dict[str, Any]) -> list[dict]:
        hub_id = config.get("hub_id", "Rowan/hellaswag")
        # IMPORTANT: split="test" is UNLABELED in Rowan/hellaswag (empty
        # labels for every row) — use "validation", which has real labels.
        data = self._apply_max_samples(self._load_from_hub(hub_id, split="validation"), config)

        # Real schema: ind, activity_label, ctx_a, ctx_b, ctx, endings
        # (list[str]), label (STRING "0".."3", not an int).
        examples = []
        for item in data:
            ctx = item.get("ctx") or f"{item.get('ctx_a', '')} {item.get('ctx_b', '')}".strip()
            examples.append(
                {
                    "question": ctx,
                    "options": item.get("endings", []),
                    "answer": item.get("label", ""),
                }
            )
        return examples

    def parse_prediction(self, raw_output: str) -> str:
        text = raw_output.strip()
        # Parser robusto de letra (evita falso positivo em texto livre).
        letter = self._extract_letter(text)
        if letter and letter in "ABCD":
            return letter
        # Fallback: dígito isolado 0-3 (endings são indexados numericamente
        # no dataset original). Usa isolamento (não substring) para evitar
        # capturar dígitos de datas/números soltos no texto.
        match = re.search(r"(?<!\d)([0123])(?!\d)", text)
        if match:
            return chr(65 + int(match.group(1)))
        return letter

    def get_gold_label(self, example: dict) -> str:
        label = example.get("label", example.get("answer", ""))
        # label real é STRING ("0".."3"), não int — converte robustamente
        # em vez de assumir isinstance(label, int) (que nunca era True).
        try:
            return chr(65 + int(label))
        except (TypeError, ValueError):
            return str(label).strip().upper()


class ARCEn(BaseTask):
    """ARC-Challenge — raciocínio científico (EN)."""

    task_name = "arc_en"

    def load_data(self, config: dict[str, Any]) -> list[dict]:
        hub_id = config.get("hub_id", "allenai/ai2_arc")
        subset = config.get("subset", "ARC-Challenge")
        data = self._apply_max_samples(
            self._load_from_hub(hub_id, subset=subset, split="test"), config
        )

        # Real schema: id, question, choices {"text": [...], "label": [...]},
        # answerKey. "choices" is a dict, not a flat list.
        examples = []
        for item in data:
            examples.append(
                {
                    "question": item.get("question", ""),
                    "options": self._choices_to_options(item.get("choices", [])),
                    "answerKey": item.get("answerKey", ""),
                }
            )
        return examples

    def parse_prediction(self, raw_output: str) -> str:
        return self._extract_letter(raw_output)

    def get_gold_label(self, example: dict) -> str:
        return str(example.get("answerKey", example.get("answer", ""))).strip().upper()
