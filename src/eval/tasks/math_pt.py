"""Tarefa Math-PT — Problemas matemáticos em português brasileiro.

Formato: problema de matemática de múltipla escolha (5 alternativas) →
letra da alternativa correta.
Métrica: accuracy.

hub_id real: tiagoteixeira03/MATH-PT (arXiv:2604.25926), config
"ptbr_multiple_choice" (638 problemas de fontes BR como OBMEP/ITA, split
"test" único). Verificado ao vivo em 2026-08-01 — dataset público, sem
gate. O Se7enB/math_pt original era fabricado; nenhum substituto público
existia até esta verificação. O dataset também expõe "ptpt_*" (português
europeu, não usado aqui) e uma variante "*_open_ended" com apenas 128
exemplos e resposta em forma de solução longa (sem valor final isolado) —
menos adequada para scoring automático que a multiple_choice.
"""

import re
from typing import Any

from src.eval.tasks.base_task import BaseTask


class MathPT(BaseTask):
    """Problemas matemáticos em português brasileiro (múltipla escolha)."""

    task_name = "math_pt"

    def load_data(self, config: dict[str, Any]) -> list[dict]:
        hub_id = config.get("hub_id", "tiagoteixeira03/MATH-PT")
        subset = config.get("subset", "ptbr_multiple_choice")
        data = self._apply_max_samples(
            self._load_from_hub(hub_id, subset=subset, split="test"), config
        )

        examples = []
        for item in data:
            choices = item.get("choices", {})
            # Schema real: choices é um dict {"A": "...", ..., "E": "..."},
            # não a forma {"text": [...], "label": [...]} do ARC/BLUEX/OAB —
            # _choices_to_options não serve aqui, ordena direto pelas chaves.
            options = [choices[k] for k in sorted(choices.keys())]
            examples.append(
                {
                    "question": item.get("problem", ""),
                    "options": options,
                    "answer": item.get("answer", ""),
                }
            )
        return examples

    def parse_prediction(self, raw_output: str) -> str:
        """Extrai a letra da alternativa escolhida.

        Múltipla escolha (gold é sempre uma letra A-E): tenta a letra
        primeiro via BaseTask._extract_letter (parser robusto, usado por
        enem/bluex/oab_bench). Cai para extração numérica só como último
        recurso — não resolve o caso do modelo responder com o valor da
        alternativa em vez da letra (ex.: "16 cm" em vez de "D"); isso
        contaria como erro de scoring, não como acerto.
        """
        text = raw_output.strip()

        letter = self._extract_letter(text)
        if letter:
            return letter

        numbers = re.findall(r"-?\d+(?:[.,]\d+)?", text)
        if numbers:
            return numbers[-1].replace(",", ".")

        return text[:20]  # Fallback: primeiros 20 chars

    def get_gold_label(self, example: dict) -> str:
        """Extrai resposta correta (letra)."""
        answer = example.get("answer", example.get("label", ""))
        return str(answer).strip().upper()
