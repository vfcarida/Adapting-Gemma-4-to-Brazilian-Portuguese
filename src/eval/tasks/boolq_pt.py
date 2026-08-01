"""Tarefa BoolQ-PT — Perguntas booleanas em português.

Formato: pergunta + passagem → sim/não
Métrica: boolq_accuracy (normaliza sim/não/yes/no)
Fonte: PORTULAN/extraglue, config "boolq_pt-BR" (train 9427 / val 3270 / test 3245).
"""

from typing import Any

from src.eval.tasks.base_task import BaseTask


class BoolQPT(BaseTask):
    """BoolQ traduzido para português."""

    task_name = "boolq_pt"

    def load_data(self, config: dict[str, Any]) -> list[dict]:
        # Se7enB/boolq_pt was fabricated. Replaced with PORTULAN/extraglue,
        # config "boolq_pt-BR". This task previously had no load_data override
        # at all, so raw HF dicts passed through unnormalized.
        hub_id = config.get("hub_id", "PORTULAN/extraglue")
        subset = config.get("subset", "boolq_pt-BR")
        local_path = config.get("local_path")

        if local_path:
            data = self._load_from_local(local_path)
        else:
            # NOTE: like copa_pt-BR, the "test" split has its label masked to
            # -1 for every row (GLUE/SuperGLUE test-label-hiding convention).
            # "validation" is the only split with real, usable gold labels.
            data = self._load_from_hub(hub_id, subset=subset, split="validation")

        # Real schema: passage, question, idx, label (int: 0=No, 1=Yes).
        # _boolq_formatter (prompt_templates.py) reads "passage"/"question"/
        # "answer"; get_gold_label below reads "answer" too, so populate that
        # exact key (as a bool, matching the isinstance(answer, bool) branch
        # already handled by both).
        examples = []
        for item in data:
            label = item.get("label", item.get("answer"))
            if isinstance(label, bool):
                answer = label
            else:
                answer = int(label) == 1
            examples.append(
                {
                    "passage": item.get("passage", item.get("text", "")),
                    "question": item.get("question", ""),
                    "answer": answer,
                }
            )
        return examples

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
