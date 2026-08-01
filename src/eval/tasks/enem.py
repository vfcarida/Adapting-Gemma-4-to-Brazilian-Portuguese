"""ENEM benchmark task."""

from typing import Any

from src.eval.tasks.base_task import BaseTask


class EnemTask(BaseTask):
    """ENEM multiple choice questions (2022-2024)."""

    def load_data(self, config: dict[str, Any]) -> list[dict]:
        hub_id = config.get("hub_id", "maritaca-ai/enem")
        subset = config.get("subset")
        year = config.get("year")

        # maritaca-ai/enem only has a "train" split per config (2022/2023/2024);
        # BaseTask._load_from_hub already falls back test -> validation -> train.
        data = self._load_from_hub(hub_id, subset=subset)

        # Normalize format.
        # Real schema (maritaca-ai/enem): question, alternatives (list[str]), label (letter).
        # "pergunta"/"alternativas"/"resposta"/"options"/"answer" are kept only
        # as a secondary fallback in case a differently-shaped source is used.
        examples = []
        for item in data:
            example = {
                "question": item.get("question") or item.get("pergunta", ""),
                "options": item.get("alternatives")
                or item.get("options")
                or item.get("alternativas", []),
                "answer": item.get("label") or item.get("answer") or item.get("resposta", ""),
            }
            if year:
                example["year"] = year
            examples.append(example)
        return examples

    def get_gold_label(self, example: dict) -> str:
        return str(example.get("answer", "")).strip().upper()

    def parse_prediction(self, raw_prediction: str) -> str:
        return self._extract_letter(raw_prediction)
