"""BLUEX benchmark task."""

from typing import Any

from src.eval.tasks.base_task import BaseTask


class BluexTask(BaseTask):
    """BLUEX university entrance exam questions."""

    def load_data(self, config: dict[str, Any]) -> list[dict]:
        hub_id = config.get("hub_id", "Se7enB/bluex")
        data = self._load_from_hub(hub_id)

        examples = []
        for item in data:
            example = {
                "question": item.get("question", item.get("pergunta", "")),
                "options": item.get("options", item.get("alternatives", [])),
                "answer": item.get("answer", item.get("correct", "")),
            }
            examples.append(example)
        return examples

    def get_gold_label(self, example: dict) -> str:
        return str(example.get("answer", "")).strip().upper()

    def parse_prediction(self, raw_prediction: str) -> str:
        return self._extract_letter(raw_prediction)
