"""BRoverbs task."""

from typing import Any

from src.eval.tasks.base_task import BaseTask


class BRoverbsTask(BaseTask):
    """Brazilian proverbs completion task."""

    def load_data(self, config: dict[str, Any]) -> list[dict]:
        local_path = config.get("local_path")
        hub_id = config.get("hub_id")

        if local_path:
            data = self._load_from_local(local_path)
        elif hub_id:
            data = self._load_from_hub(hub_id)
        else:
            return []

        examples = []
        for item in data:
            example = {
                "question": item.get("question", item.get("proverb_start", "")),
                "options": item.get("options", item.get("choices", [])),
                "answer": item.get("answer", item.get("correct", "")),
            }
            examples.append(example)
        return examples

    def get_gold_label(self, example: dict) -> str:
        return str(example.get("answer", "")).strip().upper()

    def parse_prediction(self, raw_prediction: str) -> str:
        return self._extract_letter(raw_prediction)
