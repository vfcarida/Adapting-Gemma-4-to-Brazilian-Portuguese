"""HateBR task."""

from typing import Any

from src.eval.tasks.base_task import BaseTask


class HateBRTask(BaseTask):
    """HateBR hate speech detection."""

    def load_data(self, config: dict[str, Any]) -> list[dict]:
        hub_id = config.get("hub_id", "Se7enB/HateBR")
        data = self._load_from_hub(hub_id)

        examples = []
        for item in data:
            label_val = item.get("label", item.get("offensive_language", 0))
            label = "odio" if label_val == 1 else "nao_odio"
            example = {
                "text": item.get("text", item.get("instagram_comment", "")),
                "label": label,
            }
            examples.append(example)
        return examples

    def get_gold_label(self, example: dict) -> str:
        return example["label"]

    def parse_prediction(self, raw_prediction: str) -> str:
        text = raw_prediction.strip().lower()
        if "nao" in text or "nao_odio" in text or "no" in text:
            return "nao_odio"
        if "odio" in text or "hate" in text or "sim" in text:
            return "odio"
        return "nao_odio"
