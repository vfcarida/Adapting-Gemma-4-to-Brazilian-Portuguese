"""XLSum-PT task."""

from typing import Any

from src.eval.tasks.base_task import BaseTask


class XLSumPTTask(BaseTask):
    """XLSum Portuguese summarization task."""

    def load_data(self, config: dict[str, Any]) -> list[dict]:
        hub_id = config.get("hub_id", "csebuetnlp/xlsum")
        subset = config.get("subset", "portuguese")

        data = self._load_from_hub(hub_id, subset=subset)

        examples = []
        for item in data[:500]:  # Limit for evaluation
            example = {
                "text": item.get("text", item.get("document", "")),
                "summary": item.get("summary", item.get("target", "")),
            }
            if example["text"]:
                examples.append(example)
        return examples

    def get_gold_label(self, example: dict) -> str:
        return example.get("summary", "")

    def parse_prediction(self, raw_prediction: str) -> str:
        return raw_prediction.strip()
