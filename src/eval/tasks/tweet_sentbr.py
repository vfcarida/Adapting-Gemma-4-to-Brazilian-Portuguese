"""TweetSentBR task."""

from typing import Any

from src.eval.tasks.base_task import BaseTask


class TweetSentBRTask(BaseTask):
    """TweetSentBR sentiment analysis."""

    def load_data(self, config: dict[str, Any]) -> list[dict]:
        hub_id = config.get("hub_id", "Se7enB/TweetSentBR")
        data = self._load_from_hub(hub_id)

        label_map = {0: "negativo", 1: "neutro", 2: "positivo"}
        examples = []
        for item in data:
            label_val = item.get("label", item.get("sentiment", 1))
            example = {
                "text": item.get("text", item.get("tweet_text", "")),
                "label": label_map.get(label_val, str(label_val)),
            }
            examples.append(example)
        return examples

    def get_gold_label(self, example: dict) -> str:
        return example["label"]

    def parse_prediction(self, raw_prediction: str) -> str:
        text = raw_prediction.strip().lower()
        if "positiv" in text:
            return "positivo"
        if "negativ" in text:
            return "negativo"
        if "neutr" in text:
            return "neutro"
        return "neutro"
