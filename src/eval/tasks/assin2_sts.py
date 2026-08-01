"""ASSIN2-STS task."""

from typing import Any

from src.eval.tasks.base_task import BaseTask


class Assin2STSTask(BaseTask):
    """ASSIN2 Semantic Textual Similarity."""

    def load_data(self, config: dict[str, Any]) -> list[dict]:
        # "assin2" now resolves via a Hub redirect; use the canonical
        # nilc-nlp/assin2 id directly to avoid the redirect/script-loading issues.
        hub_id = config.get("hub_id", "nilc-nlp/assin2")
        data = self._load_from_hub(hub_id, split="test")

        examples = []
        for item in data:
            example = {
                "sentence1": item.get("premise", item.get("sentence1", "")),
                "sentence2": item.get("hypothesis", item.get("sentence2", "")),
                "score": float(item.get("relatedness_score", item.get("similarity", 3.0))),
            }
            examples.append(example)
        return examples

    def get_gold_label(self, example: dict) -> float:
        return example["score"]

    def parse_prediction(self, raw_prediction: str) -> str:
        return self._extract_number(raw_prediction)
