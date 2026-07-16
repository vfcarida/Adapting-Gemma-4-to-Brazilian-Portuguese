"""ASSIN2-RTE task."""

from typing import Any

from src.eval.tasks.base_task import BaseTask


class Assin2RTETask(BaseTask):
    """ASSIN2 Recognizing Textual Entailment."""

    def load_data(self, config: dict[str, Any]) -> list[dict]:
        hub_id = config.get("hub_id", "assin2")
        data = self._load_from_hub(hub_id, split="test")

        examples = []
        for item in data:
            example = {
                "premise": item.get("premise", item.get("sentence1", "")),
                "hypothesis": item.get("hypothesis", item.get("sentence2", "")),
                "label": "entailment"
                if item.get("entailment_judgment", 0) == 1
                else "not_entailment",
            }
            examples.append(example)
        return examples

    def get_gold_label(self, example: dict) -> str:
        return example["label"]

    def parse_prediction(self, raw_prediction: str) -> str:
        text = raw_prediction.strip().lower()
        if "entailment" in text and "not" not in text.split("entailment")[0][-5:]:
            return "entailment"
        return "not_entailment"
