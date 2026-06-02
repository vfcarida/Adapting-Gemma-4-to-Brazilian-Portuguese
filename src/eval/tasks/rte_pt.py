"""RTE-PT task."""

from typing import Any

from src.eval.tasks.base_task import BaseTask


class RTEPTTask(BaseTask):
    """Recognizing Textual Entailment in Portuguese."""

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
                "premise": item.get("premise", item.get("sentence1", "")),
                "hypothesis": item.get("hypothesis", item.get("sentence2", "")),
                "label": item.get("label", "not_entailment"),
            }
            if isinstance(example["label"], int):
                example["label"] = "entailment" if example["label"] == 0 else "not_entailment"
            examples.append(example)
        return examples

    def get_gold_label(self, example: dict) -> str:
        return example["label"]

    def parse_prediction(self, raw_prediction: str) -> str:
        text = raw_prediction.strip().lower()
        if "entailment" in text and "not" not in text.split("entailment")[0][-5:]:
            return "entailment"
        return "not_entailment"
