"""MRPC-PT task."""

from typing import Any

from src.eval.tasks.base_task import BaseTask


class MRPCPTTask(BaseTask):
    """MRPC paraphrase detection in Portuguese."""

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
                "sentence1": item.get("sentence1", item.get("premise", "")),
                "sentence2": item.get("sentence2", item.get("hypothesis", "")),
                "label": "sim" if item.get("label", 0) == 1 else "nao",
            }
            examples.append(example)
        return examples

    def get_gold_label(self, example: dict) -> str:
        return example["label"]

    def parse_prediction(self, raw_prediction: str) -> str:
        text = raw_prediction.strip().lower()
        if "sim" in text or "yes" in text or "parafrase" in text:
            return "sim"
        return "nao"
