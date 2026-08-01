"""RTE-PT task.

Real dataset: PORTULAN/extraglue, config "rte_pt-BR"
(train 2490 / validation 277 / test 3000). Previously dead/orphaned (not
referenced in configs/eval/benchmarks.yaml and had no default hub_id).
"""

from typing import Any

from src.eval.tasks.base_task import BaseTask


class RTEPTTask(BaseTask):
    """Recognizing Textual Entailment in Portuguese."""

    def load_data(self, config: dict[str, Any]) -> list[dict]:
        local_path = config.get("local_path")
        hub_id = config.get("hub_id", "PORTULAN/extraglue")
        subset = config.get("subset", "rte_pt-BR")

        if local_path:
            data = self._load_from_local(local_path)
        elif hub_id:
            # NOTE: the "test" split has its label masked to -1 for every row
            # (GLUE test-label-hiding convention, inherited by extraglue).
            # "validation" is the only split with real, usable gold labels.
            data = self._load_from_hub(hub_id, subset=subset, split="validation")
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
