"""MRPC-PT task.

Real dataset: PORTULAN/extraglue, config "mrpc_pt-BR"
(train 3668 / validation 408 / test 1725). Previously dead/orphaned (not
referenced in configs/eval/benchmarks.yaml and had no default hub_id).
"""

from typing import Any

from src.eval.tasks.base_task import BaseTask


class MRPCPTTask(BaseTask):
    """MRPC paraphrase detection in Portuguese."""

    def load_data(self, config: dict[str, Any]) -> list[dict]:
        local_path = config.get("local_path")
        hub_id = config.get("hub_id", "PORTULAN/extraglue")
        subset = config.get("subset", "mrpc_pt-BR")

        if local_path:
            data = self._load_from_local(local_path)
        elif hub_id:
            # Unlike copa/boolq/rte, the "test" split of mrpc_pt-BR does
            # carry real (non-masked) labels, so it's safe to use directly.
            data = self._load_from_hub(hub_id, subset=subset, split="test")
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
