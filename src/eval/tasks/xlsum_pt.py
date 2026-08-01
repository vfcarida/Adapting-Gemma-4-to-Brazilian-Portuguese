"""XLSum-PT task."""

from typing import Any

from src.eval.tasks.base_task import BaseTask


class XLSumPTTask(BaseTask):
    """XLSum Portuguese summarization task."""

    def load_data(self, config: dict[str, Any]) -> list[dict]:
        hub_id = config.get("hub_id", "csebuetnlp/xlsum")
        subset = config.get("subset", "portuguese")

        # csebuetnlp/xlsum relies on a legacy loading script ("Dataset
        # scripts are no longer supported" on datasets>=3.0); pin to the
        # Hub's auto-converted Parquet mirror instead, same underlying issue
        # as hatebr.py/lener_br.py. Unlike those, the mirror doesn't expose
        # "portuguese" as a builder config (only "default") — verified live
        # that the language split lives at a fixed per-language file path
        # instead, so it must be loaded via data_files, not subset=.
        data = self._load_from_hub(
            hub_id,
            split="test",
            revision="refs/convert/parquet",
            data_files={"test": f"{subset}/test/0000.parquet"},
        )

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
