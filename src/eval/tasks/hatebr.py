"""HateBR task."""

from typing import Any

from src.eval.tasks.base_task import BaseTask


class HateBRTask(BaseTask):
    """HateBR hate speech detection."""

    def load_data(self, config: dict[str, Any]) -> list[dict]:
        # Se7enB/HateBR was fabricated. Replaced with ruanchaves/hatebr (real).
        # That repo's main branch uses a legacy loading script, which the
        # installed datasets>=3.0 no longer executes ("Dataset scripts are no
        # longer supported"); revision="refs/convert/parquet" loads the Hub's
        # auto-converted Parquet mirror instead (verified: 1400 test rows).
        hub_id = config.get("hub_id", "ruanchaves/hatebr")
        data = self._load_from_hub(
            hub_id, split="test", revision="refs/convert/parquet", trust_remote_code=True
        )

        examples = []
        for item in data:
            # Real schema: instagram_comments (str), offensive_language (bool).
            text = item.get("instagram_comments", item.get("text", ""))
            offensive = item.get("offensive_language")
            if offensive is None:
                offensive = item.get("label", False)
            label = "odio" if bool(offensive) else "nao_odio"
            example = {
                "text": text,
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
