"""DoNotAnswer-PT task.

Se7enB/donotanswer_pt is fabricated. No public HF dataset exists for a
Brazilian-Portuguese version of DoNotAnswer as of this audit (a CLARIN-PT-LDB
translation is referenced in arXiv:2603.12872 but is not published as a
standalone HF dataset). Kept disabled by default in
configs/eval/benchmarks.yaml. As an approximate fallback signal (language
mismatch — English, not pt-BR), `LibrAI/do-not-answer` (cc-by-nc-sa) exists
on the Hub, but it is intentionally NOT wired in automatically here.
"""

from typing import Any

from src.eval.tasks.base_task import BaseTask


class DoNotAnswerPTTask(BaseTask):
    """Safety benchmark - model should refuse harmful questions."""

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
                "question": item.get("question", item.get("prompt", "")),
                "category": item.get("category", item.get("risk_area", "")),
                "expected_behavior": "refuse",
            }
            examples.append(example)
        return examples

    def get_gold_label(self, example: dict) -> str:
        return "refuse"

    def parse_prediction(self, raw_prediction: str) -> str:
        # For this task, we check refusal in the metrics module
        return raw_prediction
