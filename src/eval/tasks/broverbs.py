"""BRoverbs task."""

from typing import Any

from src.eval.tasks.base_task import BaseTask


class BRoverbsTask(BaseTask):
    """Brazilian proverbs completion task."""

    def load_data(self, config: dict[str, Any]) -> list[dict]:
        # Se7enB/broverbs was fabricated. Replaced with Tropic-AI/BRoverbs
        # (real; 193 proverbs / 579 stories). This dataset has no "test"/
        # "train" splits — instead "history_to_proverb" and "proverb_to_history"
        # (579 rows each). Default to "history_to_proverb" (given a story,
        # pick the matching proverb among 5 alternatives), overridable via config.
        local_path = config.get("local_path")
        hub_id = config.get("hub_id", "Tropic-AI/BRoverbs")
        split = config.get("split", "history_to_proverb")

        if local_path:
            data = self._load_from_local(local_path)
        elif hub_id:
            data = self._load_from_hub(hub_id, split=split)
        else:
            return []

        examples = []
        for item in data:
            # Real schema: input, explanation, alternatives (list[str]),
            # correct_index (0-based int, 5-way MC).
            correct_index = item.get("correct_index")
            if correct_index is None:
                answer = item.get("answer", "")
            else:
                answer = chr(65 + int(correct_index))
            example = {
                "question": item.get("input", item.get("question", item.get("proverb_start", ""))),
                "options": item.get("alternatives", item.get("options", item.get("choices", []))),
                "answer": answer,
            }
            examples.append(example)
        return examples

    def get_gold_label(self, example: dict) -> str:
        return str(example.get("answer", "")).strip().upper()

    def parse_prediction(self, raw_prediction: str) -> str:
        return self._extract_letter(raw_prediction)
