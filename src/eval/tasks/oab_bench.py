"""OAB-Bench task."""

from typing import Any

from src.eval.tasks.base_task import BaseTask


class OABBenchTask(BaseTask):
    """OAB (Brazilian Bar Exam) benchmark."""

    def load_data(self, config: dict[str, Any]) -> list[dict]:
        # Se7enB/oab_exams was fabricated. Replaced with eduagarcia/oab_exams
        # (real; split "train", 2210 rows).
        local_path = config.get("local_path")
        hub_id = config.get("hub_id", "eduagarcia/oab_exams")

        if local_path:
            data = self._load_from_local(local_path)
        elif hub_id:
            data = self._load_from_hub(hub_id, split="train")
        else:
            return []

        examples = []
        for item in data:
            # Real schema: question, choices {"text": [...], "label": [...]},
            # answerKey (letter). Same "choices" dict shape as BLUEX/ARC.
            example = {
                "question": item.get("question", item.get("enunciado", "")),
                "options": self._choices_to_options(item.get("choices", item.get("options", []))),
                "answer": item.get("answerKey", item.get("answer", item.get("gabarito", ""))),
            }
            examples.append(example)
        return examples

    def get_gold_label(self, example: dict) -> str:
        return str(example.get("answer", "")).strip().upper()

    def parse_prediction(self, raw_prediction: str) -> str:
        return self._extract_letter(raw_prediction)
