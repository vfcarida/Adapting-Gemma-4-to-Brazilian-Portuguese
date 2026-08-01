"""BLUEX benchmark task."""

from typing import Any

from src.eval.tasks.base_task import BaseTask


class BluexTask(BaseTask):
    """BLUEX university entrance exam questions."""

    def load_data(self, config: dict[str, Any]) -> list[dict]:
        # Se7enB/bluex was fabricated (does not exist on the Hub). Replaced with
        # eduagarcia-temp/BLUEX_without_images: public, split "train", 724 rows,
        # no images (better fit for text-only eval than the original BLUEX repo).
        hub_id = config.get("hub_id", "eduagarcia-temp/BLUEX_without_images")
        data = self._load_from_hub(hub_id, split="train")

        examples = []
        for item in data:
            # Real schema: question, choices {"text": [...], "label": [...]},
            # answerKey (letter). "choices" mirrors the allenai/ai2_arc shape.
            example = {
                "question": item.get("question", item.get("pergunta", "")),
                "options": self._choices_to_options(item.get("choices", item.get("options", []))),
                "answer": item.get("answerKey", item.get("answer", item.get("correct", ""))),
            }
            examples.append(example)
        return examples

    def get_gold_label(self, example: dict) -> str:
        return str(example.get("answer", "")).strip().upper()

    def parse_prediction(self, raw_prediction: str) -> str:
        return self._extract_letter(raw_prediction)
