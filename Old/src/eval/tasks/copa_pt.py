"""CoPA-PT task."""

from typing import Any

from src.eval.tasks.base_task import BaseTask


class CopaPTTask(BaseTask):
    """Choice of Plausible Alternatives in Portuguese."""

    def load_data(self, config: dict[str, Any]) -> list[dict]:
        hub_id = config.get("hub_id", "Se7enB/copa_pt")
        local_path = config.get("local_path")

        if local_path:
            data = self._load_from_local(local_path)
        else:
            data = self._load_from_hub(hub_id)

        examples = []
        for item in data:
            premise = item.get("premise", item.get("premissa", ""))
            choice1 = item.get("choice1", item.get("alternativa1", ""))
            choice2 = item.get("choice2", item.get("alternativa2", ""))
            question_type = item.get("question", item.get("tipo", "cause"))
            label = item.get("label", item.get("resposta", 0))

            connector = "porque" if question_type == "cause" else "portanto"
            question = f"{premise} {connector}..."
            options = [choice1, choice2]

            example = {
                "question": question,
                "options": options,
                "answer": str(int(label) + 1),  # 0-indexed to 1-indexed
            }
            examples.append(example)
        return examples

    def get_gold_label(self, example: dict) -> str:
        return str(example.get("answer", ""))

    def parse_prediction(self, raw_prediction: str) -> str:
        text = raw_prediction.strip()
        if "1" in text[:3]:
            return "1"
        if "2" in text[:3]:
            return "2"
        return self._extract_number(text) or "1"
