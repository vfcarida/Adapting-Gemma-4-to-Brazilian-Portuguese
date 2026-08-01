"""CoPA-PT task."""

from typing import Any

from src.eval.tasks.base_task import BaseTask


class CopaPTTask(BaseTask):
    """Choice of Plausible Alternatives in Portuguese (PORTULAN extraglue)."""

    def load_data(self, config: dict[str, Any]) -> list[dict]:
        # Se7enB/copa_pt was fabricated. Replaced with PORTULAN/extraglue,
        # config "copa_pt-BR" (train 400 / validation 100 / test 500).
        hub_id = config.get("hub_id", "PORTULAN/extraglue")
        subset = config.get("subset", "copa_pt-BR")
        local_path = config.get("local_path")

        if local_path:
            data = self._load_from_local(local_path)
        else:
            # NOTE: the "test" split of PORTULAN/extraglue has its label
            # masked to -1 for every row (inherited GLUE/SuperGLUE convention
            # of hiding test labels). "validation" is the only split with
            # real, usable gold labels, so it is used here for evaluation.
            data = self._load_from_hub(hub_id, subset=subset, split="validation")

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
                # 0/1 (int) -> "A"/"B", matching the A)/B) options rendered by
                # the default MC formatter (see prompt_templates._default_formatter)
                # and the letter-based instruction in TASK_INSTRUCTIONS["copa_pt"].
                "answer": chr(65 + int(label)),
            }
            examples.append(example)
        return examples

    def get_gold_label(self, example: dict) -> str:
        return str(example.get("answer", "")).strip().upper()

    def parse_prediction(self, raw_prediction: str) -> str:
        # Previously scanned for "1"/"2" in the raw text, which mismatched
        # the rendered "A) / B)" options and TASK_INSTRUCTIONS wording,
        # pinning accuracy near chance regardless of model quality. Use the
        # shared, well-tested letter extractor instead (see base_task.py).
        return self._extract_letter(raw_prediction)
