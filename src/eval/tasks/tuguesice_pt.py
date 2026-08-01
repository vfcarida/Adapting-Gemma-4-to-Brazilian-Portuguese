"""Tuguesice-PT task.

CORRECTION (live-verified 2026-08-01): this benchmark is real, NOT
fabricated as a prior pass here claimed. It's introduced in "CLARIN-PT-LDB:
An Open LLM Leaderboard for Portuguese to assess Language, Culture and
Civility" (Silva, Gomes, Branco — U. Lisbon, PROPOR 2026, arXiv:2603.12872,
ACL Anthology 2026.propor-1.7) — a manually-created 327-item QA benchmark
covering Society/Geography/History/Politics/Cuisine/Sports, part of the
`PORTULAN/portuguese-llm-leaderboard` HF Space.

Still not usable here, for two DIFFERENT reasons than "fabricated":
1. No public dataset file was found — the HF Space's visible file tree
   doesn't include the actual QA items (likely held out to prevent
   contamination, standard practice for leaderboard benchmarks).
2. Even if the data surfaces, it targets *European* Portuguese cultural
   knowledge (PORTULAN/Gervásio's focus), not Brazilian — a poor fit for
   this project's PT-BR adaptation goal regardless of availability.

Not wired into configs/eval/benchmarks.yaml.
"""

from typing import Any

from src.eval.tasks.base_task import BaseTask


class TuguesicePTTask(BaseTask):
    """Portuguese language and culture knowledge task."""

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
                "question": item.get("question", item.get("pergunta", "")),
                "options": item.get("options", item.get("alternativas", [])),
                "answer": item.get("answer", item.get("resposta", "")),
            }
            examples.append(example)
        return examples

    def get_gold_label(self, example: dict) -> str:
        return str(example.get("answer", "")).strip().upper()

    def parse_prediction(self, raw_prediction: str) -> str:
        return self._extract_letter(raw_prediction)
