"""Base class for evaluation tasks."""

import re
from abc import ABC, abstractmethod
from typing import Any


class BaseTask(ABC):
    """Abstract base class for evaluation tasks."""

    def load_data(self, config: dict[str, Any]) -> list[dict]:
        """Load task data from hub or local path.

        Default implementation tries hub_id first, then falls back to local_path.
        Subclasses can override for custom loading logic.
        """
        hub_id = config.get("hub_id")
        subset = config.get("subset")
        local_path = config.get("local_path")
        max_samples = config.get("max_samples")

        data = []
        if hub_id:
            data = self._load_from_hub(hub_id, subset=subset)
        if not data and local_path:
            data = self._load_from_local(local_path)

        if max_samples and len(data) > max_samples:
            data = data[:max_samples]

        return data

    @abstractmethod
    def get_gold_label(self, example: dict) -> Any:
        """Extract gold label from an example."""
        ...

    def parse_prediction(self, raw_prediction: str) -> str:
        """Parse model output into a standardized prediction."""
        # Default: extract first letter/word
        text = raw_prediction.strip()
        return text

    def _extract_letter(self, text: str) -> str:
        """Extract a single letter answer (A-E) from model output.

        Hierarquia de parsing:
        1. Letra isolada
        2. Padrão "A)" ou "A."
        3. Padrão "Resposta: X" / "alternativa X"
        4. Letra em parênteses "(X)"
        5. Letra isolada no início, seguida de separador (não de mais prosa)
        6. Última letra isolada A-E em qualquer posição do texto
        7. Fallback: primeiro caractere, só para texto curto (<=3 chars);
           caso contrário "" (deixa o caller decidir um fallback próprio)
        """
        text = text.strip()
        if not text:
            return ""

        # 1. Exact single letter
        if len(text) == 1 and text.upper() in "ABCDE":
            return text.upper()

        # 2. "A)" or "A." at start (but NOT "A " followed by word)
        match = re.match(r"^([A-Ea-e])[)\.]", text)
        if match:
            return match.group(1).upper()

        # 3. "Resposta: X" / "Answer: X" / "alternativa X" / "é X"
        match = re.search(
            r"(?:resposta|answer|alternativa|é)\s*[:\s]\s*([A-Ea-e])\b", text, re.IGNORECASE
        )
        if match:
            return match.group(1).upper()

        # 4. Letter in parentheses "(X)"
        match = re.search(r"\(([A-Ea-e])\)", text)
        if match:
            return match.group(1).upper()

        # 5. Standalone letter at start (letter + end, or letter + an
        # answer-like separator such as ":"/"-"/")"). Does NOT match a bare
        # letter followed by a space and more prose (e.g. Portuguese "A
        # resposta ...", "A alternativa ...") — that used to match here
        # (any non-alpha, including a plain space, was accepted), which
        # silently mis-scored any free-text response starting with the
        # common article "a"/"A" as if it had answered "A". Prose like that
        # now falls through to steps 6/7 below, which look for an isolated
        # letter anywhere rather than assuming the first word is the answer.
        match = re.match(r"^([A-Ea-e])(?:\s*$|[:\-)])", text)
        if match:
            return match.group(1).upper()

        # 6. Last resort: find isolated letters A-E anywhere, preferring the
        # LAST one. Models conventionally state their final answer near the
        # end of a reasoning chain ("... portanto, a resposta é B"), so the
        # last isolated letter is more often the real answer than the
        # first. A candidate at position 0 immediately followed by more
        # lowercase prose (not punctuation/end-of-string) is excluded
        # entirely — same rationale as step 5, it's an ordinary leading word
        # ("A resposta...", "E" as the conjunction "and"), never a real
        # answer token in that shape.
        candidates = list(re.finditer(r"(?<![a-zA-Z])([A-Ea-e])(?![a-zA-Z])", text))
        candidates = [
            m for m in candidates if not (m.start() == 0 and re.match(r"\s[a-z]", text[m.end() :]))
        ]
        if candidates:
            return candidates[-1].group(1).upper()

        # 7. Final fallback: only trust a bare leading character for
        # genuinely short text (plausibly just the answer itself). For
        # longer free text with no recognizable answer pattern, guessing
        # the first character is usually just its first ordinary word, not
        # an answer — return "" so callers with their own fallback (e.g.
        # numeric extraction) get a chance instead of a wrong letter.
        if len(text) <= 3 and text[0].upper() in "ABCDE":
            return text[0].upper()
        return ""

    def _extract_number(self, text: str) -> str:
        """Extract a number from text."""
        match = re.search(r"(\d+\.?\d*)", text.strip())
        return match.group(1) if match else ""

    def _load_from_hub(
        self,
        hub_id: str,
        subset: str | None = None,
        split: str = "test",
        revision: str | None = None,
        data_files: Any = None,
        trust_remote_code: bool = False,
    ) -> list[dict]:
        """Load data from HuggingFace Hub.

        Args:
            hub_id: Dataset repo id (e.g. "maritaca-ai/enem").
            subset: Dataset config name (passed as `name=` to load_dataset).
            split: Split to load. Falls back to "validation" then "train"
                if the requested split fails to load.
            revision: Specific repo revision/branch. Useful for datasets whose
                default branch relies on a legacy loading script, which the
                installed `datasets` library (>=3.0) no longer executes.
                Passing revision="refs/convert/parquet" loads the Hub's
                auto-converted Parquet mirror instead (verified to work for
                script-based repos like ruanchaves/hatebr and peluz/lener_br).
            data_files: Specific file(s) to load within the repo. Useful when
                a repo mixes multiple incompatible schemas across files
                (e.g. unicamp-dl/PublicHearingBR ships two JSONL files with
                different columns under one repo).
            trust_remote_code: Kept for backwards compatibility/documentation
                purposes only. `datasets>=3.0` no longer executes dataset
                loading scripts even when this is True (it just logs an
                error and ignores the flag) — use `revision` above instead
                for those datasets.
        """
        from datasets import load_dataset

        kwargs: dict[str, Any] = {"split": split}
        if subset:
            kwargs["name"] = subset
        if revision:
            kwargs["revision"] = revision
        if data_files:
            kwargs["data_files"] = data_files
        if trust_remote_code:
            kwargs["trust_remote_code"] = True
        try:
            ds = load_dataset(hub_id, **kwargs)
            return [dict(ex) for ex in ds]
        except Exception:
            # Try other splits
            for fallback_split in ["validation", "train"]:
                try:
                    kwargs["split"] = fallback_split
                    ds = load_dataset(hub_id, **kwargs)
                    return [dict(ex) for ex in ds]
                except Exception:
                    continue
        return []

    def _apply_max_samples(self, data: list, config: dict[str, Any]) -> list:
        """Truncate `data` to config["max_samples"] if set.

        `load_data`'s default implementation applies this automatically, but
        subclasses that override `load_data` (as most task files do, to
        normalize real HF schemas) must call this explicitly, or a
        `max_samples` set in configs/eval/benchmarks.yaml (e.g. the 500-example
        retention_en samples) would silently be ignored.
        """
        max_samples = config.get("max_samples")
        if max_samples and len(data) > max_samples:
            return data[:max_samples]
        return data

    def _choices_to_options(self, choices: Any) -> list:
        """Convert an ARC/BLUEX/OAB-style choices structure into a flat list.

        Several real HF datasets (allenai/ai2_arc, eduagarcia-temp/BLUEX_without_images,
        eduagarcia/oab_exams) encode multiple-choice alternatives as a dict
        `{"text": [...], "label": [...]}` (HF Sequence/ClassLabel style) rather
        than a plain list. This normalizes that shape into an ordered list of
        option strings (ordered by the "label" letters, e.g. A, B, C...), so
        it can be used directly as an "options" field for the existing
        multiple-choice formatter/parsing pipeline.

        Args:
            choices: Either a plain list of option strings, or a dict with
                "text" and "label" keys (parallel lists).

        Returns:
            Flat list of option strings.
        """
        if isinstance(choices, list):
            return choices
        if not choices:
            return []
        texts = list(choices.get("text", []))
        labels = list(choices.get("label", []))
        if labels and len(labels) == len(texts):
            try:
                order = sorted(range(len(labels)), key=lambda i: labels[i])
                return [texts[i] for i in order]
            except TypeError:
                pass
        return texts

    def _load_from_local(self, path: str) -> list[dict]:
        """Load data from local JSONL file."""
        import json
        from pathlib import Path

        p = Path(path)
        if not p.exists():
            return []
        data = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
