"""Tarefa LegalBench-BR — Benchmark jurídico brasileiro.

Formato: classificação do julgamento de decisões judiciais brasileiras,
reaproveitado como múltipla escolha de 3 alternativas.
Métrica: accuracy

`eduagarcia/legalbench_br` não existe no Hub (fabricado), e não há um
equivalente direto de LegalBench em português brasileiro publicado no HF Hub.
Substituído por `eduagarcia/portuguese_benchmark`, config
"brazilian_court_decisions_judgment" (split "test", 405 linhas): classifica
o resultado de acórdãos brasileiros em 3 classes ("no"/"partial"/"yes").
Um segundo benchmark baseado na OAB (como `eduagarcia/oab_exams`) seria
redundante com `oab_bench`, por isso não foi usado aqui.

(Alternativa mais próxima, mas em português europeu, não brasileiro, caso
se queira algo mais próximo do LegalBench original: `BeatrizCanaverde/LegalBench.PT`.)

Para reaproveitar a infraestrutura de múltipla escolha já existente
(`_default_formatter`/`TASK_INSTRUCTIONS["legalbench_br"]` em
prompt_templates.py, que não é alterado neste fix), as 3 classes são
mapeadas para alternativas A/B/C.
"""

from typing import Any

from src.eval.tasks.base_task import BaseTask

# Ordem fixa das classes reais (ClassLabel names do dataset) -> letras A/B/C.
_LABEL_ORDER = ["no", "partial", "yes"]
_LABEL_TO_OPTION_TEXT = {
    "no": "O pedido foi negado (improcedente).",
    "partial": "O pedido foi parcialmente provido.",
    "yes": "O pedido foi provido (procedente).",
}


class LegalBenchBR(BaseTask):
    """Benchmark de competência jurídica brasileira (classificação de julgamento)."""

    task_name = "legalbench_br"

    def load_data(self, config: dict[str, Any]) -> list[dict]:
        local_path = config.get("local_path")
        if local_path:
            return self._load_from_local(local_path)

        hub_id = config.get("hub_id", "eduagarcia/portuguese_benchmark")
        subset = config.get("subset", "brazilian_court_decisions_judgment")

        from datasets import load_dataset as _hf_load_dataset

        try:
            ds = _hf_load_dataset(hub_id, name=subset, split="test")
        except Exception:
            try:
                ds = _hf_load_dataset(hub_id, name=subset, split="validation")
            except Exception:
                return []

        # "label" is a ClassLabel; resolve int -> class name via ds.features
        # rather than hardcoding an id->name mapping.
        label_feature = ds.features.get("label")
        label_names = list(getattr(label_feature, "names", _LABEL_ORDER))

        options = [_LABEL_TO_OPTION_TEXT[name] for name in _LABEL_ORDER]

        examples = []
        for item in ds:
            label_val = item.get("label")
            if isinstance(label_val, int) and 0 <= label_val < len(label_names):
                label_str = label_names[label_val]
            else:
                label_str = str(label_val)
            answer = chr(65 + _LABEL_ORDER.index(label_str)) if label_str in _LABEL_ORDER else ""
            examples.append(
                {
                    "question": item.get("sentence", ""),
                    "options": options,
                    "answer": answer,
                }
            )
        return examples

    def parse_prediction(self, raw_output: str) -> str:
        """Extrai letra da alternativa escolhida (parser robusto de BaseTask)."""
        return self._extract_letter(raw_output)

    def get_gold_label(self, example: dict) -> str:
        """Extrai resposta correta."""
        return str(example.get("answer", example.get("label", ""))).strip().upper()
