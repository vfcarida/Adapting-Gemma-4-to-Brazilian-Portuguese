"""Tarefa PublicHearingBR — Classificação de audiências públicas.

Formato: texto de audiência pública → classificação temática
Métrica: macro_f1
Fonte: unicamp-dl/PublicHearingBR (206 transcrições de audiências públicas).

Se7enB/publichearing_br era fabricado; o dataset real existe mas é
naturalmente um dataset de SUMARIZAÇÃO de documentos longos (campo
"transcricao" -> campo "materia"), não de classificação categórica. Para
reaproveitar a infraestrutura de classificação já existente
(macro_f1 + _classification_formatter, definidos em prompt_templates.py e
não alterados neste fix), usamos `metadados.assunto` (tema principal
extraído do artigo de notícia) como rótulo aproximado de classificação.
O campo `summary` (materia) é mantido no exemplo normalizado para uso
futuro caso esta tarefa seja migrada para sumarização (rouge_l).

O repositório do dataset mistura dois arquivos JSONL com esquemas
incompatíveis (LDS e NLI); `data_files` seleciona explicitamente o arquivo
de sumarização (LDS) para evitar um erro de cast ao carregar via
`load_dataset` sem esse parâmetro.
"""

from typing import Any

from src.eval.tasks.base_task import BaseTask


class PublicHearingBR(BaseTask):
    """Classificação de transcrições de audiências públicas brasileiras."""

    task_name = "publichearing_br"

    def load_data(self, config: dict[str, Any]) -> list[dict]:
        local_path = config.get("local_path")
        if local_path:
            data = self._load_from_local(local_path)
        else:
            hub_id = config.get("hub_id", "unicamp-dl/PublicHearingBR")
            data_files = config.get("data_files", "PublicHearingBR_LDS.jsonl")
            data = self._load_from_hub(hub_id, split="train", data_files=data_files)

        examples = []
        for item in data:
            metadados = item.get("metadados") or {}
            label = str(metadados.get("assunto", item.get("label", ""))).strip().lower()
            examples.append(
                {
                    "text": item.get("transcricao", item.get("text", "")),
                    "label": label,
                    "summary": item.get("materia", ""),
                }
            )
        return examples

    def parse_prediction(self, raw_output: str) -> str:
        """Extrai classificação temática."""
        return raw_output.strip().lower()

    def get_gold_label(self, example: dict) -> str:
        """Extrai label gold."""
        return str(example.get("label", example.get("category", ""))).strip().lower()
