"""Tarefa LeNER-Br — NER jurídico em português.

Formato: texto jurídico → entidades nomeadas (PESSOA, LOCAL, ORGANIZAÇÃO, etc.)
Métrica: entity_micro_f1
Fonte: https://huggingface.co/datasets/lener_br

TODO: Implementar parsing de entidades em formato BIO/IOB2
TODO: Suportar geração de entidades em formato estruturado
"""

from src.eval.tasks.base_task import BaseTask


class LeNERBr(BaseTask):
    """Reconhecimento de entidades nomeadas em textos jurídicos brasileiros."""

    task_name = "lener_br"

    def parse_prediction(self, raw_output: str) -> list[dict]:
        """Parseia output do modelo para lista de entidades.

        O modelo deve gerar entidades no formato:
        ENTIDADE: texto | TIPO: tipo

        TODO: Implementar parsing robusto de diferentes formatos de saída.
        """
        entities = []
        lines = raw_output.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Tentar parsear formato "ENTIDADE: X | TIPO: Y"
            if "|" in line and ":" in line:
                parts = line.split("|")
                entity_text = ""
                entity_type = ""
                for part in parts:
                    part = part.strip()
                    if part.upper().startswith("ENTIDADE:"):
                        entity_text = part.split(":", 1)[1].strip()
                    elif part.upper().startswith("TIPO:"):
                        entity_type = part.split(":", 1)[1].strip()
                if entity_text and entity_type:
                    entities.append({"text": entity_text, "label": entity_type})
        return entities

    def get_gold_label(self, example: dict) -> list[dict]:
        """Extrai entidades gold do exemplo."""
        # Formato esperado: lista de dicts com text, label, start, end
        return example.get("entities", [])
