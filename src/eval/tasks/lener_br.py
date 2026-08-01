"""Tarefa LeNER-Br — NER jurídico em português.

Formato: texto jurídico → entidades nomeadas (PESSOA, LOCAL, ORGANIZAÇÃO, etc.)
Métrica: entity_micro_f1
Fonte: peluz/lener_br (HuggingFaceGECLNLP/lener_br não existe no Hub).

peluz/lener_br usa um script de carregamento legado, que a biblioteca
`datasets` >= 3.0 instalada não executa mais ("Dataset scripts are no longer
supported"). `revision="refs/convert/parquet"` carrega o mirror Parquet
auto-convertido do Hub (verificado: train 7828 / validation 1177 / test 1390).

Schema real: id, tokens (list[str]), ner_tags (list[int], ClassLabel com
nomes B-/I-ORGANIZACAO, PESSOA, TEMPO, LOCAL, LEGISLACAO, JURISPRUDENCIA, e O).

O `get_gold_label` antigo lia `example.get("entities", [])`, campo que não
existe no schema real (o dataset real é tokens+tags, não spans já prontos),
então o gold era sempre `[]` e entity_micro_f1 ficava hardcoded em 0.0.
Reescrito para decodificar BIO -> spans a partir de tokens+ner_tags.

NOTA sobre offsets: `parse_prediction` só recebe o texto bruto gerado pelo
modelo (sem acesso ao exemplo/tokens de origem — ver benchmark_runner.py),
então não é possível reconstruir offsets de token reais para os spans
preditos. Como simplificação pragmática (documentada aqui e aceitável dado
o formato de saída livre do modelo), tanto o gold quanto a predição usam o
texto normalizado da entidade (lowercased/stripped) no campo "start" e
`None` em "end", de forma que `entity_micro_f1` (que compara tuplas
`(start, end, label)`) efetivamente compara "texto do span normalizado +
tipo" em vez de offsets exatos.
"""

from typing import Any

from src.eval.tasks.base_task import BaseTask


class LeNERBr(BaseTask):
    """Reconhecimento de entidades nomeadas em textos jurídicos brasileiros."""

    task_name = "lener_br"

    def load_data(self, config: dict[str, Any]) -> list[dict]:
        local_path = config.get("local_path")
        if local_path:
            return self._load_from_local(local_path)

        hub_id = config.get("hub_id", "peluz/lener_br")

        from datasets import load_dataset as _hf_load_dataset

        ds = None
        for split in ("test", "validation", "train"):
            try:
                ds = _hf_load_dataset(hub_id, revision="refs/convert/parquet", split=split)
                break
            except Exception:
                continue
        if ds is None:
            return []

        # ner_tags is a Sequence[ClassLabel]; resolve id -> name dynamically
        # from the dataset's own feature metadata rather than hardcoding it.
        tag_names = ds.features["ner_tags"].feature.names

        examples = []
        for item in ds:
            tokens = item.get("tokens", [])
            raw_tags = item.get("ner_tags", [])
            tags = [tag_names[t] if isinstance(t, int) else str(t) for t in raw_tags]
            examples.append(
                {
                    "id": item.get("id"),
                    "tokens": tokens,
                    "ner_tags": tags,
                    "text": " ".join(tokens),
                }
            )
        return examples

    def _bio_to_spans(self, tokens: list, tags: list) -> list[dict]:
        """Converte tags BIO (ner_tags) + tokens em spans de entidade.

        Regra padrão BIO2: uma tag B-X inicia um novo span do tipo X; tags
        I-X contíguas estendem o span enquanto o tipo bater; qualquer outra
        tag (O, B- diferente, ou I- "órfã" sem B- correspondente) fecha o
        span atual. Uma I-X órfã é tratada defensivamente como início de
        span (dados reais podem ter pequenas inconsistências de anotação).
        """
        spans: list[tuple[int, int, str]] = []
        start = None
        label = None
        for i, tag in enumerate(tags):
            if tag.startswith("B-"):
                if label is not None:
                    spans.append((start, i - 1, label))
                start, label = i, tag[2:]
            elif tag.startswith("I-") and label == tag[2:]:
                continue
            else:
                if label is not None:
                    spans.append((start, i - 1, label))
                start, label = None, None
                if tag.startswith("I-"):
                    start, label = i, tag[2:]
        if label is not None:
            spans.append((start, len(tags) - 1, label))

        result = []
        for s, e, lbl in spans:
            text = " ".join(tokens[s : e + 1]).strip().lower()
            result.append({"start": text, "end": None, "label": lbl, "text": text})
        return result

    def parse_prediction(self, raw_output: str) -> list[dict]:
        """Parseia output do modelo para lista de entidades.

        O modelo deve gerar entidades no formato:
        ENTIDADE: texto | TIPO: tipo

        Ver nota de módulo sobre a simplificação de comparação por texto
        normalizado (sem offsets de token) usada para casar com o gold.
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
                    norm_text = entity_text.strip().lower()
                    entities.append(
                        {
                            "start": norm_text,
                            "end": None,
                            "label": entity_type,
                            "text": entity_text,
                        }
                    )
        return entities

    def get_gold_label(self, example: dict) -> list[dict]:
        """Extrai entidades gold do exemplo, decodificando BIO -> spans."""
        tokens = example.get("tokens", [])
        tags = example.get("ner_tags", [])
        if not tokens or not tags:
            # Compat: se o exemplo já veio com spans prontos (ex.: fixture
            # legada com "entities"), usa-os diretamente.
            return example.get("entities", [])
        return self._bio_to_spans(tokens, tags)
