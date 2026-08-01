"""Tarefa CAPITU — Benchmark de compreensão cultural brasileira.

Formato: questões sobre literatura, cultura e história brasileira
Métrica: accuracy

Não existe dataset público no HF Hub para este benchmark (papel
arXiv:2603.22576, código em github.com/maritaca-ai/capitu, sem dataset no HF
Hub). Mantido desabilitado por padrão em configs/eval/benchmarks.yaml; para
habilitar, obtenha os dados diretamente do repositório do GitHub e aponte
`local_path` para um JSONL local (ver BaseTask.load_data / o mesmo padrão
usado em broverbs.py e donotanswer_pt.py).
"""

from src.eval.tasks.base_task import BaseTask


class Capitu(BaseTask):
    """Benchmark CAPITU de compreensão cultural brasileira."""

    task_name = "capitu"

    def parse_prediction(self, raw_output: str) -> str:
        """Extrai letra da alternativa (parser robusto de BaseTask).

        Substituído o scan ingênuo por caractere (que dava falso positivo em
        palavras como "RESULTADO"/"RESPOSTA", que contêm letras A-E) pelo
        parser já testado em base_task.py.
        """
        return self._extract_letter(raw_output)

    def get_gold_label(self, example: dict) -> str:
        """Extrai resposta correta."""
        return str(example.get("answer", example.get("label", ""))).strip().upper()
