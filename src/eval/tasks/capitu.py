"""Tarefa CAPITU — Benchmark de instruction-following com contexto literário.

Verificado ao vivo em 2026-08-01 (arXiv:2603.22576, "CAPITU: A Benchmark for
Evaluating Instruction-Following in Brazilian Portuguese with Literary
Context", Maritaca AI, 2026): este NÃO é um benchmark de compreensão de
leitura/cultura — é instruction-following (59 tipos de instrução
verificáveis, ex. restrições de morfologia do português) usando 8 obras
canônicas da literatura brasileira como contexto, 200 prompts single-turn +
100 multi-turn (3 turnos), construído sobre o framework IFEval do Google.
Métrica esperada é taxa de instruções seguidas corretamente (estilo IFEval),
não accuracy de múltipla escolha — `get_gold_label`/`parse_prediction`
abaixo assumem incorretamente um formato de alternativa (A-E) herdado do
padrão genérico de outras tasks; precisam ser reescritos para o formato
real (verificação programática por tipo de instrução) antes de habilitar.

Não existe dataset público no HF Hub para este benchmark — os dados reais
estão em github.com/maritaca-ai/capitu (Apache-2.0), não no HF Hub. Mantido
desabilitado por padrão em configs/eval/benchmarks.yaml; para habilitar,
obtenha os dados diretamente do repositório do GitHub e aponte `local_path`
para um JSONL local (ver BaseTask.load_data / o mesmo padrão usado em
broverbs.py e donotanswer_pt.py) — mas note que isso sozinho não basta: a
lógica de scoring abaixo também precisa ser trocada.
"""

from src.eval.tasks.base_task import BaseTask


class Capitu(BaseTask):
    """Benchmark CAPITU de instruction-following com contexto literário."""

    task_name = "capitu"

    def parse_prediction(self, raw_output: str) -> str:
        """PLACEHOLDER — não reflete o formato real do benchmark.

        CAPITU verifica se instruções específicas foram seguidas (estilo
        IFEval — ex. "responda usando apenas substantivos", "não use a
        letra 'a'"), não múltipla escolha A-E. Esta implementação letter-
        based é herdada do padrão genérico das outras tasks e está aqui só
        para não quebrar a interface de BaseTask; precisa ser substituída
        por verificadores programáticos por tipo de instrução antes deste
        benchmark ser habilitado de verdade.
        """
        return self._extract_letter(raw_output)

    def get_gold_label(self, example: dict) -> str:
        """PLACEHOLDER — ver nota em parse_prediction."""
        return str(example.get("answer", example.get("label", ""))).strip().upper()
