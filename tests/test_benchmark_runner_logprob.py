"""Tests for BenchmarkRunner._build_logprob_answer_options.

use_logprob scoring used to hardcode ["A", "B", "C", "D"] as the answer
candidates for ANY accuracy-metric benchmark that lacked an "options"
field — meaningless for tasks like assin2_rte/rte_pt ("entailment"/
"not_entailment") or mrpc_pt ("sim"/"nao"), whose prompts never ask the
model for a letter at all. No test caught this because _inference_logprob
(the part that WAS tested indirectly via manual runs) is generic and
correct; the bug was entirely in how its inputs were built, which had zero
direct coverage.
"""

from src.eval.benchmark_runner import BenchmarkRunner
from src.eval.tasks.assin2_rte import Assin2RTETask
from src.eval.tasks.enem import EnemTask


def _runner():
    return BenchmarkRunner({"evaluation": {}})


class TestLetteredMultipleChoice:
    def test_uses_letters_matching_each_examples_option_count(self):
        runner = _runner()
        task = EnemTask()
        examples = [
            {"question": "q1", "options": ["a", "b", "c", "d"], "answer": "A"},
            {"question": "q2", "options": ["a", "b", "c", "d", "e"], "answer": "E"},
        ]

        result = runner._build_logprob_answer_options(task, examples)

        assert result == [["A", "B", "C", "D"], ["A", "B", "C", "D", "E"]]


class TestFixedVocabularyClassification:
    def test_derives_real_label_set_instead_of_letters(self):
        """The actual regression: assin2_rte examples have no "options"
        field at all — the answer candidates must be the task's real
        labels ("entailment"/"not_entailment"), never A-D letters, since
        that's what the prompt template asks the model to produce."""
        runner = _runner()
        task = Assin2RTETask()
        examples = [
            {"premise": "p1", "hypothesis": "h1", "label": "entailment"},
            {"premise": "p2", "hypothesis": "h2", "label": "not_entailment"},
        ]

        result = runner._build_logprob_answer_options(task, examples)

        assert result == [
            ["entailment", "not_entailment"],
            ["entailment", "not_entailment"],
        ]
        for options in result:
            assert "A" not in options and "B" not in options

    def test_label_set_is_shared_across_all_examples_even_if_one_only_saw_one_label(self):
        """A benchmark's fixed label vocabulary must come from ALL loaded
        examples, not be recomputed per-example (which would silently drop
        the other class whenever a single example's gold happened to be
        only one of the two labels)."""
        runner = _runner()
        task = Assin2RTETask()
        examples = [
            {"premise": "p1", "hypothesis": "h1", "label": "entailment"},
            {"premise": "p2", "hypothesis": "h2", "label": "entailment"},
            {"premise": "p3", "hypothesis": "h3", "label": "not_entailment"},
        ]

        result = runner._build_logprob_answer_options(task, examples)

        assert all(set(options) == {"entailment", "not_entailment"} for options in result)
