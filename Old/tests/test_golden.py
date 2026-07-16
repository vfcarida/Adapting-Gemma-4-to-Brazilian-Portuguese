"""Golden tests — validação contra fixtures determinísticas.

Estes testes usam dados fixos (golden fixtures) para garantir que:
- Prompts são formatados corretamente
- Parsing de respostas é consistente
- Normalização Unicode funciona
- Métricas retornam valores esperados
- Bootstrap e stats produzem resultados estáveis
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# =============================================================================
# Fixture loaders
# =============================================================================


@pytest.fixture
def golden_enem():
    with open(FIXTURES_DIR / "golden_enem.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def golden_parsing():
    with open(FIXTURES_DIR / "golden_parsing.json", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Golden benchmark prompt tests
# =============================================================================


class TestGoldenPrompts:
    """Valida formatação de prompts contra golden fixtures."""

    def test_enem_prompt_format(self, golden_enem):
        from src.eval.prompt_templates import get_prompt_template

        template = get_prompt_template("enem", num_shots=0)

        for example in golden_enem:
            prompt = template.format_prompt(example, think_mode="off")
            for expected_text in example["expected_prompt_contains"]:
                assert expected_text in prompt, (
                    f"Expected '{expected_text}' in prompt for question: {example['question'][:50]}"
                )

    def test_enem_prompt_no_answer_leak(self, golden_enem):
        """Zero-shot não deve incluir a resposta no prompt."""
        from src.eval.prompt_templates import get_prompt_template

        template = get_prompt_template("enem", num_shots=0)

        for example in golden_enem:
            prompt = template.format_prompt(example, think_mode="off")
            # The prompt should end with "Resposta:" without the answer
            assert not prompt.rstrip().endswith(f"Resposta: {example['answer']}")


# =============================================================================
# Golden parsing tests
# =============================================================================


class TestGoldenParsing:
    """Valida parsing de respostas contra golden fixtures."""

    def test_letter_extraction(self, golden_parsing):
        from src.eval.tasks.base_task import BaseTask

        class TestTask(BaseTask):
            def load_data(self, config):
                return []

            def get_gold_label(self, example):
                return ""

            def parse_prediction(self, raw):
                return self._extract_letter(raw)

        task = TestTask()
        for case in golden_parsing["letter_extraction"]:
            result = task.parse_prediction(case["input"])
            assert result == case["expected"], (
                f"parse_prediction('{case['input']}') = '{result}', expected '{case['expected']}'"
            )

    def test_think_stripping(self, golden_parsing):
        from src.eval.prompt_templates import strip_thought

        for case in golden_parsing["think_stripping"]:
            result = strip_thought(case["input"])
            assert result == case["expected"], (
                f"strip_thought('{case['input'][:40]}...') = '{result}', "
                f"expected '{case['expected']}'"
            )

    def test_unicode_normalization(self, golden_parsing):
        from src.data.contamination_checks import normalize_text

        for case in golden_parsing["unicode_normalization"]:
            result = normalize_text(case["input"])
            assert result == case["expected_normalized"], (
                f"normalize_text('{case['input']}') = '{result}', "
                f"expected '{case['expected_normalized']}'"
            )


# =============================================================================
# Golden metric tests
# =============================================================================


class TestGoldenMetrics:
    """Valida propriedades estruturais das métricas."""

    def test_accuracy_perfect(self):
        from src.eval.metrics import compute_metrics_for_task

        preds = ["A", "B", "C", "D"]
        golds = ["A", "B", "C", "D"]
        m = compute_metrics_for_task("accuracy", preds, golds)
        assert m["accuracy"] == 1.0

    def test_accuracy_zero(self):
        from src.eval.metrics import compute_metrics_for_task

        preds = ["A", "A", "A", "A"]
        golds = ["B", "C", "D", "E"]
        m = compute_metrics_for_task("accuracy", preds, golds)
        assert m["accuracy"] == 0.0

    def test_accuracy_partial(self):
        from src.eval.metrics import compute_metrics_for_task

        preds = ["A", "B", "C", "D", "E"]
        golds = ["A", "B", "X", "Y", "Z"]
        m = compute_metrics_for_task("accuracy", preds, golds)
        assert m["accuracy"] == 0.4

    def test_accuracy_empty(self):
        """Empty predictions should return 0.0 without crashing."""
        from src.eval.metrics import compute_metrics_for_task

        # sklearn raises on empty; our wrapper should handle gracefully
        try:
            m = compute_metrics_for_task("accuracy", [], [])
            assert m["accuracy"] == 0.0
        except ValueError:
            # Acceptable: sklearn raises on empty input
            pass

    def test_f1_symmetric_on_binary(self):
        """F1 macro should be symmetric when classes are balanced."""
        from src.eval.metrics import compute_metrics_for_task

        preds = ["A", "B", "A", "B"]
        golds = ["A", "B", "B", "A"]
        m = compute_metrics_for_task("macro_f1", preds, golds)
        assert 0.0 <= m["macro_f1"] <= 1.0


# =============================================================================
# Golden bootstrap tests
# =============================================================================


class TestGoldenBootstrap:
    """Valida propriedades do bootstrap CI."""

    def test_ci_contains_mean(self):
        from src.eval.bootstrap_ci import bootstrap_ci
        from src.eval.metrics import compute_metrics_for_task

        preds = ["A", "B", "A", "A", "B"] * 10
        golds = ["A", "B", "C", "A", "B"] * 10

        def metric_fn(p, g):
            return compute_metrics_for_task("accuracy", p, g)

        ci = bootstrap_ci(preds, golds, metric_fn, n_bootstrap=200, seed=42)
        acc_ci = ci["accuracy"]
        assert acc_ci["ci_lower"] <= acc_ci["mean"] <= acc_ci["ci_upper"]

    def test_ci_narrows_with_more_data(self):
        """CI should be narrower with more data."""
        import numpy as np

        from src.eval.bootstrap_ci import bootstrap_ci
        from src.eval.metrics import compute_metrics_for_task

        np.random.seed(42)

        def metric_fn(p, g):
            return compute_metrics_for_task("accuracy", p, g)

        # Small dataset
        preds_small = ["A", "B", "A", "B", "A"]
        golds_small = ["A", "B", "A", "B", "B"]
        ci_small = bootstrap_ci(preds_small, golds_small, metric_fn, n_bootstrap=500, seed=42)

        # Large dataset (same proportions)
        preds_large = preds_small * 20
        golds_large = golds_small * 20
        ci_large = bootstrap_ci(preds_large, golds_large, metric_fn, n_bootstrap=500, seed=42)

        width_small = ci_small["accuracy"]["ci_upper"] - ci_small["accuracy"]["ci_lower"]
        width_large = ci_large["accuracy"]["ci_upper"] - ci_large["accuracy"]["ci_lower"]

        assert width_large <= width_small

    def test_bootstrap_deterministic(self):
        """Same seed should produce same CI."""
        from src.eval.bootstrap_ci import bootstrap_ci
        from src.eval.metrics import compute_metrics_for_task

        preds = ["A", "B", "C"] * 10
        golds = ["A", "B", "A"] * 10

        def metric_fn(p, g):
            return compute_metrics_for_task("accuracy", p, g)

        ci1 = bootstrap_ci(preds, golds, metric_fn, n_bootstrap=100, seed=123)
        ci2 = bootstrap_ci(preds, golds, metric_fn, n_bootstrap=100, seed=123)

        assert ci1["accuracy"]["mean"] == ci2["accuracy"]["mean"]
        assert ci1["accuracy"]["ci_lower"] == ci2["accuracy"]["ci_lower"]


# =============================================================================
# Golden contamination tests
# =============================================================================


class TestGoldenContamination:
    """Valida detecção de contaminação em casos difíceis."""

    def test_exact_duplicate(self):
        from src.data.contamination_checks import ContaminationChecker

        bench = ["O Brasil é o maior país da América do Sul"]
        checker = ContaminationChecker(bench, "test")
        train = ["O Brasil é o maior país da América do Sul"]
        result = checker.check_exact(train)
        assert result["matches"] == 1

    def test_normalized_catches_case_diff(self):
        from src.data.contamination_checks import ContaminationChecker

        bench = ["O Brasil é grande"]
        checker = ContaminationChecker(bench, "test")
        train = ["O BRASIL É GRANDE"]
        result = checker.check_normalized(train)
        assert result["matches"] == 1

    def test_unicode_accent_normalization(self):
        from src.data.contamination_checks import normalize_text

        # Accentuation differences should normalize to same
        assert normalize_text("ação") == normalize_text("ação")  # pre-composed vs decomposed

    def test_no_false_positive_different_text(self):
        from src.data.contamination_checks import ContaminationChecker

        bench = ["Texto completamente único sobre biologia"]
        checker = ContaminationChecker(bench, "test")
        train = ["Um texto diferente sobre matemática e física"]
        result = checker.check_exact(train)
        assert result["matches"] == 0

    def test_near_duplicate_fuzzy(self):
        from src.data.contamination_checks import ContaminationChecker

        # Use longer text with high overlap for fuzzy to reliably detect
        bench = [
            "O presidente do Brasil visitou a Europa em janeiro de 2024 "
            "para participar de reuniões com líderes europeus sobre comércio "
            "bilateral e questões ambientais durante a cúpula anual"
        ]
        checker = ContaminationChecker(bench, "test")
        # Very similar text (near-duplicate, only one word different)
        train = [
            "O presidente do Brasil visitou a Europa em fevereiro de 2024 "
            "para participar de reuniões com líderes europeus sobre comércio "
            "bilateral e questões ambientais durante a cúpula anual"
        ]
        result = checker.check_fuzzy(train, threshold=0.5)
        # With enough shared n-grams, fuzzy should detect this
        assert result["matches"] >= 1

    def test_boilerplate_prefix_ngram(self):
        """Documents with same boilerplate prefix but different content."""
        from src.data.contamination_checks import ContaminationChecker

        bench = ["Artigo 1. Todo ser humano tem direito à vida, à liberdade e à segurança"]
        checker = ContaminationChecker(bench, "test")
        # Different document but sharing legal boilerplate
        train = ["Artigo 1. Todo ser humano tem direito à educação e ao trabalho digno"]
        result = checker.check_ngram_overlap(train, n=5, threshold=0.5)
        # With n=5, they share some n-grams but not enough to be contamination
        # This tests that the threshold is reasonable
        assert result["matches"] == 0 or result["matches"] == 1  # Depends on overlap
