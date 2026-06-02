"""Tests for evaluation metrics (requires scipy, sklearn)."""

import pytest

pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from src.eval.metrics import (
    accuracy,
    compute_metrics_for_task,
    macro_f1,
    pearson,
    refusal_rate,
    rouge_l,
)


class TestAccuracy:
    def test_perfect(self):
        result = accuracy(["A", "B", "C"], ["A", "B", "C"])
        assert result["accuracy"] == 1.0

    def test_zero(self):
        result = accuracy(["A", "A", "A"], ["B", "B", "B"])
        assert result["accuracy"] == 0.0

    def test_partial(self):
        result = accuracy(["A", "B", "C", "D"], ["A", "B", "X", "Y"])
        assert result["accuracy"] == 0.5

    def test_case_insensitive(self):
        result = accuracy(["a", "b"], ["A", "B"])
        assert result["accuracy"] == 1.0

    def test_whitespace_handling(self):
        result = accuracy(["  A  ", " B"], ["A", "B"])
        assert result["accuracy"] == 1.0


class TestMacroF1:
    def test_perfect(self):
        result = macro_f1(["pos", "neg", "pos"], ["pos", "neg", "pos"])
        assert result["macro_f1"] == 1.0

    def test_zero(self):
        result = macro_f1(["a", "a", "a"], ["b", "b", "b"])
        assert result["macro_f1"] == 0.0


class TestPearson:
    def test_perfect_correlation(self):
        result = pearson(["1", "2", "3", "4", "5"], [1, 2, 3, 4, 5])
        assert result["pearson"] == pytest.approx(1.0, abs=0.001)

    def test_invalid_inputs(self):
        result = pearson(["abc", "def"], [1, 2])
        assert result["n_valid"] == 0


class TestRefusalRate:
    def test_all_refuse(self):
        preds = ["Desculpe, nao posso ajudar", "Nao posso responder isso"]
        result = refusal_rate(preds, ["refuse", "refuse"])
        assert result["refusal_rate"] == 1.0

    def test_no_refuse(self):
        preds = ["Claro, aqui esta a resposta", "A resposta e 42"]
        result = refusal_rate(preds, ["refuse", "refuse"])
        assert result["refusal_rate"] == 0.0

    def test_partial_refuse(self):
        preds = ["Desculpe, nao posso", "A resposta e 42"]
        result = refusal_rate(preds, ["refuse", "refuse"])
        assert result["refusal_rate"] == 0.5


class TestRougeL:
    def test_identical(self):
        result = rouge_l(["hello world"], ["hello world"])
        assert result["rouge_l"] == 1.0

    def test_no_overlap(self):
        result = rouge_l(["foo bar"], ["baz qux"])
        assert result["rouge_l"] == 0.0


class TestComputeMetricsForTask:
    def test_unknown_metric(self):
        with pytest.raises(ValueError):
            compute_metrics_for_task("unknown_metric", [], [])

    def test_dispatch(self):
        result = compute_metrics_for_task("accuracy", ["A", "B"], ["A", "B"])
        assert result["accuracy"] == 1.0
