"""Tests for bootstrap confidence intervals (requires scipy, sklearn, numpy)."""

import pytest

pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from src.eval.bootstrap_ci import bootstrap_ci, paired_bootstrap_test
from src.eval.metrics import accuracy


class TestBootstrapCI:
    def test_perfect_predictions(self):
        preds = ["A"] * 100
        gold = ["A"] * 100
        result = bootstrap_ci(preds, gold, accuracy, n_bootstrap=100)
        assert result["accuracy"]["mean"] == pytest.approx(1.0, abs=0.01)
        assert result["accuracy"]["ci_lower"] >= 0.95

    def test_random_predictions(self):
        preds = ["A", "B"] * 50
        gold = ["A", "A"] * 50
        result = bootstrap_ci(preds, gold, accuracy, n_bootstrap=100)
        assert 0.3 < result["accuracy"]["mean"] < 0.7

    def test_ci_contains_mean(self):
        preds = ["A", "B", "A", "A"] * 25
        gold = ["A", "A", "B", "A"] * 25
        result = bootstrap_ci(preds, gold, accuracy, n_bootstrap=500)
        for key in result:
            assert result[key]["ci_lower"] <= result[key]["mean"] <= result[key]["ci_upper"]


class TestPairedBootstrap:
    def test_better_model_wins(self):
        gold = ["A"] * 100
        preds_a = ["A"] * 90 + ["B"] * 10  # 90% acc
        preds_b = ["A"] * 50 + ["B"] * 50  # 50% acc
        result = paired_bootstrap_test(
            preds_a, preds_b, gold, accuracy, "accuracy", n_bootstrap=200
        )
        assert result["p_value_a_gt_b"] < 0.05
        assert result["significant_at_05"] is True

    def test_equal_models(self):
        gold = ["A"] * 100
        preds = ["A"] * 70 + ["B"] * 30
        result = paired_bootstrap_test(preds, preds, gold, accuracy, "accuracy", n_bootstrap=200)
        # Identical predictions: A never beats B (score_a == score_b always)
        # So wins_a=0, p_value = 1.0 (cannot reject null that B >= A)
        assert result["p_value_a_gt_b"] >= 0.5
        assert result["significant_at_05"] is False
