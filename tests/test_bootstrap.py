"""Tests for bootstrap confidence intervals (requires scipy, sklearn, numpy)."""

import pytest

pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from src.eval.bootstrap_ci import bootstrap_ci, paired_bootstrap_test, wilson_score_interval
from src.eval.metrics import accuracy


class TestWilsonScoreInterval:
    def test_perfect_score_interval_below_one(self):
        # Wilson interval never claims 100% confidence even at 100% observed
        # accuracy — unlike a naive percentile bootstrap on a degenerate
        # sample, which can collapse to a point.
        result = wilson_score_interval(100, 100)
        assert result["point_estimate"] == 1.0
        assert result["ci_upper"] <= 1.0
        assert result["ci_lower"] < 1.0

    def test_half_score_interval_contains_half(self):
        result = wilson_score_interval(50, 100)
        assert result["ci_lower"] <= 0.5 <= result["ci_upper"]

    def test_small_n_interval_is_wide(self):
        # ENEM-sized benchmark (180 items): a 90% observed accuracy should
        # still have real width, not a falsely tight interval.
        small = wilson_score_interval(162, 180)
        large = wilson_score_interval(1620, 1800)
        small_width = small["ci_upper"] - small["ci_lower"]
        large_width = large["ci_upper"] - large["ci_lower"]
        assert small_width > large_width

    def test_zero_total(self):
        result = wilson_score_interval(0, 0)
        assert result["n_total"] == 0


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
        # A wins almost every resample and the 95% CI of the paired
        # difference should sit clear of zero given the 40pp gap.
        assert result["prob_a_gt_b"] > 0.95
        assert result["mean_diff_a_minus_b"] == pytest.approx(0.4, abs=0.05)
        assert result["ci_lower"] > 0
        assert result["significant"] is True

    def test_equal_models(self):
        gold = ["A"] * 100
        preds = ["A"] * 70 + ["B"] * 30
        result = paired_bootstrap_test(preds, preds, gold, accuracy, "accuracy", n_bootstrap=200)
        # Identical predictions: A is never STRICTLY better than B (scores
        # are always equal), and the paired difference is 0 in every
        # resample, so the CI is degenerate at 0 (not significant).
        assert result["prob_a_gt_b"] == pytest.approx(0.0)
        assert result["mean_diff_a_minus_b"] == pytest.approx(0.0)
        assert result["significant"] is False
