"""Confidence intervals and paired comparisons for evaluation metrics.

Method choice, briefly (see docs/EVAL_PROTOCOL.md and
docs/CPT_BEST_PRACTICES_RESEARCH.md for the full literature discussion):

- For a single model's accuracy-like metric on a SMALL benchmark (a few
  hundred items or fewer — true for every Portuguese exam benchmark here:
  ENEM is 180 items/year, BLUEX ~724, OAB ~2210), percentile/BCa bootstrap
  and the normal (CLT) approximation both under-cover the nominal confidence
  level (Bowyer, Aitchison & Ivanova, "Don't Use the CLT in LLM Evals With
  Fewer Than a Few Hundred Datapoints", ICML 2025 Spotlight,
  arXiv:2503.01747). Use `wilson_score_interval` instead — it holds ~95%
  coverage at all N and requires no resampling.
- For metrics that aren't a simple mean of a per-item 0/1 score (Pearson
  correlation for ASSIN2-STS, macro-F1 for HateBR/TweetSentBR, entity-F1 for
  LeNER-Br), there's no simple closed form, so `bootstrap_ci` resamples
  ITEMS (not bootstrap draws of an already-computed statistic) and
  recomputes the full metric function each time — the standard
  paired-item bootstrap for NLP metrics (Efron & Tibshirani; Dror et al.,
  "The Hitchhiker's Guide to Testing Statistical Significance in NLP", ACL
  2018). Interval endpoints need many resamples to be stable — this module
  defaults to 10,000, matching what this project's docs have long claimed
  (previously the code silently used only 1,000).
- Comparing two models on the SAME items should always be paired (resample
  the same indices for both models each draw) — `paired_bootstrap_test`
  does this and reports a difference + CI, not a single point estimate.
"""

import math
from typing import Any, Callable

import numpy as np


def wilson_score_interval(
    n_correct: int, n_total: int, confidence_level: float = 0.95
) -> dict[str, float]:
    """Wilson score confidence interval for a single proportion (accuracy).

    Closed-form, no resampling, and — unlike the normal/CLT approximation or
    naive bootstrap — holds close to nominal coverage even at small N (tens
    to low hundreds of items), per Bowyer et al. 2025 (arXiv:2503.01747).
    This is the recommended interval for a single model's accuracy on
    exam-style benchmarks (ENEM, BLUEX, OAB, ...), which are all far smaller
    than the "few hundred" threshold where CLT/bootstrap become reliable.

    Args:
        n_correct: Number of correct predictions.
        n_total: Total number of items scored.
        confidence_level: Confidence level (default 0.95).

    Returns:
        Dict with `point_estimate`, `ci_lower`, `ci_upper`, `n_total`.
    """
    if n_total == 0:
        return {"point_estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n_total": 0}

    from scipy import stats as _stats

    z = _stats.norm.ppf(1 - (1 - confidence_level) / 2)
    p_hat = n_correct / n_total
    denom = 1 + z**2 / n_total
    center = p_hat + z**2 / (2 * n_total)
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n_total + z**2 / (4 * n_total**2))
    return {
        "point_estimate": float(p_hat),
        "ci_lower": float(max(0.0, (center - margin) / denom)),
        "ci_upper": float(min(1.0, (center + margin) / denom)),
        "n_total": n_total,
    }


def bootstrap_ci(
    predictions: list[Any],
    gold_labels: list[Any],
    metric_fn: Callable[[list, list], dict[str, float]],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute item-resampling bootstrap confidence intervals for a metric.

    Resamples (prediction, gold) pairs WITH replacement and recomputes the
    full metric on each resample — the correct approach for metrics that
    aren't a plain mean of a per-item score (Pearson, F1, ...). For plain
    accuracy on a small benchmark, prefer `wilson_score_interval` on the
    ORIGINAL (non-resampled) correct/total counts instead — see module
    docstring.

    Returns:
        Dict mapping metric names to {mean, ci_lower, ci_upper, std,
        n_bootstrap, confidence_level}.
    """
    rng = np.random.default_rng(seed)
    n = len(predictions)

    bootstrap_metrics: dict[str, list[float]] = {}

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        boot_preds = [predictions[i] for i in indices]
        boot_gold = [gold_labels[i] for i in indices]

        metrics = metric_fn(boot_preds, boot_gold)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                bootstrap_metrics.setdefault(key, []).append(value)

    alpha = (1 - confidence_level) / 2
    results = {}

    for key, values in bootstrap_metrics.items():
        values_arr = np.array(values)
        results[key] = {
            "mean": float(np.mean(values_arr)),
            "std": float(np.std(values_arr)),
            "ci_lower": float(np.percentile(values_arr, alpha * 100)),
            "ci_upper": float(np.percentile(values_arr, (1 - alpha) * 100)),
            "n_bootstrap": n_bootstrap,
            "confidence_level": confidence_level,
        }

    return results


def paired_bootstrap_test(
    predictions_a: list[Any],
    predictions_b: list[Any],
    gold_labels: list[Any],
    metric_fn: Callable[[list, list], dict[str, float]],
    metric_key: str,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap comparison of two models scored on the SAME items.

    Each resample draws one set of item indices and applies it to BOTH
    models (paired resampling — correct: the same benchmark items are
    compared under the same resampled sample every draw, following Dror et
    al. 2018 / Koehn 2004's paired bootstrap for NLP significance testing).

    Reports the difference (A - B) and its bootstrap CI, which is the valid
    way to read significance here: "significant" means the CI excludes zero.
    (The previous version of this function returned
    `p_value = 1 - wins_a/n_bootstrap`, which is a resampled win-probability,
    not a p-value for any null hypothesis — it could not distinguish a
    trivial win margin from a large one, and always returns ~0 once A wins
    almost every resample. `prob_a_gt_b` below is that same quantity, kept
    as a secondary "probability of superiority" statistic, but the
    difference-CI is now the primary significance criterion.)

    Returns:
        Dict with `mean_diff_a_minus_b`, `ci_lower`, `ci_upper`,
        `prob_a_gt_b`, `significant` (CI excludes 0), `n_bootstrap`.
    """
    rng = np.random.default_rng(seed)
    n = len(gold_labels)

    point_a = metric_fn(predictions_a, gold_labels).get(metric_key, 0.0)
    point_b = metric_fn(predictions_b, gold_labels).get(metric_key, 0.0)

    diffs = np.empty(n_bootstrap)
    wins_a = 0

    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        boot_gold = [gold_labels[j] for j in indices]
        boot_preds_a = [predictions_a[j] for j in indices]
        boot_preds_b = [predictions_b[j] for j in indices]

        score_a = metric_fn(boot_preds_a, boot_gold).get(metric_key, 0.0)
        score_b = metric_fn(boot_preds_b, boot_gold).get(metric_key, 0.0)
        diffs[i] = score_a - score_b
        if score_a > score_b:
            wins_a += 1

    alpha = (1 - confidence_level) / 2
    ci_lower = float(np.percentile(diffs, alpha * 100))
    ci_upper = float(np.percentile(diffs, (1 - alpha) * 100))

    return {
        "point_estimate_a": float(point_a),
        "point_estimate_b": float(point_b),
        "mean_diff_a_minus_b": float(np.mean(diffs)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "prob_a_gt_b": float(wins_a / n_bootstrap),
        "significant": bool(ci_lower > 0 or ci_upper < 0),
        "n_bootstrap": n_bootstrap,
        "confidence_level": confidence_level,
    }
