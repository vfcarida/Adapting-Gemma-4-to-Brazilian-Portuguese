"""
src/eval/bootstrap_ci.py
────────────────────────
Bootstrap confidence intervals and paired bootstrap tests for
statistical significance in model comparisons.

Provides:
  • bootstrap_confidence_interval — 95% CI for a metric
  • paired_bootstrap_test — p-value for paired difference
"""

from __future__ import annotations

from typing import Any

import numpy as np


def bootstrap_confidence_interval(
    scores: list[float] | np.ndarray,
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    statistic: str = "mean",
    seed: int = 42,
) -> dict[str, float]:
    """Compute bootstrap confidence interval for a set of scores.

    Parameters
    ----------
    scores : array-like
        Individual sample scores (e.g. per-example accuracy).
    n_bootstrap : int
        Number of bootstrap resamples.
    confidence_level : float
        Confidence level (e.g. 0.95 for 95% CI).
    statistic : str
        Statistic to compute: ``"mean"`` or ``"median"``.
    seed : int
        Random seed.

    Returns
    -------
    dict[str, float]
        ``{"estimate": ..., "ci_lower": ..., "ci_upper": ..., "ci_width": ...}``
    """
    scores = np.asarray(scores, dtype=np.float64)
    rng = np.random.RandomState(seed)

    stat_fn = np.mean if statistic == "mean" else np.median
    point_estimate = float(stat_fn(scores))

    # Bootstrap resampling
    boot_stats = np.empty(n_bootstrap)
    n = len(scores)
    for i in range(n_bootstrap):
        sample = scores[rng.randint(0, n, size=n)]
        boot_stats[i] = stat_fn(sample)

    # Percentile method
    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(boot_stats, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return {
        "estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_width": ci_upper - ci_lower,
        "n_bootstrap": n_bootstrap,
        "confidence_level": confidence_level,
    }


def paired_bootstrap_test(
    scores_a: list[float] | np.ndarray,
    scores_b: list[float] | np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap test for statistical significance.

    Tests whether model A is significantly different from model B
    by resampling their per-example score differences.

    Parameters
    ----------
    scores_a : array-like
        Per-example scores for model A.
    scores_b : array-like
        Per-example scores for model B (same examples, same order).
    n_bootstrap : int
        Number of bootstrap resamples.
    seed : int
        Random seed.

    Returns
    -------
    dict[str, float]
        ``{"mean_diff": ..., "p_value": ..., "ci_lower": ..., "ci_upper": ...,
           "significant_005": ..., "significant_001": ..., "significant_0001": ...}``
    """
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)

    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"Score arrays must have the same length: {len(scores_a)} vs {len(scores_b)}"
        )

    # Observed difference
    diffs = scores_a - scores_b
    observed_diff = float(np.mean(diffs))

    # Bootstrap null distribution (centered differences)
    rng = np.random.RandomState(seed)
    n = len(diffs)
    centered_diffs = diffs - observed_diff  # center under null

    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = centered_diffs[rng.randint(0, n, size=n)]
        boot_diffs[i] = np.mean(sample)

    # Two-tailed p-value
    p_value = float(np.mean(np.abs(boot_diffs) >= np.abs(observed_diff)))

    # CI on the difference
    ci_lower = float(np.percentile(diffs[rng.randint(0, n, size=(n_bootstrap, n))].mean(axis=1), 2.5))
    ci_upper = float(np.percentile(diffs[rng.randint(0, n, size=(n_bootstrap, n))].mean(axis=1), 97.5))

    return {
        "mean_diff": observed_diff,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
        "significant_0001": p_value < 0.001,
        "n_samples": n,
        "n_bootstrap": n_bootstrap,
    }
