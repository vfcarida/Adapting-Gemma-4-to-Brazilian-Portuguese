"""Bootstrap confidence intervals for evaluation metrics."""

from typing import Any, Callable

import numpy as np


def bootstrap_ci(
    predictions: list[Any],
    gold_labels: list[Any],
    metric_fn: Callable[[list, list], dict[str, float]],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Compute bootstrap confidence intervals for a metric.

    Returns:
        Dict mapping metric names to {mean, ci_lower, ci_upper, std}
    """
    rng = np.random.default_rng(seed)
    n = len(predictions)

    # Collect bootstrap samples
    bootstrap_metrics: dict[str, list[float]] = {}

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        boot_preds = [predictions[i] for i in indices]
        boot_gold = [gold_labels[i] for i in indices]

        metrics = metric_fn(boot_preds, boot_gold)
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                bootstrap_metrics.setdefault(key, []).append(value)

    # Compute CIs
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
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap test comparing two models.

    Returns p-value for the hypothesis that model A > model B.
    """
    rng = np.random.default_rng(seed)
    n = len(gold_labels)

    wins_a = 0

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        boot_gold = [gold_labels[i] for i in indices]
        boot_preds_a = [predictions_a[i] for i in indices]
        boot_preds_b = [predictions_b[i] for i in indices]

        score_a = metric_fn(boot_preds_a, boot_gold).get(metric_key, 0)
        score_b = metric_fn(boot_preds_b, boot_gold).get(metric_key, 0)

        if score_a > score_b:
            wins_a += 1

    p_value = 1.0 - (wins_a / n_bootstrap)

    return {
        "p_value_a_gt_b": float(p_value),
        "wins_a": wins_a,
        "n_bootstrap": n_bootstrap,
        "significant_at_05": p_value < 0.05,
    }
