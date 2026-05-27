"""
src/eval/metrics.py
───────────────────
Evaluation metrics for PT-BR benchmarks.

Provides:
  • macro-F1       — classification tasks (HateBR, TweetSentBR, ASSIN2-RTE)
  • Pearson r      — semantic similarity (ASSIN2-STS)
  • Spearman rho   — semantic similarity rank correlation
  • Approval Rate  — exam-style (ENEM, BluEx, OAB)
  • Accuracy       — standard accuracy (COPA-PT, RTE-PT)
  • Refusal Rate   — safety evaluation (DoNotAnswer-PT)
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score


def compute_macro_f1(
    y_true: list[int | str],
    y_pred: list[int | str],
) -> float:
    """Compute macro-averaged F1 score.

    Parameters
    ----------
    y_true : list
        Ground truth labels.
    y_pred : list
        Predicted labels.

    Returns
    -------
    float
        Macro-F1 score in [0, 1].
    """
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def compute_accuracy(
    y_true: list[int | str],
    y_pred: list[int | str],
) -> float:
    """Compute standard accuracy.

    Parameters
    ----------
    y_true : list
        Ground truth labels.
    y_pred : list
        Predicted labels.

    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    return float(accuracy_score(y_true, y_pred))


def compute_pearson(
    y_true: list[float],
    y_pred: list[float],
) -> float:
    """Compute Pearson correlation coefficient.

    Parameters
    ----------
    y_true : list[float]
        Ground truth scores.
    y_pred : list[float]
        Predicted scores.

    Returns
    -------
    float
        Pearson r in [-1, 1].
    """
    if len(y_true) < 2:
        return 0.0
    r, _ = stats.pearsonr(y_true, y_pred)
    return float(r)


def compute_spearman(
    y_true: list[float],
    y_pred: list[float],
) -> float:
    """Compute Spearman rank correlation coefficient.

    Parameters
    ----------
    y_true : list[float]
        Ground truth scores.
    y_pred : list[float]
        Predicted scores.

    Returns
    -------
    float
        Spearman rho in [-1, 1].
    """
    if len(y_true) < 2:
        return 0.0
    rho, _ = stats.spearmanr(y_true, y_pred)
    return float(rho)


def compute_approval_rate(
    correct: list[bool],
) -> float:
    """Compute approval rate (percentage of correct answers).

    Used for exam-style benchmarks (ENEM, BluEx, OAB).

    Parameters
    ----------
    correct : list[bool]
        Boolean list indicating correctness of each answer.

    Returns
    -------
    float
        Approval rate in [0, 1].
    """
    if not correct:
        return 0.0
    return float(np.mean(correct))


def compute_refusal_rate(
    refused: list[bool],
) -> float:
    """Compute refusal rate for safety benchmarks.

    Parameters
    ----------
    refused : list[bool]
        Boolean list indicating whether the model refused each prompt.

    Returns
    -------
    float
        Refusal rate in [0, 1].
    """
    if not refused:
        return 0.0
    return float(np.mean(refused))


# ──────────────────────────────────────────────────────────────────────
# Metric registry
# ──────────────────────────────────────────────────────────────────────

METRIC_FUNCTIONS = {
    "macro_f1": compute_macro_f1,
    "accuracy": compute_accuracy,
    "pearson": compute_pearson,
    "spearman": compute_spearman,
    "approval_rate": compute_approval_rate,
    "refusal_rate": compute_refusal_rate,
}


def get_metric_fn(name: str):
    """Retrieve a metric function by name.

    Parameters
    ----------
    name : str
        Metric name (e.g. ``"macro_f1"``).

    Returns
    -------
    callable
        The metric function.

    Raises
    ------
    KeyError
        If the metric name is not registered.
    """
    if name not in METRIC_FUNCTIONS:
        raise KeyError(
            f"Unknown metric '{name}'. Available: {list(METRIC_FUNCTIONS.keys())}"
        )
    return METRIC_FUNCTIONS[name]
