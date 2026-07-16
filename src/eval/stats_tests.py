"""Testes estatísticos para avaliação comparativa de modelos.

Este módulo fornece funções de teste estatístico para determinar se
diferenças entre modelos são significativas, incluindo testes de
permutação, McNemar, Wilcoxon e correção para comparações múltiplas.

Utiliza scipy quando disponível; caso contrário, sinaliza graciosamente.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

try:
    from scipy import stats as scipy_stats
    from scipy.stats import wilcoxon as _scipy_wilcoxon

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def paired_permutation_test(
    scores_a: list | np.ndarray,
    scores_b: list | np.ndarray,
    n_permutations: int = 10000,
) -> dict[str, float]:
    """Teste de permutação pareado para diferença de médias.

    Avalia se a diferença observada entre dois conjuntos de scores
    é estatisticamente significativa via reamostragem sem suposições
    de distribuição.

    Args:
        scores_a: Scores do modelo A (um por exemplo).
        scores_b: Scores do modelo B (um por exemplo).
        n_permutations: Número de permutações para estimar a distribuição nula.

    Returns:
        Dict com p_value, mean_diff, ci_lower, ci_upper (IC 95% da diferença).

    Raises:
        ValueError: Se os arrays tiverem tamanhos diferentes ou menos de 2 amostras.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    if len(a) != len(b):
        raise ValueError(
            f"Arrays devem ter o mesmo tamanho: len(a)={len(a)}, len(b)={len(b)}"
        )

    if len(a) < 2:
        raise ValueError("Necessário pelo menos 2 amostras para o teste de permutação.")

    # Caso trivial: arrays idênticos
    if np.array_equal(a, b):
        return {
            "p_value": 1.0,
            "mean_diff": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
        }

    diffs = a - b
    observed_diff = float(np.mean(diffs))

    # Permutação: aleatoriza sinais das diferenças
    rng = np.random.default_rng(seed=42)
    count_extreme = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_diff = np.mean(diffs * signs)
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)

    # Intervalo de confiança bootstrap 95% da diferença
    bootstrap_diffs = []
    for _ in range(n_permutations):
        idx = rng.integers(0, len(diffs), size=len(diffs))
        bootstrap_diffs.append(float(np.mean(diffs[idx])))

    ci_lower = float(np.percentile(bootstrap_diffs, 2.5))
    ci_upper = float(np.percentile(bootstrap_diffs, 97.5))

    return {
        "p_value": float(p_value),
        "mean_diff": observed_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def mcnemar_test(
    correct_a: list | np.ndarray,
    correct_b: list | np.ndarray,
) -> dict[str, float]:
    """Teste de McNemar para dados binários pareados (correto/incorreto).

    Compara se dois modelos diferem significativamente em termos de
    acerto/erro em cada exemplo. Útil para tarefas de classificação.

    Args:
        correct_a: Array binário (1=correto, 0=incorreto) para modelo A.
        correct_b: Array binário (1=correto, 0=incorreto) para modelo B.

    Returns:
        Dict com statistic (chi-quadrado) e p_value.

    Raises:
        ValueError: Se arrays tiverem tamanhos diferentes.
    """
    a = np.asarray(correct_a, dtype=int)
    b = np.asarray(correct_b, dtype=int)

    if len(a) != len(b):
        raise ValueError(
            f"Arrays devem ter o mesmo tamanho: len(a)={len(a)}, len(b)={len(b)}"
        )

    if len(a) == 0:
        return {"statistic": 0.0, "p_value": 1.0}

    # Tabela de contingência 2x2
    # b=1 e a=0 (B certo, A errado)
    b_correct_a_wrong = int(np.sum((b == 1) & (a == 0)))
    # b=0 e a=1 (A certo, B errado)
    a_correct_b_wrong = int(np.sum((a == 1) & (b == 0)))

    # Caso trivial: sem discordância
    n_discordant = b_correct_a_wrong + a_correct_b_wrong
    if n_discordant == 0:
        return {"statistic": 0.0, "p_value": 1.0}

    # Correção de continuidade de Edwards
    statistic = (abs(b_correct_a_wrong - a_correct_b_wrong) - 1) ** 2 / n_discordant

    # p-valor via distribuição chi-quadrado com 1 grau de liberdade
    if SCIPY_AVAILABLE:
        p_value = float(scipy_stats.chi2.sf(statistic, df=1))
    else:
        # Aproximação simples sem scipy
        # Para estatística chi2 com 1 gl, p ~= exp(-statistic/2) como aproximação grosseira
        p_value = float(np.exp(-statistic / 2))
        warnings.warn(
            "scipy não disponível; p-valor é aproximado. Instale scipy para resultados exatos.",
            stacklevel=2,
        )

    return {"statistic": float(statistic), "p_value": p_value}


def wilcoxon_signed_rank(
    scores_a: list | np.ndarray,
    scores_b: list | np.ndarray,
) -> dict[str, float]:
    """Teste de postos sinalizados de Wilcoxon.

    Teste não-paramétrico para diferenças pareadas. Alternativa ao
    t-test pareado quando não se pode assumir normalidade.

    Args:
        scores_a: Scores do modelo A.
        scores_b: Scores do modelo B.

    Returns:
        Dict com statistic e p_value.

    Raises:
        ValueError: Se arrays tiverem tamanhos diferentes ou menos de 6 amostras.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    if len(a) != len(b):
        raise ValueError(
            f"Arrays devem ter o mesmo tamanho: len(a)={len(a)}, len(b)={len(b)}"
        )

    # Arrays idênticos
    if np.array_equal(a, b):
        return {"statistic": 0.0, "p_value": 1.0}

    # Wilcoxon precisa de diferenças não-nulas
    diffs = a - b
    non_zero = diffs[diffs != 0]

    if len(non_zero) < 6:
        warnings.warn(
            f"Apenas {len(non_zero)} diferenças não-nulas; mínimo recomendado é 6. "
            "Resultado pode não ser confiável.",
            stacklevel=2,
        )
        if len(non_zero) == 0:
            return {"statistic": 0.0, "p_value": 1.0}

    if not SCIPY_AVAILABLE:
        warnings.warn(
            "scipy não disponível; teste de Wilcoxon não pode ser calculado.",
            stacklevel=2,
        )
        return {"statistic": float("nan"), "p_value": float("nan")}

    stat, p_value = _scipy_wilcoxon(a, b, zero_method="wilcox")
    return {"statistic": float(stat), "p_value": float(p_value)}


def compute_effect_size(
    scores_a: list | np.ndarray,
    scores_b: list | np.ndarray,
) -> dict[str, Any]:
    """Calcula o tamanho de efeito (d de Cohen) entre dois conjuntos de scores.

    Interpretação convencional (Cohen, 1988):
        - |d| < 0.2: negligível
        - 0.2 <= |d| < 0.5: pequeno
        - 0.5 <= |d| < 0.8: médio
        - |d| >= 0.8: grande

    Args:
        scores_a: Scores do modelo A.
        scores_b: Scores do modelo B.

    Returns:
        Dict com cohens_d (valor numérico) e interpretation (string).
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)

    if len(a) < 2 or len(b) < 2:
        return {"cohens_d": 0.0, "interpretation": "insuficiente (n < 2)"}

    # Desvio padrão pooled
    n_a, n_b = len(a), len(b)
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)

    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    if pooled_std == 0:
        # Sem variabilidade
        if np.mean(a) == np.mean(b):
            return {"cohens_d": 0.0, "interpretation": "negligível"}
        else:
            return {"cohens_d": float("inf"), "interpretation": "indeterminado (std=0)"}

    d = float((np.mean(a) - np.mean(b)) / pooled_std)

    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligível"
    elif abs_d < 0.5:
        interpretation = "pequeno"
    elif abs_d < 0.8:
        interpretation = "médio"
    else:
        interpretation = "grande"

    return {"cohens_d": d, "interpretation": interpretation}


def multiple_comparison_correction(
    p_values: list | np.ndarray,
    method: str = "holm",
) -> np.ndarray:
    """Correção para comparações múltiplas (controle de FWER).

    Métodos disponíveis:
        - "holm": Holm-Bonferroni (step-down, menos conservador que Bonferroni)
        - "bonferroni": Bonferroni (multiplicação simples pelo nº de testes)

    Args:
        p_values: Lista de p-valores originais.
        method: Método de correção ("holm" ou "bonferroni").

    Returns:
        Array de p-valores corrigidos (mesmo tamanho que entrada).

    Raises:
        ValueError: Se método não é reconhecido.
    """
    pvals = np.asarray(p_values, dtype=float)

    if len(pvals) == 0:
        return np.array([])

    if len(pvals) == 1:
        return pvals.copy()

    n = len(pvals)

    if method == "bonferroni":
        corrected = np.minimum(pvals * n, 1.0)

    elif method == "holm":
        # Holm step-down: ordena, multiplica por (n - rank), garante monotonicidade
        order = np.argsort(pvals)
        corrected = np.empty(n)
        cummax = 0.0
        for i, idx in enumerate(order):
            adjusted = pvals[idx] * (n - i)
            cummax = max(cummax, adjusted)
            corrected[idx] = min(cummax, 1.0)

    else:
        raise ValueError(
            f"Método desconhecido: '{method}'. Use 'holm' ou 'bonferroni'."
        )

    return corrected
