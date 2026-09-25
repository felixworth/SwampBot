"""Put the Bayesian arithmetic in code; let the LLM supply the inputs.

The LLM's job becomes: pick a reference class, count events, estimate a bounded
adjustment, list scenarios. The code's job: turn those into probabilities that
scale correctly with time remaining and stay coherent across related questions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def hazard_prob_at_least_one(
    k_events: float,
    exposure_years: float,
    horizon_years: float,
    rate_multiplier: float = 1.0,
    prior_events: float = 0.5,
    prior_exposure_years: float = 0.0,
    multiplier_bounds: tuple[float, float] = (1 / 3, 3.0),
) -> float:
    """P(at least one event before the deadline) under a Gamma-Poisson model.

    Reference class: k_events observed over exposure_years. With a Gamma prior
    (prior_events, prior_exposure_years) the posterior rate is
    Gamma(k + prior_events, exposure + prior_exposure), and the predictive
    probability of zero events in horizon T with rate multiplier m is
        (beta / (beta + m*T)) ** alpha.
    The half-event prior keeps zero-count reference classes above 0%.

    rate_multiplier is the LLM's inside-view adjustment; it is clamped to
    multiplier_bounds so one enthusiastic rationale cannot move the rate 20x.
    """
    if exposure_years + prior_exposure_years <= 0:
        raise ValueError("need positive exposure")
    if horizon_years < 0 or k_events < 0:
        raise ValueError("negative inputs")
    m = float(np.clip(rate_multiplier, *multiplier_bounds))
    alpha = k_events + prior_events
    beta = exposure_years + prior_exposure_years
    p_none = (beta / (beta + m * horizon_years)) ** alpha
    return float(1.0 - p_none)


def prob_exceeds(grid: Sequence[float], cdf: Sequence[float], threshold: float) -> float:
    """P(X > threshold) from CDF values on an increasing grid (linear interpolation).

    Below the grid: returns 1 - cdf[0]; above: 1 - cdf[-1] (the open-bound tail mass).
    Use this to answer "Will X exceed T?" from ONE distribution of X, so related
    thresholds are automatically consistent.
    """
    g = np.asarray(grid, dtype=float)
    c = np.asarray(cdf, dtype=float)
    if threshold <= g[0]:
        return float(1 - c[0])
    if threshold >= g[-1]:
        return float(1 - c[-1])
    return float(1 - np.interp(threshold, g, c))


def _pava_increasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators: closest non-decreasing sequence."""
    blocks = [[float(v), float(wt), 1] for v, wt in zip(y, w)]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0] + 1e-12:
            v1, w1, n1 = blocks[i]
            v2, w2, n2 = blocks[i + 1]
            blocks[i] = [(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for v, _, n in blocks:
        out.extend([v] * n)
    return np.asarray(out)


def enforce_monotone_deadlines(probs_by_deadline: Sequence[float], weights: Sequence[float] | None = None) -> list[float]:
    """P(event by d1) <= P(event by d2) for d1 < d2 (inputs sorted by deadline)."""
    y = np.asarray(probs_by_deadline, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    return _pava_increasing(y, w).tolist()


def enforce_monotone_thresholds(probs_exceed_by_threshold: Sequence[float], weights: Sequence[float] | None = None) -> list[float]:
    """P(X > t1) >= P(X > t2) for t1 < t2 (inputs sorted by threshold)."""
    y = np.asarray(probs_exceed_by_threshold, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    return (-_pava_increasing(-y, w)).tolist()


def negation_gap(p_question: float, p_negated_question: float) -> float:
    """|P(A) + P(not A) - 1|. A cheap incoherence signal to trigger review."""
    return abs(p_question + p_negated_question - 1.0)


@dataclass
class Scenario:
    name: str
    p_scenario: float
    p_event_given_scenario: float


def scenario_tree(scenarios: Sequence[Scenario], tol: float = 0.02) -> float:
    """Law of total probability over a short, mutually exclusive scenario list.

    Scenario weights are renormalised if they sum within tol of 1; a larger
    mismatch means the LLM's scenarios are not exhaustive/exclusive -> error.
    """
    total = sum(s.p_scenario for s in scenarios)
    if abs(total - 1) > tol:
        raise ValueError(f"scenario probabilities sum to {total:.3f}; not exhaustive/exclusive")
    return float(sum(s.p_scenario / total * s.p_event_given_scenario for s in scenarios))
