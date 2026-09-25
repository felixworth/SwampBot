"""Aggregation of member forecasts.

Binary: mean, median, trimmed mean, log-odds mean (= geometric mean of odds),
optionally extremized. Multiple choice: per-option mean/median/log-mean, renormalized.
Numeric: mixture of CDFs on a shared grid (preserves disagreement instead of
averaging medians). Also a log-odds blend with an external anchor (e.g. a
matched, liquid prediction market).

Everything here is pure numpy so it can be replayed offline on logged forecasts.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

EPS = 1e-6


def logit(p: float | np.ndarray) -> float | np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def aggregate_binary(
    probs: Sequence[float],
    method: str = "median",
    trim: float = 0.2,
    extremize: float = 1.0,
    weights: Sequence[float] | None = None,
) -> float:
    """Combine member probabilities for a binary question.

    method: "mean" | "median" | "trimmed_mean" | "logodds_mean"
    extremize: multiplier d on the mean log-odds (only for "logodds_mean");
        d > 1 pushes away from 50% (Baron et al. 2014 style). Keep at 1.0 until
        your own resolved data justify something else.
    """
    p = np.asarray(probs, dtype=float)
    if p.size == 0:
        raise ValueError("no member forecasts")
    if np.any((p < 0) | (p > 1)) or np.any(np.isnan(p)):
        raise ValueError(f"invalid probabilities: {p}")
    w = None if weights is None else np.asarray(weights, dtype=float)
    if method == "mean":
        return float(np.average(p, weights=w))
    if method == "median":
        return float(np.median(p))
    if method == "trimmed_mean":
        k = int(np.floor(trim * p.size))
        s = np.sort(p)
        s = s[k : p.size - k] if p.size - 2 * k > 0 else s
        return float(s.mean())
    if method == "logodds_mean":
        return float(sigmoid(extremize * np.average(logit(p), weights=w)))
    raise ValueError(f"unknown method {method}")


def aggregate_mc(
    prob_vectors: Sequence[Sequence[float]],
    method: str = "mean",
    floor: float = 1e-3,
) -> list[float]:
    """Combine multiple-choice probability vectors (one per member).

    Each vector is normalized first; the output is floored and renormalized so no
    option ever gets ~0 (a single unexpected option would otherwise be catastrophic).
    """
    m = np.asarray(prob_vectors, dtype=float)
    if m.ndim != 2:
        raise ValueError("expected a 2-D array: members x options")
    m = np.clip(m, 0, None)
    m = m / m.sum(axis=1, keepdims=True)
    if method == "mean":
        agg = m.mean(axis=0)
    elif method == "median":
        agg = np.median(m, axis=0)
    elif method == "logodds_mean":
        agg = np.exp(np.log(np.clip(m, EPS, 1)).mean(axis=0))
    else:
        raise ValueError(f"unknown method {method}")
    agg = agg / agg.sum()
    return normalize_with_floor(agg, floor)


def normalize_with_floor(p: Iterable[float], floor: float) -> list[float]:
    """Renormalize so every entry >= floor and the vector sums to 1."""
    p = np.asarray(list(p), dtype=float)
    n = p.size
    if floor * n >= 1:
        raise ValueError("floor too large for number of options")
    p = p / p.sum()
    # mix with uniform just enough to lift the smallest entry to the floor
    lo = p.min()
    if lo < floor:
        lam = (floor - lo) / (1.0 / n - lo)
        p = (1 - lam) * p + lam / n
    return (p / p.sum()).tolist()


def mix_cdfs(cdfs: Sequence[Sequence[float]], weights: Sequence[float] | None = None) -> np.ndarray:
    """Mixture of distributions evaluated on a shared grid (CDF values)."""
    c = np.asarray(cdfs, dtype=float)
    if c.ndim != 2:
        raise ValueError("expected members x grid")
    w = np.ones(c.shape[0]) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    return (w[:, None] * c).sum(axis=0)


def blend_logodds(p_model: float, p_anchor: float, w_anchor: float) -> float:
    """Blend two probabilities in log-odds space. w_anchor in [0, 1].

    Use for a well-matched, liquid market price; learn w_anchor offline on
    resolved questions where a market existed.
    """
    if not 0 <= w_anchor <= 1:
        raise ValueError("w_anchor must be in [0, 1]")
    return float(sigmoid((1 - w_anchor) * logit(p_model) + w_anchor * logit(p_anchor)))


def spread(probs: Sequence[float]) -> float:
    """Max-min range of member probabilities; the simplest disagreement trigger."""
    p = np.asarray(probs, dtype=float)
    return float(p.max() - p.min())


def logodds_spread(probs: Sequence[float]) -> float:
    """Disagreement in log-odds; more sensitive near 0/1 than raw range."""
    lo = logit(np.asarray(probs, dtype=float))
    return float(np.max(lo) - np.min(lo))
