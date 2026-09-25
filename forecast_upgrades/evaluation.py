"""Offline evaluation on resolved questions.

The key trick: log every MEMBER forecast, then replay alternative aggregation /
calibration / capping pipelines on resolved questions without any new LLM calls.
Compare pipelines with PAIRED per-question differences (same questions, same
members) and a bootstrap CI - never with two leaderboard totals.
"""
from __future__ import annotations

from typing import Callable, Hashable, Mapping, Sequence

import numpy as np

EPS = 1e-9


def log_score_binary(p: float, outcome: int) -> float:
    """Natural-log score of the realised outcome (higher is better, max 0)."""
    p = float(np.clip(p, EPS, 1 - EPS))
    return float(np.log(p if outcome == 1 else 1 - p))


def head_to_head(p_a: float, p_b: float, outcome: int) -> float:
    """Metaculus-style head-to-head points: 100 * (ln P_a(outcome) - ln P_b(outcome))."""
    return 100.0 * (log_score_binary(p_a, outcome) - log_score_binary(p_b, outcome))


def brier(p: float, outcome: int) -> float:
    return float((p - outcome) ** 2)


def log_score_mc(p: Sequence[float], outcome_idx: int) -> float:
    return float(np.log(max(float(p[outcome_idx]), EPS)))


def paired_bootstrap(
    deltas: Sequence[float],
    clusters: Sequence[Hashable] | None = None,
    n_boot: int = 5000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Mean paired difference with a (cluster) bootstrap CI.

    Pass clusters (e.g. question-group ids) when questions are correlated so the
    interval isn't falsely narrow.
    """
    d = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    if clusters is None:
        boots = rng.choice(d, size=(n_boot, d.size), replace=True).mean(axis=1)
    else:
        cl = np.asarray(clusters, dtype=object)
        keys = list(dict.fromkeys(cl.tolist()))
        groups = [d[cl == k] for k in keys]
        boots = np.empty(n_boot)
        for i in range(n_boot):
            pick = rng.integers(0, len(groups), len(groups))
            boots[i] = np.concatenate([groups[j] for j in pick]).mean()
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": int(d.size)}


def calibration_table(probs: Sequence[float], outcomes: Sequence[int], n_bins: int = 10) -> list[dict]:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.sum():
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "mean_forecast": float(p[m].mean()), "observed": float(y[m].mean())})
    return rows


def replay_binary(
    rows: Sequence[Mapping],
    pipelines: Mapping[str, Callable[[Mapping], float]],
    baseline: str,
    cluster_key: str | None = None,
) -> dict[str, dict]:
    """Score several post-processing pipelines on the same logged, resolved questions.

    rows: dicts with at least {"members": [p1, p2, ...], "outcome": 0/1} and any
          extra fields your pipelines need (segment, market price, spread...).
    pipelines: name -> function(row) -> final probability.
    Returns per-pipeline mean log score, Brier and the paired head-to-head vs baseline.
    """
    finals = {name: np.array([fn(r) for r in rows], dtype=float) for name, fn in pipelines.items()}
    ys = np.array([int(r["outcome"]) for r in rows])
    clusters = [r.get(cluster_key) for r in rows] if cluster_key else None
    out = {}
    for name, ps in finals.items():
        h2h = [head_to_head(pa, pb, y) for pa, pb, y in zip(ps, finals[baseline], ys)]
        out[name] = {
            "mean_log_score": float(np.mean([log_score_binary(p, y) for p, y in zip(ps, ys)])),
            "mean_brier": float(np.mean([brier(p, y) for p, y in zip(ps, ys)])),
            "h2h_vs_baseline": paired_bootstrap(h2h, clusters=clusters),
        }
    return out


def interval_coverage(lower: Sequence[float], upper: Sequence[float], outcomes: Sequence[float]) -> float:
    """Share of numeric outcomes inside [lower, upper]; compare with nominal (e.g. 0.8 for p10-p90)."""
    lo, hi, y = (np.asarray(a, dtype=float) for a in (lower, upper, outcomes))
    return float(((y >= lo) & (y <= hi)).mean())


def continuous_log_score_proxy(cdf: Sequence[float], unscaled_outcome: float, floor_pdf: float = 0.01) -> float:
    """Approximate log score for a CDF on a uniform unscaled grid (for A/B comparison only).

    In range: log of bin density (mixed with a uniform floor). Out of range: log of tail mass.
    """
    c = np.asarray(cdf, dtype=float)
    n_bins = c.size - 1
    if unscaled_outcome < 0:
        return float(np.log(max(c[0], EPS)))
    if unscaled_outcome > 1:
        return float(np.log(max(1 - c[-1], EPS)))
    i = min(int(unscaled_outcome * n_bins), n_bins - 1)
    density = (c[i + 1] - c[i]) * n_bins
    return float(np.log(0.99 * density + floor_pdf))
