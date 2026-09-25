"""Post-hoc calibration fitted on YOUR bot's resolved forecasts.

Platt scaling: p' = sigmoid(a * logit(p) + b)
  a > 1  -> undo hedging towards 50%
  b < 0  -> correct a lean towards "Yes"
The fit is L2-penalised towards the identity (a=1, b=0), so with few resolved
questions it barely moves your forecasts. Choose the penalty by cross-validation.

Caps are applied LAST: they encode the probability that the whole pipeline is
wrong (misread criteria, stale news, resolver error), which calibration on
past data cannot see.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Sequence

import numpy as np
from scipy.optimize import minimize

from .aggregation import logit, sigmoid

EPS = 1e-9


@dataclass
class PlattParams:
    a: float = 1.0
    b: float = 0.0
    n: int = 0  # number of resolved forecasts used

    def apply(self, p: float | np.ndarray) -> float | np.ndarray:
        return sigmoid(self.a * logit(p) + self.b)


def _nll(params, x, y, w, lam, prior):
    a, b = params
    z = a * x + b
    # numerically stable log-loss
    ll = y * -np.logaddexp(0, -z) + (1 - y) * -np.logaddexp(0, z)
    return -(w * ll).sum() + lam * ((a - prior[0]) ** 2 + (b - prior[1]) ** 2)


def fit_platt(
    probs: Sequence[float],
    outcomes: Sequence[int],
    lam: float = 3.0,
    prior: tuple[float, float] = (1.0, 0.0),
    weights: Sequence[float] | None = None,
    a_bounds: tuple[float, float] = (0.5, 2.5),
    b_bounds: tuple[float, float] = (-1.5, 1.5),
) -> PlattParams:
    """Fit (a, b) by penalised maximum likelihood. outcomes are 0/1."""
    x = logit(np.asarray(probs, dtype=float))
    y = np.asarray(outcomes, dtype=float)
    if x.shape != y.shape:
        raise ValueError("probs and outcomes must have the same length")
    if not set(np.unique(y)).issubset({0.0, 1.0}):
        raise ValueError("outcomes must be 0/1")
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    res = minimize(
        _nll,
        x0=np.array(prior, dtype=float),
        args=(x, y, w, lam, prior),
        method="L-BFGS-B",
        bounds=[a_bounds, b_bounds],
    )
    a, b = res.x
    return PlattParams(a=float(a), b=float(b), n=int(x.size))


def log_loss(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    p = np.clip(np.asarray(probs, dtype=float), EPS, 1 - EPS)
    y = np.asarray(outcomes, dtype=float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def fit_platt_cv(
    probs: Sequence[float],
    outcomes: Sequence[int],
    lams: Sequence[float] = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0),
    k: int = 5,
    seed: int = 0,
) -> tuple[PlattParams, dict[float, float]]:
    """Choose the shrinkage strength by K-fold CV log loss, then refit on all data.

    Returns (params, {lam: cv_log_loss}). Compare against the identity's log loss
    before trusting the fit: if CV loss is not better, keep a=1, b=0.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(p.size)
    folds = np.array_split(idx, k)
    scores: dict[float, float] = {}
    for lam in lams:
        losses = []
        for f in folds:
            train = np.setdiff1d(idx, f)
            if len(np.unique(y[train])) < 2 or f.size == 0:
                continue
            prm = fit_platt(p[train], y[train], lam=lam)
            losses.append(log_loss(prm.apply(p[f]), y[f]) * f.size)
        scores[lam] = float(np.sum(losses) / p.size) if losses else float("inf")
    best = min(scores, key=scores.get)
    return fit_platt(p, y, lam=best), scores


def cap(p: float | np.ndarray, lo: float = 0.02, hi: float = 0.98) -> float | np.ndarray:
    """Clip (not shrink) so well-justified confident forecasts are not dragged to 50%."""
    out = np.clip(p, lo, hi)
    return float(out) if np.ndim(out) == 0 else out


@dataclass
class SegmentedCalibrator:
    """Per-segment Platt scaling that falls back to (and shrinks towards) a global fit.

    Segment keys are whatever you log, e.g. ("binary", "before-deadline") or
    evidence richness buckets. Segments with fewer than min_count resolved
    forecasts just use the global parameters.
    """

    min_count: int = 60
    lam_global: float = 3.0
    lam_segment: float = 10.0
    global_params: PlattParams = field(default_factory=PlattParams)
    segment_params: dict[Hashable, PlattParams] = field(default_factory=dict)

    def fit(self, probs, outcomes, segments) -> "SegmentedCalibrator":
        p = np.asarray(probs, dtype=float)
        y = np.asarray(outcomes, dtype=int)
        s = np.asarray(segments, dtype=object)
        self.global_params = fit_platt(p, y, lam=self.lam_global)
        prior = (self.global_params.a, self.global_params.b)
        self.segment_params = {}
        for key in set(s.tolist()):
            mask = s == key
            if mask.sum() >= self.min_count and len(np.unique(y[mask])) == 2:
                self.segment_params[key] = fit_platt(p[mask], y[mask], lam=self.lam_segment, prior=prior)
        return self

    def apply(self, p: float, segment: Hashable) -> float:
        prm = self.segment_params.get(segment, self.global_params)
        return float(prm.apply(p))


def mc_temperature_fit(
    prob_vectors: Sequence[Sequence[float]],
    outcome_idx: Sequence[int],
    lam: float = 3.0,
    bounds: tuple[float, float] = (0.5, 2.5),
) -> float:
    """Fit exponent t for multiple choice: p_k' proportional to p_k ** t.

    t > 1 sharpens (the MC analogue of Platt's a > 1). Penalised towards t = 1.
    """
    logs = [np.log(np.clip(np.asarray(v, dtype=float), EPS, 1)) for v in prob_vectors]
    idx = np.asarray(outcome_idx, dtype=int)

    def nll(t):
        t = float(t[0])
        total = 0.0
        for lv, k in zip(logs, idx):
            z = t * lv
            total -= z[k] - np.logaddexp.reduce(z)
        return total + lam * (t - 1) ** 2

    res = minimize(nll, x0=[1.0], method="L-BFGS-B", bounds=[bounds])
    return float(res.x[0])


def mc_temperature_apply(p: Sequence[float], t: float) -> list[float]:
    lv = t * np.log(np.clip(np.asarray(p, dtype=float), EPS, 1))
    q = np.exp(lv - np.logaddexp.reduce(lv))
    return q.tolist()
