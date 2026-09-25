"""Numeric / date / discrete questions: percentiles -> submission CDF, plus data baselines.

Conventions follow the Metaculus API as documented publicly (openapi docs and
bot makers who checked the server source):
  * CDF values sit on a grid of n points over the question range (201 for
    continuous questions; inbound_outcome_count + 1 for discrete ones).
  * Log-scaled questions (zero_point set) use a geometric grid.
  * Closed lower bound -> cdf[0] == 0.0; closed upper bound -> cdf[-1] == 1.0.
  * Open bounds: cdf[0] = P(below range_min), 1 - cdf[-1] = P(above range_max).
  * Per-step increases must stay within [min_step, max_step].

Step limits and the 0.001 open-bound minimum were checked against the official
forecasting-tools v0.3.1 source (NumericDefaults and _standardize_cdf). Re-check
if Metaculus changes its validation.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import expit
from scipy.stats import norm

EPS = 1e-9


# ---------------------------------------------------------------- grid scaling
def unscaled_to_scaled(u, range_min: float, range_max: float, zero_point: float | None = None):
    u = np.asarray(u, dtype=float)
    if zero_point is None:
        return range_min + (range_max - range_min) * u
    ratio = (range_max - zero_point) / (range_min - zero_point)
    return range_min + (range_max - range_min) * (ratio**u - 1) / (ratio - 1)


def scaled_to_unscaled(x, range_min: float, range_max: float, zero_point: float | None = None):
    x = np.asarray(x, dtype=float)
    if zero_point is None:
        return (x - range_min) / (range_max - range_min)
    ratio = (range_max - zero_point) / (range_min - zero_point)
    arg = (x - range_min) * (ratio - 1) / (range_max - range_min) + 1
    # values far outside the range on the "wrong" side of zero_point: push to extremes
    arg = np.where(arg <= 0, EPS, arg)
    return np.log(arg) / np.log(ratio)


def make_grid(range_min: float, range_max: float, zero_point: float | None = None, n_points: int = 201) -> np.ndarray:
    return unscaled_to_scaled(np.linspace(0, 1, n_points), range_min, range_max, zero_point)


def default_step_bounds(n_points: int) -> tuple[float, float]:
    """(min_step, max_step) matching the official forecasting-tools v0.3.1 rules.

    Minimum rise per step is 0.01 / inbound_outcome_count (the official code mixes in
    1% uniform); maximum per-step mass is 0.2 * 200 / inbound_outcome_count. We keep
    5% headroom on both, as the official code does for the cap.
    """
    bins = n_points - 1
    min_step = 1.05 * 0.01 / bins
    max_step = min(1.0, 0.95 * 0.2 * 200 / bins)
    return min_step, max_step


# ---------------------------------------------------------- percentiles -> CDF
def _logistic_tail(u_pts: np.ndarray, q_pts: np.ndarray, u_eval: np.ndarray) -> np.ndarray:
    """Extrapolate beyond two outer points with a straight line in logit(q)."""
    z = np.log(q_pts / (1 - q_pts))
    slope = (z[1] - z[0]) / max(u_pts[1] - u_pts[0], EPS)
    slope = max(slope, 1.0)  # keep tails decaying even for odd inputs
    ze = z[0] + slope * (u_eval - u_pts[0])
    return expit(ze)


def _raw_cdf(percentiles, range_min, range_max, zero_point, n_points):
    """Monotone CDF on the unscaled grid BEFORE any bound rules (may put mass outside)."""
    if len(percentiles) < 3:
        raise ValueError("need at least 3 percentiles")
    qs = np.array(sorted(percentiles), dtype=float)
    if qs[0] <= 0 or qs[-1] >= 1:
        raise ValueError("quantiles must be strictly inside (0, 1)")
    vals = np.array([percentiles[q] for q in qs], dtype=float)
    if np.any(np.diff(vals) < 0):
        raise ValueError("percentile values must be non-decreasing")
    u = scaled_to_unscaled(vals, range_min, range_max, zero_point)
    for i in range(1, u.size):  # break ties so the interpolant is strictly monotone
        if u[i] <= u[i - 1]:
            u[i] = u[i - 1] + 1e-6
    ug = np.linspace(0, 1, n_points)
    f = np.empty_like(ug)
    mid = (ug >= u[0]) & (ug <= u[-1])
    f[mid] = PchipInterpolator(u, qs, extrapolate=False)(ug[mid])
    lo = ug < u[0]
    hi = ug > u[-1]
    f[lo] = _logistic_tail(u[:2], qs[:2], ug[lo])
    f[hi] = 1 - _logistic_tail(-u[::-1][:2], (1 - qs[::-1])[:2], -ug[hi])
    return np.maximum.accumulate(np.clip(f, 0, 1))


def bound_diagnostics(
    percentiles: Mapping[float, float],
    range_min: float,
    range_max: float,
    zero_point: float | None = None,
    n_points: int = 201,
) -> dict[str, float]:
    """Probability the model's own percentiles put below/above the range.

    On a CLOSED bound this mass is impossible and gets silently squashed into
    the edge bin - log it and treat > ~2% as a red flag (units error, misread
    criteria, or a bound the model ignored) that should trigger a re-check.
    """
    f = _raw_cdf(percentiles, range_min, range_max, zero_point, n_points)
    return {"mass_below_min": float(f[0]), "mass_above_max": float(1 - f[-1])}


def percentiles_to_cdf(
    percentiles: Mapping[float, float],
    range_min: float,
    range_max: float,
    open_lower: bool,
    open_upper: bool,
    zero_point: float | None = None,
    n_points: int = 201,
    min_step: float | None = None,
    max_step: float | None = None,
    open_bound_min_mass: float = 0.002,
) -> np.ndarray:
    """Turn {quantile: value} into a submission-ready CDF on the question grid.

    Monotone PCHIP between the elicited percentiles (as Panshul42/nostreambot do),
    logit-linear tails beyond them, bound rules applied, then step constraints.
    Include extreme quantiles (0.025/0.975 or 0.01/0.99) in the elicitation so
    the tails come from the model rather than from this extrapolation.
    """
    f = _raw_cdf(percentiles, range_min, range_max, zero_point, n_points)

    d_min, d_max = default_step_bounds(n_points)
    min_step = d_min if min_step is None else min_step
    max_step = d_max if max_step is None else max_step
    min_inbound = 1.5 * min_step * (n_points - 1)

    lo_val = float(np.clip(f[0], open_bound_min_mass, 1 - open_bound_min_mass)) if open_lower else 0.0
    hi_val = float(np.clip(f[-1], open_bound_min_mass, 1 - open_bound_min_mass)) if open_upper else 1.0
    if hi_val - lo_val < min_inbound:
        # model put almost everything out of range: keep a thin but legal inbound band
        if open_upper and hi_val < 1 - open_bound_min_mass:
            hi_val = min(1 - open_bound_min_mass, lo_val + min_inbound)
        if hi_val - lo_val < min_inbound and open_lower:
            lo_val = max(open_bound_min_mass, hi_val - min_inbound)
    f[0], f[-1] = lo_val, hi_val
    f[1:-1] = np.clip(f[1:-1], lo_val, hi_val)
    f = np.maximum.accumulate(f)
    return enforce_step_bounds(f, min_step=min_step, max_step=max_step)


def enforce_step_bounds(cdf: Sequence[float], min_step: float, max_step: float, max_iter: int = 20000) -> np.ndarray:
    """Adjust per-bin increments into [min_step, max_step], keeping cdf[0] and cdf[-1] fixed.

    Small bins are lifted to min_step with a proportional (shape-preserving) reduction
    elsewhere; any bin above max_step pushes its excess into its NEIGHBOURS, so a
    very sharp forecast stays sharp instead of being smeared across the range.
    """
    c = np.asarray(cdf, dtype=float)
    start, end = float(c[0]), float(c[-1])
    total = end - start
    d = np.maximum(np.diff(c), 0.0)
    n = d.size
    if min_step * n > total + 1e-12:
        raise ValueError(f"min_step {min_step} infeasible for inbound mass {total:.4f}")
    if max_step * n < total - 1e-12:
        raise ValueError(f"max_step {max_step} infeasible for inbound mass {total:.4f}")
    # lift to the floor, shrink the excess above the floor proportionally
    above = np.maximum(d, min_step) - min_step
    free = total - n * min_step
    d = min_step + (above * free / above.sum() if above.sum() > 0 else np.full(n, free / n))
    # spill anything above the ceiling into adjacent bins
    for _ in range(max_iter):
        over = np.flatnonzero(d > max_step + 1e-15)
        if over.size == 0:
            break
        for i in over:
            excess = d[i] - max_step
            d[i] = max_step
            if 0 < i < n - 1:
                d[i - 1] += excess / 2
                d[i + 1] += excess / 2
            elif i == 0:
                d[1] += excess
            else:
                d[n - 2] += excess
    out = start + np.concatenate([[0.0], np.cumsum(d)])
    out[-1] = end
    return out


def validate_cdf(
    cdf: Sequence[float],
    open_lower: bool,
    open_upper: bool,
    min_step: float,
    max_step: float,
    open_bound_min_mass: float = 0.001,
) -> list[str]:
    """Return a list of human-readable violations (empty list == OK). Run before every submit."""
    c = np.asarray(cdf, dtype=float)
    errs = []
    if np.any(np.isnan(c)):
        errs.append("NaN in CDF")
    d = np.diff(c)
    if np.any(d < min_step - 1e-12):
        errs.append(f"step below min_step at bins {np.where(d < min_step - 1e-12)[0][:5].tolist()}")
    if np.any(d > max_step + 1e-12):
        errs.append(f"step above max_step at bins {np.where(d > max_step + 1e-12)[0][:5].tolist()}")
    if not open_lower and c[0] != 0.0:
        errs.append("closed lower bound requires cdf[0] == 0")
    if not open_upper and c[-1] != 1.0:
        errs.append("closed upper bound requires cdf[-1] == 1")
    if open_lower and c[0] < open_bound_min_mass:
        errs.append("open lower bound needs some mass below range_min")
    if open_upper and c[-1] > 1 - open_bound_min_mass:
        errs.append("open upper bound needs some mass above range_max")
    return errs


def cdf_percentile(grid: Sequence[float], cdf: Sequence[float], q: float) -> float:
    """Inverse lookup (for comments, sanity checks and interval-coverage evaluation)."""
    return float(np.interp(q, np.asarray(cdf, dtype=float), np.asarray(grid, dtype=float)))


# ------------------------------------------------------------- tail widening
def widen_percentiles(percentiles: Mapping[float, float], factor: float = 1.25, log_space: bool = False) -> dict[float, float]:
    """Push outer percentiles away from the median; widening grows with |z| of the quantile.

    LLM intervals tend to be too narrow; tune `factor` on resolved questions via
    interval coverage (does ~80% of outcomes land inside your 10-90 range?).
    """
    qs = sorted(percentiles)
    vals = np.array([percentiles[q] for q in qs], dtype=float)
    if log_space:
        if np.any(vals <= 0):
            raise ValueError("log_space needs positive values")
        vals = np.log(vals)
    med = float(np.interp(0.5, qs, vals))
    zref = norm.ppf(0.975)
    out = {}
    for q, v in zip(qs, vals):
        k = 1 + (factor - 1) * min(1.0, abs(norm.ppf(q)) / zref)
        w = med + (v - med) * k
        out[q] = float(np.exp(w)) if log_space else float(w)
    return out


# ------------------------------------------------------------- data baselines
def empirical_baseline_percentiles(
    series: Sequence[float],
    horizon_steps: int,
    quantiles: Sequence[float] = (0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975),
    log_space: bool = True,
    window: int | None = None,
    drift_damping: float = 0.5,
    min_changes: int = 20,
) -> dict[float, float]:
    """Outside-view distribution of the value `horizon_steps` ahead from the series' own history.

    Uses overlapping historical h-step changes (log changes if log_space). The
    historical drift is multiplied by drift_damping (0 = pure random walk,
    1 = full trend) because extrapolating strong trends is exactly where more
    capable LLMs over-commit their upper tail.
    """
    x = np.asarray(series, dtype=float)
    if window is not None:
        x = x[-(window + horizon_steps):]
    if log_space:
        if np.any(x <= 0):
            raise ValueError("log_space needs a strictly positive series")
        x = np.log(x)
    ch = x[horizon_steps:] - x[:-horizon_steps]
    if ch.size < min_changes:
        raise ValueError(f"only {ch.size} historical changes; widen the window or use judgment")
    mu = ch.mean()
    centred = ch - mu
    out = {}
    for q in sorted(quantiles):
        v = x[-1] + drift_damping * mu + np.quantile(centred, q)
        out[q] = float(np.exp(v)) if log_space else float(v)
    return out
