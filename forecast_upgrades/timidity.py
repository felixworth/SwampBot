"""Helpers for making the bot less conservative where the evidence justifies it."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .calibration import fit_platt

STAGE_ORDER = ["median", "after_review", "after_shrinkage", "after_anchor", "after_calibration", "final"]


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def extreme_trigger(probs: Sequence[float], lo: float = 0.05, hi: float = 0.95) -> str | None:
    """Review extremes only when members are split, not when they all agree."""
    p = np.asarray(probs, dtype=float)
    low, high = p <= lo, p >= hi
    if not (low.any() or high.any()):
        return None
    if low.all() or high.all():
        return None
    return "split_extreme"


def members_agree_extreme(probs: Sequence[float], side: str, threshold: float = 0.9) -> bool:
    p = np.asarray(probs, dtype=float)
    if side == "yes":
        return bool((p >= threshold).all())
    if side == "no":
        return bool((p <= 1 - threshold).all())
    raise ValueError("side must be 'yes' or 'no'")


def choose_prior(
    structured_base_p: float | None = None,
    outside_view_rate: float | None = None,
    default: float = 0.35,
    lo: float = 0.03,
    hi: float = 0.97,
) -> tuple[float, str]:
    for value, source in ((structured_base_p, "structured_baseline"), (outside_view_rate, "outside_view")):
        if isinstance(value, (int, float)) and np.isfinite(value) and 0 < value < 1:
            return float(np.clip(value, lo, hi)), source
    return default, "default"


def stage_pulls(stages: Mapping[str, float | None], order: Sequence[str] = STAGE_ORDER) -> dict[str, float]:
    present = [s for s in order if isinstance(stages.get(s), (int, float))]
    return {b: float(abs(_logit(stages[a])) - abs(_logit(stages[b]))) for a, b in zip(present[:-1], present[1:])}


def summarize_pulls(stage_dicts: Sequence[Mapping[str, float | None]], order: Sequence[str] = STAGE_ORDER) -> list[dict]:
    pulls: dict[str, list[float]] = {}
    for stage in stage_dicts:
        for key, value in stage_pulls(stage, order).items():
            pulls.setdefault(key, []).append(value)
    rows = []
    for key in order:
        if key in pulls:
            values = np.asarray(pulls[key])
            rows.append({
                "stage": key,
                "n": int(values.size),
                "mean_pull": float(values.mean()),
                "share_towards_middle": float((values > 1e-9).mean()),
                "p90_pull": float(np.quantile(values, 0.9)),
            })
    return rows


def calibration_slope(probs: Sequence[float], outcomes: Sequence[int]) -> tuple[float, float]:
    params = fit_platt(probs, outcomes, lam=0.0, a_bounds=(0.05, 10.0), b_bounds=(-5.0, 5.0))
    return params.a, params.b
