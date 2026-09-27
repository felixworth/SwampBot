"""Round-2 upgrades, based on research published up to September 2026.

- lift_floor: multiple-choice floor that lifts only tiny options (replaces uniform mixing).
- shrink_logits / hierarchical_logit_aggregate: combine samples in log-odds, shrinking
  towards a prior when they disagree (in the spirit of BLF, arXiv 2604.18576).
- HierarchicalPlatt: shared slope, per-source intercepts with partial pooling
  (e.g. separate intercepts for the seasonal tournament, MiniBench and the Cup).
- conditional_cap: caps that widen only when the outcome is verified at the source.
- evidence_update: mechanical odds-space updates from a base probability, one
  factor or likelihood ratio per independence group (in the spirit of AuditForecast,
  arXiv 2609.05905, and LEAP, arXiv 2609.01337).
Pure numpy/scipy; no network or LLM calls.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

EPS = 1e-9


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


# ------------------------------------------------------------ multiple choice floor
def lift_floor(probs: Sequence[float], floor: float = 0.01, max_iter: int = 100) -> list[float]:
    """Raise options below `floor` to exactly `floor`; take that mass proportionally
    from the options above it. Unlike mixing with a uniform distribution, the
    favourite loses only what the tiny options actually need.

    forecasting-tools clamps every option to >= 0.01 and rejects any option that moves
    by more than 0.05, so pre-flooring at 0.01 this way keeps the SDK from changing anything.
    """
    p = np.clip(np.asarray(probs, dtype=float), 0, None)
    if p.sum() <= 0:
        raise ValueError("probabilities sum to zero")
    p = p / p.sum()
    n = p.size
    if floor * n >= 1:
        raise ValueError("floor too large for the number of options")
    fixed = np.zeros(n, dtype=bool)
    for _ in range(max_iter):
        low = (p < floor - 1e-15) & ~fixed
        if not low.any():
            break
        fixed |= low
        p[fixed] = floor
        free = ~fixed
        p[free] = p[free] / p[free].sum() * (1 - floor * fixed.sum())
    return (p / p.sum()).tolist()


# ------------------------------------------------------------ aggregation
def shrink_logits(
    probs: Sequence[float],
    prior_p: float = 0.35,
    tau: float = 2.0,
    min_sd: float = 0.3,
) -> tuple[float, dict]:
    """Normal-normal shrinkage of the mean logit towards logit(prior_p).

    Each sample logit ~ N(theta, s^2); prior theta ~ N(logit(prior_p), tau^2).
    Weight on the samples = tau^2 / (tau^2 + s^2 / K). Agreeing samples keep their
    mean; disagreeing samples are pulled towards the prior. prior_p should be your
    base rate for the question type (about 1/3 for binary questions overall),
    tau the spread of true logits across questions (fit it from your logs).
    """
    x = _logit(probs)
    k = x.size
    if k == 0:
        raise ValueError("no forecasts")
    s2 = max(float(np.var(x, ddof=1)) if k > 1 else 0.0, min_sd**2)
    se2 = s2 / k
    w = tau**2 / (tau**2 + se2)
    post = w * x.mean() + (1 - w) * float(_logit(prior_p))
    return float(_sigmoid(post)), {"weight_on_samples": w, "mean_logit": float(x.mean()), "se": se2**0.5}


def hierarchical_logit_aggregate(
    samples_by_model: Mapping[str, Sequence[float]],
    prior_p: float = 0.35,
    tau: float = 2.0,
    min_sd: float = 0.3,
) -> tuple[float, dict]:
    """Average samples within each model first (so a model with more samples does not
    dominate), then shrink the across-model mean towards the prior based on how much
    the models disagree."""
    model_means = [float(_sigmoid(_logit(v).mean())) for v in samples_by_model.values() if len(v)]
    return shrink_logits(model_means, prior_p=prior_p, tau=tau, min_sd=min_sd)


# ------------------------------------------------------------ calibration
@dataclass
class HierarchicalPlatt:
    """p' = sigmoid(a * logit(p) + b_source), with b_source ~ N(b_global, 1 / lam_source).

    Sources with few resolved questions borrow strength from the global intercept;
    unknown sources use the global intercept. Penalties keep a near 1 and b near 0
    when data are scarce.
    """

    lam_slope: float = 3.0
    lam_global: float = 3.0
    lam_source: float = 10.0
    a: float = 1.0
    b_global: float = 0.0
    b_source: dict[Hashable, float] = field(default_factory=dict)

    def fit(self, probs, outcomes, sources) -> "HierarchicalPlatt":
        x = _logit(probs)
        y = np.asarray(outcomes, dtype=float)
        src = list(sources)
        keys = sorted(set(src), key=str)
        idx = np.array([keys.index(s) for s in src])

        def nll(theta):
            a, bg, bs = theta[0], theta[1], theta[2:]
            z = a * x + bs[idx]
            ll = y * -np.logaddexp(0, -z) + (1 - y) * -np.logaddexp(0, z)
            return (-ll.sum() + self.lam_slope * (a - 1) ** 2 + self.lam_global * bg**2
                    + self.lam_source * ((bs - bg) ** 2).sum())

        theta0 = np.concatenate([[1.0, 0.0], np.zeros(len(keys))])
        bounds = [(0.5, 2.5), (-2, 2)] + [(-2, 2)] * len(keys)
        res = minimize(nll, theta0, method="L-BFGS-B", bounds=bounds)
        self.a, self.b_global = float(res.x[0]), float(res.x[1])
        self.b_source = {k: float(v) for k, v in zip(keys, res.x[2:])}
        return self

    def apply(self, p: float, source: Hashable) -> float:
        b = self.b_source.get(source, self.b_global)
        return float(_sigmoid(self.a * _logit(p) + b))


# ------------------------------------------------------------ caps
def conditional_cap(
    p: float,
    auto_resolved: bool,
    verified_at_source: bool = False,
    members_agree_extreme: bool = False,
) -> float:
    """Default caps 2-98% (3-97% on LLM-resolved questions). Widen to 0.5-99.5%
    (1-99% LLM-resolved) ONLY when the resolution source itself shows the outcome
    (already happened / can no longer happen) AND every member is on that extreme side.

    Break-even: moving a cap from 2% to 0.5% gains ~1.5 points when right and loses
    ~139 when wrong, so it pays only if the true chance of being wrong is below ~1.1%.
    """
    if verified_at_source and members_agree_extreme:
        lo, hi = (0.01, 0.99) if auto_resolved else (0.005, 0.995)
    else:
        lo, hi = (0.03, 0.97) if auto_resolved else (0.02, 0.98)
    return float(np.clip(p, lo, hi))


def cap_break_even(cap_tight: float, cap_loose: float) -> float:
    """Error rate above which the tighter (less extreme) cap scores better under log loss."""
    gain_if_right = np.log(1 - cap_loose) - np.log(1 - cap_tight)
    loss_if_wrong = np.log(cap_tight) - np.log(cap_loose)
    return float(gain_if_right / (gain_if_right + loss_if_wrong))


# ------------------------------------------------------------ structured evidence updates
def evidence_update(
    base_p: float,
    items: Sequence[Mapping],
    eta: float = 1.0,
    per_item_cap: float = float(np.log(4.0)),
    total_cap: float = float(np.log(20.0)),
    within_group: str = "max",
) -> tuple[float, list[dict]]:
    """Posterior from a base probability plus odds multipliers.

    items: [{"group": "g1", "odds_multiplier": 1.8, "note": "..."}] where the multiplier is
    P(evidence | YES) / P(evidence | NO), or equivalently the factor's odds multiplier.
    Within an independence group only the strongest item counts (within_group="max") or
    the group mean ("mean"): correlated reports must not be multiplied together.
    Each item is clipped to +/- per_item_cap in log-odds and the total shift to +/- total_cap.
    eta scales all evidence (LEAP found 1.0 best in its tests; tune on your data).
    """
    groups: dict[Hashable, list[float]] = {}
    for it in items:
        m = float(it["odds_multiplier"])
        if m <= 0:
            raise ValueError("odds multipliers must be positive")
        groups.setdefault(it.get("group", id(it)), []).append(float(np.clip(np.log(m), -per_item_cap, per_item_cap)))
    trace, total = [], 0.0
    for g, lls in groups.items():
        contrib = max(lls, key=abs) if within_group == "max" else float(np.mean(lls))
        total += contrib
        trace.append({"group": g, "log_odds": contrib, "n_items": len(lls)})
    shift = float(np.clip(eta * total, -total_cap, total_cap))
    return float(_sigmoid(_logit(base_p) + shift)), trace
