"""Structured forecasting track. Shadow mode logs it and does not submit it."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .advanced import evidence_update
from .structured import Scenario, hazard_prob_at_least_one, scenario_tree

SUPPORTED_ARCHETYPES = ("deadline_event", "scheduled_decision")
FACTOR_BOUNDS = (0.25, 4.0)


def _num(x: Any, name: str, lo: float | None = None, hi: float | None = None) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"{name} must be a finite number, got {x!r}")
    if lo is not None and x < lo:
        raise ValueError(f"{name}={x} is below {lo}")
    if hi is not None and x > hi:
        raise ValueError(f"{name}={x} is above {hi}")
    return float(x)


def compute_baseline(
    archetype: str,
    baseline: Mapping[str, Any],
    time_remaining_days: float,
    prior_events: float = 0.5,
    multiplier_bounds: tuple[float, float] = (1 / 3, 3.0),
) -> tuple[float, dict]:
    if archetype == "deadline_event" and baseline.get("scheduled"):
        archetype = "scheduled_decision"
    if archetype == "deadline_event":
        k = _num(baseline.get("k_events"), "k_events", lo=0)
        years = _num(baseline.get("exposure_years"), "exposure_years", lo=0.1)
        mult = _num(baseline.get("rate_multiplier", 1.0), "rate_multiplier", lo=1e-6)
        horizon = _num(time_remaining_days, "time_remaining_days", lo=0) / 365.25
        p = hazard_prob_at_least_one(
            k, years, horizon, rate_multiplier=mult,
            prior_events=prior_events, multiplier_bounds=multiplier_bounds,
        )
        used = min(max(mult, multiplier_bounds[0]), multiplier_bounds[1])
        return p, {
            "method": "hazard", "k_events": k, "exposure_years": years,
            "rate_multiplier_used": used, "horizon_years": horizon,
        }
    if archetype == "scheduled_decision":
        raw = baseline.get("scenarios")
        if not isinstance(raw, list) or not 2 <= len(raw) <= 6:
            raise ValueError("scheduled decisions need 2-6 scenarios")
        scenarios = [
            Scenario(
                str(item.get("name", f"s{i}")),
                _num(item.get("p_scenario"), "p_scenario", 0, 1),
                _num(item.get("p_event_given_scenario"), "p_event_given_scenario", 0, 1),
            )
            for i, item in enumerate(raw)
        ]
        return scenario_tree(scenarios), {"method": "scenario_tree", "n_scenarios": len(scenarios)}
    raise ValueError(f"archetype {archetype!r} is not supported")


def clean_factors(raw: Sequence[Mapping[str, Any]] | None, max_factors: int = 6) -> list[dict]:
    out: list[dict] = []
    for i, factor in enumerate(raw or []):
        if not isinstance(factor, Mapping):
            continue
        try:
            multiplier = _num(factor.get("odds_multiplier"), "odds_multiplier", lo=1e-6)
        except ValueError:
            continue
        group = factor.get("independence_group")
        out.append({
            "odds_multiplier": min(max(multiplier, FACTOR_BOUNDS[0]), FACTOR_BOUNDS[1]),
            "group": str(group) if group not in (None, "") else f"ungrouped_{i}",
            "description": str(factor.get("description", ""))[:300],
            "claim_ids": [str(c) for c in (factor.get("claim_ids") or [])][:10],
        })
        if len(out) >= max_factors:
            break
    return out


def structured_forecast(base_p: float, factors: Sequence[Mapping[str, Any]]) -> dict:
    grouped, trace = evidence_update(base_p, factors)
    solo = [{**factor, "group": f"solo_{i}"} for i, factor in enumerate(factors)]
    naive, _unused = evidence_update(base_p, solo)
    return {
        "p_grouped": grouped, "p_naive": naive, "n_factors": len(factors),
        "n_groups": len(trace), "trace": trace,
    }


def run_structured_track(
    archetype: str,
    baseline_json: Mapping[str, Any] | None,
    factors_json: Mapping[str, Any] | None,
    time_remaining_days: float,
    **baseline_kwargs: Any,
) -> dict:
    record: dict[str, Any] = {"archetype": archetype, "valid": False, "error": None}
    if archetype not in SUPPORTED_ARCHETYPES:
        record["error"] = "not_applicable"
        return record
    try:
        base_p, detail = compute_baseline(
            archetype, baseline_json or {}, time_remaining_days, **baseline_kwargs
        )
        factors = clean_factors((factors_json or {}).get("factors"))
        record.update({
            "base_p": base_p, "baseline_detail": detail,
            **structured_forecast(base_p, factors), "factors": factors, "valid": True,
        })
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record
