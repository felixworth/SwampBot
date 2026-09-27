"""Sprint C: a code baseline plus odds-space factor updates."""
from __future__ import annotations

import json

from forecast_logic import parse_json
from forecast_upgrades.structured import hazard_prob_at_least_one, scenario_tree, Scenario
from upgrade_config import UPGRADE


def baseline_probability(spec: dict | None, direct: float) -> tuple[float | None, str]:
    """Return a non-0.5 baseline when the resolution spec supports one."""
    if not isinstance(spec, dict):
        return None, "no resolution spec"
    archetype = spec.get("archetype")
    counts = spec.get("reference_class") or {}
    if archetype == "deadline_event" and isinstance(counts, dict):
        try:
            probability = hazard_prob_at_least_one(
                float(counts["k_events"]),
                float(counts["exposure_years"]),
                max(float(spec.get("time_remaining_days") or 0), 0) / 365.25,
                prior_events=UPGRADE["hazard_prior_events"],
                multiplier_bounds=UPGRADE["hazard_multiplier_bounds"],
            )
            return probability, "deadline hazard"
        except (KeyError, TypeError, ValueError):
            return None, "deadline hazard missing counts"
    if archetype == "scheduled_decision" and isinstance(spec.get("scenarios"), list):
        try:
            scenarios = [
                Scenario(str(item.get("name") or "scenario"), float(item["p_scenario"]), float(item["p_event_given_scenario"]))
                for item in spec["scenarios"]
            ]
            return scenario_tree(scenarios), "scenario tree"
        except (KeyError, TypeError, ValueError):
            return None, "scenario tree invalid"
    if 0 < direct < 1 and abs(direct - 0.5) > 0.02:
        return direct, "ensemble used as baseline because no countable reference class"
    return None, "no baseline"


def factor_prompt(base_p: float, description: str, research: str) -> str:
    return (
        "List situational factors NOT already captured by this baseline. "
        f"Baseline probability {base_p:.3f}. {description}. "
        "For each factor give an odds multiplier between 0.25 and 4, the claim ids "
        "supporting it, and its independence group. Return ONLY JSON: "
        '{"factors":[{"group":"g1","odds_multiplier":1.5,"note":""}]}'
        "\nRESEARCH:\n" + research[:16000]
    )


def parse_factors(raw: str) -> list[dict]:
    data = parse_json(raw)
    factors = data.get("factors")
    if not isinstance(factors, list):
        raise ValueError("factors must be a list")
    cleaned = []
    for item in factors:
        if not isinstance(item, dict):
            continue
        multiplier = float(item.get("odds_multiplier"))
        if multiplier <= 0:
            continue
        cleaned.append({
            "group": str(item.get("group") or "g"),
            "odds_multiplier": multiplier,
            "note": str(item.get("note") or ""),
        })
    return cleaned


def attach_factors(candidates: list[dict], factors: list[dict]) -> None:
    if candidates and factors:
        candidates[0]["factors"] = factors
        candidates[0]["structured_trace"] = json.dumps(factors)
