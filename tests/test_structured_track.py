import json

import pytest

from structured_track import baseline_probability, parse_factors


def test_deadline_baseline_uses_hazard_not_one_half():
    probability, label = baseline_probability({
        "archetype": "deadline_event",
        "time_remaining_days": 365,
        "reference_class": {"k_events": 2, "exposure_years": 10},
    }, 0.5)
    assert label == "deadline hazard"
    assert probability is not None and probability != pytest.approx(0.5)


def test_missing_baseline_skips_update():
    probability, label = baseline_probability({"archetype": "other"}, 0.5)
    assert probability is None
    assert "no baseline" in label


def test_parse_factors_keeps_positive_multipliers():
    raw = json.dumps({"factors": [
        {"group": "g1", "odds_multiplier": 1.8, "note": "new poll"},
        {"group": "g2", "odds_multiplier": 0, "note": "bad"},
    ]})
    factors = parse_factors(raw)
    assert len(factors) == 1
    assert factors[0]["odds_multiplier"] == 1.8
