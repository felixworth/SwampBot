from forecast_upgrades.structured_track import run_structured_track


def test_unknown_group_is_ungrouped():
    baseline = {"scheduled": False, "k_events": 1, "exposure_years": 5, "rate_multiplier": 1}
    factors = {"factors": [
        {"odds_multiplier": 2, "independence_group": "not-in-ledger", "description": "x"},
    ]}
    record = run_structured_track(
        "deadline_event", baseline, factors, 100, ledger_groups={"g1"}
    )
    assert record["factors"][0]["group"].startswith("ungrouped_")


def test_other_archetypes_do_not_change_a_forecast():
    record = run_structured_track("numeric_quantity", None, None, 10)
    assert record["valid"] is False
    assert record["error"] == "not_applicable"


def test_deadline_track_groups_repeated_evidence():
    baseline = {
        "scheduled": False, "k_events": 2, "exposure_years": 10, "rate_multiplier": 1,
    }
    factors = {"factors": [
        {"odds_multiplier": 2, "independence_group": "g1", "description": "same story"},
        {"odds_multiplier": 2, "independence_group": "g1", "description": "copy"},
        {"odds_multiplier": 0.5, "independence_group": "g2", "description": "other"},
    ]}
    record = run_structured_track("deadline_event", baseline, factors, 365)
    assert record["valid"] is True
    assert record["n_groups"] == 2
    assert record["n_factors"] == 3
    assert record["p_naive"] != record["p_grouped"]
