from forecast_upgrades.timidity import extreme_trigger, stage_pulls


def test_unanimous_extreme_does_not_trigger_review():
    assert extreme_trigger([0.01, 0.02, 0.03]) is None
    assert extreme_trigger([0.99, 0.97]) is None


def test_split_extreme_triggers_review():
    assert extreme_trigger([0.02, 0.40]) == "split_extreme"
    assert extreme_trigger([0.01, 0.99]) == "split_extreme"


def test_stage_pull_towards_middle_is_positive():
    pulls = stage_pulls({"median": 0.8, "after_shrinkage": 0.7})
    assert pulls["after_shrinkage"] > 0
