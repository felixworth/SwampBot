import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on the path

import numpy as np
import pytest

from forecast_upgrades import aggregation as ag
from forecast_upgrades import calibration as cal
from forecast_upgrades import evaluation as ev
from forecast_upgrades import logging_schema as ls
from forecast_upgrades import numeric as nu
from forecast_upgrades import structured as st


# ------------------------------------------------------------------ aggregation
def test_binary_methods():
    p = [0.1, 0.2, 0.9]
    assert ag.aggregate_binary(p, "median") == pytest.approx(0.2)
    assert ag.aggregate_binary(p, "mean") == pytest.approx(0.4)
    odds = np.prod([x / (1 - x) for x in p]) ** (1 / 3)
    assert ag.aggregate_binary(p, "logodds_mean") == pytest.approx(odds / (1 + odds))
    base = ag.aggregate_binary([0.7, 0.8], "logodds_mean")
    assert ag.aggregate_binary([0.7, 0.8], "logodds_mean", extremize=1.5) > base
    with pytest.raises(ValueError):
        ag.aggregate_binary([1.2], "mean")


def test_mc_and_floor():
    out = ag.aggregate_mc([[0.7, 0.3, 0.0], [0.6, 0.4, 0.0]], floor=0.01)
    assert sum(out) == pytest.approx(1)
    assert min(out) >= 0.01 - 1e-12
    assert ag.normalize_with_floor([1, 0, 0, 0], 0.02)[1] == pytest.approx(0.02)


def test_blend_and_mix():
    assert ag.blend_logodds(0.3, 0.8, 0.0) == pytest.approx(0.3)
    assert ag.blend_logodds(0.3, 0.8, 1.0) == pytest.approx(0.8)
    m = ag.mix_cdfs([[0, 0.5, 1], [0, 0.1, 1]], weights=[3, 1])
    assert m[1] == pytest.approx(0.4)


# ------------------------------------------------------------------ calibration
def _synthetic(n, seed=1, slope=0.6, bias=0.4):
    rng = np.random.default_rng(seed)
    p_true = rng.beta(0.6, 1.2, n)  # skewed to "No", like Metaculus
    y = rng.binomial(1, p_true)
    raw = ag.sigmoid(slope * ag.logit(p_true) + bias)  # hedged + yes-leaning forecaster
    return raw, y


def test_platt_recovers_hedging_and_bias():
    raw, y = _synthetic(3000)
    prm = cal.fit_platt(raw, y, lam=1.0)
    assert prm.a > 1.2 and prm.b < 0
    assert cal.log_loss(prm.apply(raw), y) < cal.log_loss(raw, y)


def test_platt_shrinks_with_little_data():
    raw, y = _synthetic(12, seed=3)
    prm = cal.fit_platt(raw, y, lam=100.0)
    assert abs(prm.a - 1) < 0.1 and abs(prm.b) < 0.1


def test_platt_cv_and_cap():
    raw, y = _synthetic(400, seed=2)
    prm, scores = cal.fit_platt_cv(raw, y)
    assert set(scores) and prm.n == 400
    assert cal.cap(0.999) == 0.98 and cal.cap(0.001) == 0.02 and cal.cap(0.5) == 0.5


def test_segmented_calibrator_fallback():
    raw, y = _synthetic(500, seed=4)
    seg = ["big"] * 450 + ["small"] * 50
    sc = cal.SegmentedCalibrator(min_count=100).fit(raw, y, seg)
    assert "big" in sc.segment_params and "small" not in sc.segment_params
    assert sc.apply(0.3, "small") == pytest.approx(float(sc.global_params.apply(0.3)))


def test_mc_temperature_sharpens_flat_forecasts():
    rng = np.random.default_rng(0)
    vecs, idx = [], []
    for _ in range(800):
        true = rng.dirichlet([0.5, 0.5, 0.5])
        k = rng.choice(3, p=true)
        flat = true**0.5 / (true**0.5).sum()  # under-confident forecaster
        vecs.append(flat)
        idx.append(k)
    t = cal.mc_temperature_fit(vecs, idx, lam=0.1)
    assert t > 1.3
    assert sum(cal.mc_temperature_apply([0.5, 0.3, 0.2], t)) == pytest.approx(1)


# ------------------------------------------------------------------ structured
def test_hazard_model():
    assert st.hazard_prob_at_least_one(2, 10, 0.5, prior_events=0.5) == pytest.approx(1 - (10 / 10.5) ** 2.5)
    assert st.hazard_prob_at_least_one(0, 10, 0.5) > 0
    a = st.hazard_prob_at_least_one(2, 10, 0.25)
    b = st.hazard_prob_at_least_one(2, 10, 1.0)
    assert b > a  # time-scaling is automatic
    capped = st.hazard_prob_at_least_one(2, 10, 0.5, rate_multiplier=50)
    assert capped == pytest.approx(st.hazard_prob_at_least_one(2, 10, 0.5, rate_multiplier=3))


def test_coherence_tools():
    assert st.enforce_monotone_deadlines([0.3, 0.2, 0.5]) == pytest.approx([0.25, 0.25, 0.5])
    assert st.enforce_monotone_thresholds([0.5, 0.6, 0.2]) == pytest.approx([0.55, 0.55, 0.2])
    assert st.negation_gap(0.6, 0.6) == pytest.approx(0.2)
    s = [st.Scenario("united", 0.7, 0.8), st.Scenario("split", 0.3, 0.2)]
    assert st.scenario_tree(s) == pytest.approx(0.62)
    with pytest.raises(ValueError):
        st.scenario_tree([st.Scenario("a", 0.5, 0.5), st.Scenario("b", 0.3, 0.5)])


def test_prob_exceeds():
    g = np.linspace(0, 100, 101)
    c = g / 100
    assert st.prob_exceeds(g, c, 30) == pytest.approx(0.7)
    assert st.prob_exceeds(g, c, -5) == pytest.approx(1.0)


# ------------------------------------------------------------------ numeric
PCTS = {0.05: 20.0, 0.25: 35.0, 0.5: 45.0, 0.75: 55.0, 0.95: 75.0}


@pytest.mark.parametrize("zp", [None, -10.0])
def test_grid_roundtrip(zp):
    x = np.array([0.0, 12.5, 50.0, 99.0])
    u = nu.scaled_to_unscaled(x, 0, 100, zp)
    assert nu.unscaled_to_scaled(u, 0, 100, zp) == pytest.approx(x)


@pytest.mark.parametrize("open_lower,open_upper", [(False, False), (True, True), (False, True), (True, False)])
def test_percentiles_to_cdf_valid(open_lower, open_upper):
    mn, mx = nu.default_step_bounds(201)
    cdf = nu.percentiles_to_cdf(PCTS, 0, 100, open_lower, open_upper)
    assert len(cdf) == 201
    assert nu.validate_cdf(cdf, open_lower, open_upper, mn, mx) == []
    grid = nu.make_grid(0, 100)
    assert nu.cdf_percentile(grid, cdf, 0.5) == pytest.approx(45, abs=1.5)


def test_log_scaled_and_out_of_range_and_discrete():
    mn, mx = nu.default_step_bounds(201)
    cdf = nu.percentiles_to_cdf({0.1: 2.0, 0.5: 10.0, 0.9: 60.0}, 1, 1000, True, True, zero_point=0.0)
    assert nu.validate_cdf(cdf, True, True, mn, mx) == []
    # model believes the value is far above the range: still a legal CDF
    hi = nu.percentiles_to_cdf({0.1: 500.0, 0.5: 800.0, 0.9: 1200.0}, 0, 100, True, True)
    assert nu.validate_cdf(hi, True, True, mn, mx) == []
    assert 1 - hi[-1] > 0.9
    # discrete question with 8 inbound outcomes -> 9 points
    mn9, mx9 = nu.default_step_bounds(9)
    d = nu.percentiles_to_cdf({0.1: 1.0, 0.5: 3.0, 0.9: 6.0}, 0, 8, False, False, n_points=9)
    assert len(d) == 9 and nu.validate_cdf(d, False, False, mn9, mx9) == []


def test_enforce_step_bounds_on_spiky_cdf():
    c = np.concatenate([np.zeros(100), np.ones(101)])  # a point mass
    out = nu.enforce_step_bounds(c, 5.5e-5, 0.15)
    d = np.diff(out)
    assert d.min() >= 5.5e-5 - 1e-12 and d.max() <= 0.15 + 1e-12
    assert out[0] == 0 and out[-1] == 1


def test_widen_and_baseline():
    w = nu.widen_percentiles(PCTS, factor=1.5)
    assert w[0.5] == pytest.approx(45)
    assert w[0.05] < PCTS[0.05] and w[0.95] > PCTS[0.95]
    rng = np.random.default_rng(0)
    series = 100 * np.exp(np.cumsum(rng.normal(0.01, 0.02, 400)))
    b0 = nu.empirical_baseline_percentiles(series, 5, drift_damping=0.0)
    b1 = nu.empirical_baseline_percentiles(series, 5, drift_damping=1.0)
    assert b1[0.5] > b0[0.5]
    assert b0[0.025] < series[-1] < b0[0.975]


# ------------------------------------------------------------------ evaluation
def test_scores_and_replay():
    assert ev.head_to_head(0.9, 0.5, 1) == pytest.approx(100 * math.log(0.9 / 0.5))
    assert ev.head_to_head(0.01, 0.03, 1) == pytest.approx(-109.86, abs=0.01)
    assert ev.head_to_head(0.01, 0.03, 0) == pytest.approx(2.04, abs=0.01)
    raw, y = _synthetic(600, seed=5)
    rng = np.random.default_rng(6)
    rows = [{"members": list(np.clip(r + rng.normal(0, 0.05, 3), 0.01, 0.99)), "outcome": int(o)} for r, o in zip(raw, y)]
    prm = cal.fit_platt([ag.aggregate_binary(r["members"]) for r in rows[:300]], [r["outcome"] for r in rows[:300]])
    test_rows = rows[300:]
    res = ev.replay_binary(
        test_rows,
        {
            "median": lambda r: ag.aggregate_binary(r["members"], "median"),
            "median+platt+cap": lambda r: cal.cap(float(prm.apply(ag.aggregate_binary(r["members"], "median")))),
        },
        baseline="median",
    )
    assert res["median+platt+cap"]["h2h_vs_baseline"]["mean"] > 0
    table = ev.calibration_table([0.1, 0.15, 0.8], [0, 0, 1], n_bins=5)
    assert sum(r["n"] for r in table) == 3


def test_bootstrap_clusters():
    d = [1.0, 1.0, -1.0, 3.0]
    r = ev.paired_bootstrap(d, clusters=["a", "a", "b", "c"], n_boot=500)
    assert r["lo"] <= r["mean"] <= r["hi"]


# ------------------------------------------------------------------ logging
def test_logging_roundtrip(tmp_path):
    rec = ls.QuestionRecord(
        question_id=1, post_id=2, tournament="minibench", qtype="binary", title="t", close_time=None, bot_version="v1",
        members=[ls.MemberForecast(model="m", sample_idx=0, stage="initial", research_bundle_ids=["abc"], binary=0.3)],
    )
    path = tmp_path / "log.jsonl"
    ls.append_record(path, rec)
    ls.append_record(path, rec)
    loaded = ls.load_records(path)
    assert len(loaded) == 2 and ls.members_at_stage(loaded[0]) == [0.3]
    json.dumps(loaded[0])


def test_sharp_forecast_stays_sharp_and_bound_diagnostics():
    mn, mx = nu.default_step_bounds(201)
    g = nu.make_grid(0, 100)
    c = nu.percentiles_to_cdf({0.05: 49.9, 0.5: 50.0, 0.95: 50.1}, 0, 100, False, False)
    assert nu.validate_cdf(c, False, False, mn, mx) == []
    assert 47 < nu.cdf_percentile(g, c, 0.05) and nu.cdf_percentile(g, c, 0.95) < 53
    diag = nu.bound_diagnostics({0.05: 40.0, 0.5: 75.0, 0.95: 120.0}, 0, 100)
    assert diag["mass_above_max"] > 0.05  # impossible mass on a closed bound -> flag it
