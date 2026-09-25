import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # repo root on the path

from forecast_upgrades import advanced as adv
from forecast_upgrades.replay_adapter import events_to_records


def test_lift_floor_beats_uniform_mixing():
    p = [0.8, 0.2] + [0.0] * 18  # 20 options, favourite at 80%
    out = adv.lift_floor(p, 0.01)
    assert sum(out) == pytest.approx(1) and min(out) >= 0.01 - 1e-12
    assert out[0] == pytest.approx(0.8 * (1 - 18 * 0.01) / 1.0, rel=1e-6)  # only 18% moved, proportionally
    assert out[0] > 0.8 * (1 - 0.2) + 0.2 / 20  # better than the 20% uniform mix needed for the same floor
    assert adv.lift_floor([0.5, 0.3, 0.2], 0.01) == pytest.approx([0.5, 0.3, 0.2])  # untouched when not needed
    with pytest.raises(ValueError):
        adv.lift_floor([1] * 150, 0.01)


def test_shrink_logits_behaviour():
    agree, info_a = adv.shrink_logits([0.80, 0.82, 0.78], prior_p=0.35)
    split, info_s = adv.shrink_logits([0.95, 0.40, 0.85], prior_p=0.35)
    assert agree == pytest.approx(0.80, abs=0.03)          # agreement: barely moved
    assert info_s["weight_on_samples"] < info_a["weight_on_samples"]  # disagreement: more shrinkage
    raw_split = 1 / (1 + math.exp(-np.mean([math.log(p / (1 - p)) for p in (0.95, 0.40, 0.85)])))
    assert split < raw_split                                # pulled towards the prior


def test_hierarchical_aggregate_weights_models_equally():
    p, _ = adv.hierarchical_logit_aggregate({"a": [0.9, 0.9, 0.9, 0.9], "b": [0.3]}, prior_p=0.5, tau=10)
    q, _ = adv.hierarchical_logit_aggregate({"a": [0.9], "b": [0.3]}, prior_p=0.5, tau=10)
    assert p == pytest.approx(q, abs=1e-9)                 # 4 samples of model a don't outvote model b


def test_hierarchical_platt_learns_source_intercepts():
    rng = np.random.default_rng(0)
    probs, ys, srcs = [], [], []
    for src, shift in (("aib", 0.0), ("minibench", -1.0)):
        t = rng.beta(0.8, 1.2, 1500)
        y = rng.binomial(1, 1 / (1 + np.exp(-(np.log(t / (1 - t)) + shift))))
        probs += list(np.clip(t, 0.01, 0.99)); ys += list(y); srcs += [src] * 1500
    hp = adv.HierarchicalPlatt().fit(probs, ys, srcs)
    assert hp.b_source["minibench"] < hp.b_source["aib"] - 0.5
    assert hp.apply(0.5, "never_seen") == pytest.approx(1 / (1 + math.exp(-hp.b_global)))


def test_conditional_cap_and_break_even():
    assert adv.conditional_cap(0.999, auto_resolved=False) == 0.98
    assert adv.conditional_cap(0.999, auto_resolved=True) == 0.97
    assert adv.conditional_cap(0.999, False, verified_at_source=True, members_agree_extreme=True) == 0.995
    assert adv.conditional_cap(0.999, False, verified_at_source=True, members_agree_extreme=False) == 0.98
    assert adv.cap_break_even(0.02, 0.005) == pytest.approx(0.0107, abs=0.001)
    assert adv.cap_break_even(0.03, 0.01) == pytest.approx(0.0182, abs=0.001)


def test_evidence_update_groups_and_caps():
    items = [{"group": "g1", "odds_multiplier": 3.0}, {"group": "g1", "odds_multiplier": 2.5},  # same story twice
             {"group": "g2", "odds_multiplier": 0.5}]
    p, trace = adv.evidence_update(0.2, items)
    expected = 1 / (1 + math.exp(-(math.log(0.25) + math.log(3.0) + math.log(0.5))))
    assert p == pytest.approx(expected)
    assert {t["group"] for t in trace} == {"g1", "g2"}
    wild, _ = adv.evidence_update(0.2, [{"group": f"g{i}", "odds_multiplier": 100} for i in range(10)])
    assert wild == pytest.approx(1 / (1 + math.exp(-(math.log(0.25) + math.log(20)))))  # total shift capped
    with pytest.raises(ValueError):
        adv.evidence_update(0.5, [{"group": "x", "odds_multiplier": 0}])


def _fake_events(n=120, seed=3):
    rng = np.random.default_rng(seed)
    events, outcomes = [], {}
    for i in range(n):
        t = rng.beta(0.6, 1.2)
        y = int(rng.random() < t)
        raw = float(1 / (1 + np.exp(-(0.6 * np.log(t / (1 - t)) + 0.4))))
        cands = [{"model": f"m{j}", "probability": float(np.clip(raw + rng.normal(0, 0.05), 0.02, 0.98))} for j in range(3)]
        events.append({"event": "research_complete", "question_id": i, "timestamp": f"2026-09-{1 + i // 10:02d}T00:00:00"})
        events.append({"event": "reasoning_complete", "question_id": i, "question_type": "binary",
                       "timestamp": f"2026-09-{1 + i // 10:02d}T00:05:00", "initial_candidates": cands,
                       "final_candidates": cands, "submitted_value": raw,
                       "tournament": "minibench" if i % 2 else "aib"})
        outcomes[i] = y
    return events, outcomes


def test_adapter_builds_replayable_records():
    events, outcomes = _fake_events(10)
    recs = events_to_records(events, outcomes)
    assert len(recs) == 10 and all(r["qtype"] == "binary" for r in recs)
    assert sum(m["stage"] == "initial" for m in recs[0]["members"]) == 3
    assert recs[0]["outcome"] in (0, 1)


def test_end_to_end_events_to_calibration(tmp_path):
    events, outcomes = _fake_events(160)
    runs = tmp_path / "data" / "runs" / "2026" / "09"
    runs.mkdir(parents=True)
    with open(runs / "run-1.jsonl", "w") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")
    with open(tmp_path / "data" / "outcomes.jsonl", "w") as fh:
        for q, y in outcomes.items():
            fh.write(json.dumps({"question_id": q, "outcome": y}) + "\n")
    recs = tmp_path / "records.jsonl"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_replay_records.py"),
                    "--runs", str(tmp_path / "data" / "runs" / "**" / "*.jsonl"),
                    "--outcomes", str(tmp_path / "data" / "outcomes.jsonl"), "--out", str(recs)],
                   check=True, capture_output=True, text=True)
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "replay_and_fit.py"), str(recs),
                          "--out", str(tmp_path / "calibration.json"), "--min-n", "100"],
                         check=True, capture_output=True, text=True).stdout
    assert "median+hier_platt+cap" in out
    assert json.loads((tmp_path / "calibration.json").read_text())["n"] == 160
