import json
import os
from types import SimpleNamespace
from unittest.mock import patch

from adaptive_bot import AdaptiveForecastBot
from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion


def make_bot(**overrides):
    env = {
        "BOT_MODELS": "openrouter/openai/gpt-4o",
        "BOT_RESEARCHER": "no_research",
        "BOT_ENABLE_REVIEW": "false",
        "BOT_ENABLE_DURABLE_LOGGING": "false",
    }
    env.update(overrides)
    with patch.dict(os.environ, env):
        return AdaptiveForecastBot(publish_reports_to_metaculus=False)


def test_binary_uses_median_then_caps():
    bot = make_bot()
    question = BinaryQuestion(question_text="Cap me?")
    candidates = [
        {"probability": 0.01, "rationale": "a", "baseline": "b",
         "key_uncertainty": "c", "needs_review": False},
        {"probability": 0.99, "rationale": "a", "baseline": "b",
         "key_uncertainty": "c", "needs_review": False},
        {"probability": 0.4, "rationale": "a", "baseline": "b",
         "key_uncertainty": "c", "needs_review": False},
    ]
    raw, final = bot._aggregate(candidates, question)
    assert raw == 0.4
    assert final == 0.4
    raw_extreme, final_extreme = bot._aggregate(candidates[:2], question)
    assert raw_extreme == 0.5
    assert 0.02 <= final_extreme <= 0.98


def test_minibench_uses_tighter_caps():
    bot = make_bot()
    question = BinaryQuestion(question_text="Mini?")
    question.tournament_slugs = ["minibench"]
    raw, final = bot._aggregate(
        [{"probability": 0.999, "rationale": "a", "baseline": "b",
          "key_uncertainty": "c", "needs_review": False}],
        question,
    )
    assert raw == 0.999
    assert final == 0.97


def test_mc_applies_floor():
    bot = make_bot()
    question = MultipleChoiceQuestion(question_text="Which?", options=["A", "B", "C"])
    candidates = [
        {"probabilities": {"A": 1.0, "B": 0.0, "C": 0.0},
         "rationale": "a", "baseline": "b", "key_uncertainty": "c",
         "needs_review": False},
        {"probabilities": {"A": 1.0, "B": 0.0, "C": 0.0},
         "rationale": "a", "baseline": "b", "key_uncertainty": "c",
         "needs_review": False},
    ]
    raw, final = bot._aggregate(candidates, question)
    assert min(raw) >= 0.01 - 1e-12
    mapping = final.to_dict()
    assert min(mapping.values()) >= 0.01 - 1e-12
    assert abs(sum(mapping.values()) - 1) < 1e-9


def test_platt_applies_only_when_marked_applied(tmp_path, monkeypatch):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"a": 2.0, "b": -0.5, "applied": True}), encoding="utf-8")
    monkeypatch.setitem(
        __import__("upgrade_config", fromlist=["UPGRADE"]).UPGRADE,
        "calibration_path",
        str(path),
    )
    bot = make_bot()
    question = BinaryQuestion(question_text="Calibrate?")
    raw, final = bot._aggregate(
        [{"probability": 0.7, "rationale": "a", "baseline": "b",
          "key_uncertainty": "c", "needs_review": False}],
        question,
    )
    assert raw == 0.7
    assert final != 0.7
    assert 0.02 <= final <= 0.98
