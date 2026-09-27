import asyncio
import importlib.util
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from adaptive_bot import AdaptiveForecastBot, assign_research_route
from forecast_logic import ForecastEngine, review_reasons
from forecast_upgrades.advanced import hierarchical_logit_aggregate
from forecasting_tools import BinaryQuestion
from research_v2 import validate_cruxes
from upgrade_config import UPGRADE


ROOT = Path(__file__).resolve().parents[1]
SPEC = {"kind": "binary", "question_text": "Will X happen?"}


def _load_coverage():
    path = ROOT / "scripts" / "coverage_check.py"
    spec = importlib.util.spec_from_file_location("coverage_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(probability, model=None):
    data = {
        "probability": probability,
        "rationale": "Dated source supports the event.",
        "baseline": "Reference class",
        "key_uncertainty": "Remaining prerequisite",
        "needs_review": False,
    }
    if model:
        data["model"] = model
    return data


def _cruxes(queries):
    rows = []
    for index, query in enumerate(queries, start=1):
        rows.append({
            "id": f"c{index}",
            "question": f"Q{index}?",
            "why_it_matters": "it moves the forecast",
            "search_queries": query if isinstance(query, list) else [query],
        })
    return json.dumps({"cruxes": rows})


def test_validate_cruxes_keeps_one_query():
    cruxes = validate_cruxes(_cruxes([
        ["first query", "second query"],
        ["only"],
        ["kept", "dropped"],
    ]))
    assert [item["search_queries"] for item in cruxes] == [
        ["first query"],
        ["only"],
        ["kept"],
    ]


def test_first_crux_uses_the_web_model():
    primary = "asknews/news-summaries"
    fallback = "openrouter/openai/gpt-6-sol:online"
    route, used = assign_research_route(0, primary, fallback)
    assert route == fallback and used == 1
    route, used = assign_research_route(used, primary, fallback)
    assert route == primary and used == 1
    assert int(UPGRADE["research_web_queries"]) == 1


def test_research_v2_routes_only_the_first_crux_to_the_web_model(tmp_path, monkeypatch):
    spec = {
        "question_type": "binary",
        "archetype": "deadline_event",
        "target": "Event X",
        "resolution_sources": [],
        "ambiguities": [],
    }
    useful = (
        "Primary dated finding with enough independent detail to pass "
        "the usefulness check and a source at https://example.org/a"
    )

    async def exercise():
        with patch.dict(os.environ, {
            "BOT_MODELS": "openrouter/openai/gpt-4o",
            "BOT_RESEARCHER": "asknews/news-summaries",
            "BOT_RESEARCH_FALLBACK": "openrouter/openai/gpt-6-sol:online",
            "BOT_ENABLE_REVIEW": "false",
            "BOT_ENABLE_DURABLE_LOGGING": "false",
            "BOT_RESEARCH_CACHE_DIR": str(tmp_path),
            "BOT_RESEARCH_MIN_CHARS": "50",
        }):
            bot = AdaptiveForecastBot(publish_reports_to_metaculus=False)
        bot._invoke_cheap = AsyncMock(side_effect=[
            json.dumps(spec),
            _cruxes(["query a", "query b", "query c"]),
            json.dumps({"claims": []}),
        ])
        monkeypatch.setattr(
            "research_v2.fetch_resolution_fact",
            lambda url, timeout=15.0: {"source": "x", "value": None, "note": "none"},
        )
        seen = []

        async def named(route, prompt):
            seen.append(bot._research_route_label(route))
            return useful

        bot._search_named_route = named
        question = BinaryQuestion(question_text="Will Event X happen?")
        await bot.run_research(question)
        assert seen[0] == "openrouter/openai/gpt-6-sol:online"
        assert seen[1:] == ["asknews/news-summaries", "asknews/news-summaries"]

    asyncio.run(exercise())


def test_model_label_survives_revision():
    async def invoke(index, prompt):
        if "Audit these forecasts" in prompt:
            return json.dumps({"critique": "Check the date.", "search_query": ""})
        probability = 0.15 if index == 0 else 0.85
        if "Your own original forecast" in prompt:
            probability = 0.2 if index == 0 else 0.8
        return json.dumps(_candidate(probability))

    async def search(prompt):
        return "dated source https://example.org"

    async def exercise():
        engine = ForecastEngine(
            invoke, search, analyst_count=2, labels=["alpha", "beta"], disagreement=0.2,
        )
        result = await engine.run(SPEC, "sources", time.monotonic() + 120)
        assert result.review_triggered is True
        assert [item["model"] for item in result.initial_candidates] == ["alpha", "beta"]
        assert [item["model"] for item in result.candidates] == ["alpha", "beta"]
        assert result.candidates[0]["probability"] == pytest.approx(0.2)

    import time
    asyncio.run(exercise())


def test_aggregate_groups_by_model_when_one_sample_is_missing():
    with patch.dict(os.environ, {
        "BOT_MODELS": "model-a,model-b",
        "BOT_RESEARCHER": "no_research",
        "BOT_ENABLE_REVIEW": "false",
        "BOT_ENABLE_DURABLE_LOGGING": "false",
    }):
        bot = AdaptiveForecastBot(publish_reports_to_metaculus=False)
    question = BinaryQuestion(question_text="Grouped?")
    candidates = [
        _candidate(0.9, "model-a"),
        _candidate(0.8, "model-a"),
        _candidate(0.1, "model-b"),
    ]
    raw, _final, _stages = bot._aggregate(candidates, question)
    expected, _info = hierarchical_logit_aggregate(
        {"model-a": [0.9, 0.8], "model-b": [0.1]},
        prior_p=UPGRADE["shrink_prior_p"],
        tau=UPGRADE["shrink_tau"],
    )
    assert raw == pytest.approx(expected)


def test_review_reasons_use_per_model_means():
    wide_samples = [
        _candidate(0.1, "alpha"),
        _candidate(0.4, "alpha"),
        _candidate(0.3, "beta"),
        _candidate(0.32, "beta"),
    ]
    assert review_reasons(wide_samples, SPEC, "sources", 0.2) == []
    means_disagree = [
        _candidate(0.2, "alpha"),
        _candidate(0.2, "alpha"),
        _candidate(0.7, "beta"),
        _candidate(0.7, "beta"),
    ]
    reasons = review_reasons(means_disagree, SPEC, "sources", 0.2)
    assert any("disagreement" in reason for reason in reasons)
    split_samples = [
        _candidate(0.99, "alpha"),
        _candidate(0.80, "alpha"),
        _candidate(0.60, "beta"),
        _candidate(0.70, "beta"),
    ]
    assert not any("Split extreme" in reason for reason in review_reasons(split_samples, SPEC, "sources", 0.2))
    unlabeled = [_candidate(0.1), _candidate(0.4)]
    assert any("disagreement" in reason for reason in review_reasons(unlabeled, SPEC, "sources", 0.2))


def test_coverage_check_exit_codes(tmp_path, monkeypatch):
    coverage = _load_coverage()
    now = datetime.now(timezone.utc)
    old = SimpleNamespace(
        id_of_question=1, already_forecasted=False,
        open_time=now - timedelta(hours=3), question_text="Old", page_url="https://example/1",
    )
    fresh = SimpleNamespace(
        id_of_question=2, already_forecasted=False,
        open_time=now - timedelta(minutes=30), question_text="Fresh", page_url="https://example/2",
    )
    done = SimpleNamespace(
        id_of_question=3, already_forecasted=True,
        open_time=now - timedelta(days=2), question_text="Done", page_url="https://example/3",
    )
    stuck = SimpleNamespace(
        id_of_question=4, already_forecasted=True,
        open_time=now - timedelta(days=1), question_text="Stuck", page_url="https://example/4",
    )

    class Client:
        def __init__(self, questions):
            self.questions = questions

        def get_all_open_questions_from_tournament(self, tournament_id):
            assert tournament_id == "fall-futureeval-2026"
            return self.questions

    summary = tmp_path / "summary.md"
    monkeypatch.setenv("BOT_TOURNAMENT_ID", "fall-futureeval-2026")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(coverage, "MetaculusClient", lambda: Client([done, fresh]))
    monkeypatch.setattr(coverage, "load_failures", lambda path="data/failures.json": {})
    assert coverage.main() == 0
    assert "none" in summary.read_text(encoding="utf-8")

    summary.write_text("", encoding="utf-8")
    monkeypatch.setattr(coverage, "MetaculusClient", lambda: Client([old, done]))
    assert coverage.main() == 1

    summary.write_text("", encoding="utf-8")
    monkeypatch.setattr(coverage, "MetaculusClient", lambda: Client([stuck, done, fresh]))
    monkeypatch.setattr(coverage, "load_failures", lambda path="data/failures.json": {"4": 2})
    assert coverage.main() == 1
    assert "Stuck" in summary.read_text(encoding="utf-8")


def test_retry_failed_ignores_the_skip_and_clears_on_success(tmp_path):
    env = {
        "BOT_MODELS": "openrouter/openai/gpt-4o",
        "BOT_RESEARCHER": "no_research",
        "BOT_ENABLE_REVIEW": "false",
        "BOT_ENABLE_DURABLE_LOGGING": "false",
        "BOT_FAILURE_PATH": str(tmp_path / "failures.json"),
        "BOT_RETRY_FAILED": "9",
    }
    async def succeed(self, questions, return_exceptions=False):
        return [SimpleNamespace(ok=True) for _ in questions]

    async def exercise():
        with patch.dict(os.environ, env):
            bot = AdaptiveForecastBot(publish_reports_to_metaculus=False)
            bot._record_failure(9)
            bot._record_failure(9)
            question = BinaryQuestion(question_text="Retry?", id_of_question=9)
            with patch(
                "forecasting_tools.forecast_bots.forecast_bot.ForecastBot.forecast_questions",
                new=succeed,
            ):
                reports = await bot.forecast_questions([question])
            assert len(reports) == 1
            assert bot._failure_count(9) == 0
            assert bot._run_stats["attempted"] == 1
            assert bot._run_stats["succeeded"] == 1

    asyncio.run(exercise())
