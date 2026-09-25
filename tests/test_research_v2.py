import asyncio
import json
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from research_v2 import (
    fetch_resolution_fact,
    format_research_brief,
    merge_claims,
    similar_questions,
    time_remaining_days,
    validate_cruxes,
    validate_resolution_spec,
)


def test_time_remaining_is_computed_in_code():
    close = datetime(2026, 10, 1, tzinfo=timezone.utc)
    now = datetime(2026, 9, 25, tzinfo=timezone.utc)
    assert time_remaining_days(close, now) == pytest.approx(6.0)


def test_validate_resolution_spec_adds_horizon():
    raw = json.dumps({
        "question_type": "binary",
        "archetype": "deadline_event",
        "target": "Event X happens",
        "resolution_sources": [{"name": "Gov", "url": "https://example.org", "kind": "webpage"}],
        "ambiguities": [],
    })
    spec = validate_resolution_spec(
        raw,
        close_time=datetime(2026, 10, 1, tzinfo=timezone.utc),
        now=datetime(2026, 9, 24, tzinfo=timezone.utc),
    )
    assert spec["time_remaining_days"] == pytest.approx(7.0)
    assert spec["target"] == "Event X happens"


def test_validate_cruxes_enforces_count():
    too_few = json.dumps({"cruxes": [
        {"id": "c1", "question": "A?", "why_it_matters": "x", "search_queries": ["a"]}
    ]})
    with pytest.raises(ValueError):
        validate_cruxes(too_few)
    raw = json.dumps({"cruxes": [
        {"id": "c1", "question": "A?", "why_it_matters": "x", "search_queries": ["a"]},
        {"id": "c2", "question": "B?", "why_it_matters": "y", "search_queries": ["b"]},
        {"id": "c3", "question": "C?", "why_it_matters": "z", "search_queries": ["c"]},
    ]})
    assert len(validate_cruxes(raw)) == 3


def test_merge_claims_collapses_same_group():
    merged = merge_claims([
        {"claim": "One", "independence_group": "g1", "source_url": "a",
         "published": None, "event_date": None, "crux_ids": ["c1"], "direction": "raises"},
        {"claim": "Copy", "independence_group": "g1", "source_url": "b",
         "published": None, "event_date": None, "crux_ids": ["c1"], "direction": "raises"},
        {"claim": "Other", "independence_group": "g2", "source_url": "c",
         "published": None, "event_date": None, "crux_ids": ["c2"], "direction": "lowers"},
    ])
    assert len(merged) == 2
    groups = {item["independence_group"] for item in merged}
    assert groups == {"g1", "g2"}


def test_similar_questions_uses_title_overlap():
    records = [
        {"title": "Will France leave the EU before 2027?", "outcome": 0,
         "question_id": 1, "final_submitted": 0.1},
        {"title": "Unrelated weather question", "outcome": 1, "question_id": 2},
    ]
    matches = similar_questions("Will Italy leave the EU before 2028?", records)
    assert matches and matches[0]["question_id"] == 1


def test_fetch_resolution_fact_records_failures(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr("research_v2.requests.get", fail)
    fact = fetch_resolution_fact("https://example.org/source")
    assert fact["value"] is None
    assert "fetch_failed" in fact["note"]


def test_brief_has_fixed_section_order():
    brief = format_research_brief(
        {"target": "X", "time_remaining_days": 10},
        [{"source": "example.org", "value": "fact", "as_of": "2026-09-25", "note": "fetched"}],
        [],
        [{"id": "c1", "question": "A?", "finding": "unknown"}],
        [{"claim": "One", "independence_group": "g1"}],
        ["Need turnout data"],
    )
    order = [brief.index(header) for header in (
        "## Resolution spec",
        "## Resolution facts",
        "## Outside view",
        "## Cruxes with findings",
        "## Claims by independence group",
        "## Open uncertainties",
    )]
    assert order == sorted(order)


def test_research_v2_builds_brief_with_two_providers(tmp_path, monkeypatch):
    from adaptive_bot import AdaptiveForecastBot
    from forecasting_tools import BinaryQuestion

    spec = {
        "question_type": "binary",
        "archetype": "deadline_event",
        "target": "Event X",
        "resolution_sources": [{"name": "Gov", "url": "https://example.org", "kind": "webpage"}],
        "ambiguities": [],
    }
    cruxes = {"cruxes": [
        {"id": "c1", "question": "A?", "why_it_matters": "x", "search_queries": ["query a"]},
        {"id": "c2", "question": "B?", "why_it_matters": "y", "search_queries": ["query b"]},
        {"id": "c3", "question": "C?", "why_it_matters": "z", "search_queries": ["query c"]},
    ]}
    claims = {"claims": [
        {"claim": "Same event", "source_url": "https://a.example", "independence_group": "g1",
         "crux_ids": ["c1"], "direction": "raises"},
        {"claim": "Copy", "source_url": "https://b.example", "independence_group": "g1",
         "crux_ids": ["c1"], "direction": "raises"},
    ]}

    async def exercise():
        with patch.dict(os.environ, {
            "BOT_MODELS": "openrouter/openai/gpt-4o",
            "BOT_RESEARCHER": "asknews/news-summaries",
            "BOT_RESEARCH_FALLBACK": "openrouter/google/gemini-3.8-flash:online",
            "BOT_ENABLE_REVIEW": "false",
            "BOT_ENABLE_DURABLE_LOGGING": "false",
            "BOT_RESEARCH_CACHE_DIR": str(tmp_path),
            "BOT_RESEARCH_MIN_CHARS": "50",
        }):
            bot = AdaptiveForecastBot(publish_reports_to_metaculus=False)
        bot._invoke_cheap = AsyncMock(side_effect=[
            json.dumps(spec), json.dumps(cruxes), json.dumps(claims),
        ])
        monkeypatch.setattr(
            "research_v2.fetch_resolution_fact",
            lambda url, timeout=15.0: {
                "source": "example.org", "value": "official table",
                "as_of": "2026-09-25", "note": "fetched", "url": url,
            },
        )
        bot._search = AsyncMock(
            return_value=(
                "Primary dated finding with enough independent detail to pass "
                "the usefulness check and a source at https://example.org/a"
            )
        )
        bot._invoke_research_route = AsyncMock(
            return_value=(
                "Fallback dated finding with enough independent detail to pass "
                "the usefulness check and a source at https://example.org/b"
            )
        )
        question = BinaryQuestion(
            question_text="Will Event X happen?",
            close_time=datetime(2026, 12, 1, tzinfo=timezone.utc),
        )
        brief = await bot.run_research(question)
        assert "## Resolution spec" in brief
        assert "## Cruxes with findings" in brief
        assert "Event X" in brief
        assert bot._invoke_cheap.await_count == 3

    asyncio.run(exercise())
