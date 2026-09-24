import asyncio
import os
from unittest.mock import AsyncMock, patch

from adaptive_bot import AdaptiveForecastBot


GOOD_PRIMARY = (
    "A sufficiently detailed dated research response with independent facts "
    "and a source at https://example.org/primary."
)
GOOD_FALLBACK = (
    "A sufficiently detailed fallback research response with dated facts and "
    "a source at https://example.org/fallback."
)


def make_bot(tmp_path, **overrides):
    env = {
        "BOT_MODELS": "openrouter/openai/gpt-4o",
        "BOT_RESEARCHER": "asknews/news-summaries",
        "BOT_RESEARCH_FALLBACK": "openrouter/google/gemini-3.8-flash:online",
        "BOT_RESEARCH_RETRY_COUNT": "1",
        "BOT_RESEARCH_RETRY_DELAY_SECONDS": "0",
        "BOT_RESEARCH_MIN_CHARS": "50",
        "BOT_RESEARCH_CACHE_DIR": str(tmp_path),
        "BOT_RESEARCH_CACHE_TTL_SECONDS": "3600",
        "BOT_RESEARCH_BREAKER_FAILURES": "2",
        "BOT_ENABLE_REVIEW": "false",
        "BOT_ENABLE_DURABLE_LOGGING": "false",
    }
    env.update(overrides)
    with patch.dict(os.environ, env):
        return AdaptiveForecastBot(publish_reports_to_metaculus=False)


def test_transient_primary_failure_retries_once(tmp_path):
    async def exercise():
        bot = make_bot(tmp_path)
        bot._invoke_research_route = AsyncMock(
            side_effect=[TimeoutError("temporary timeout"), GOOD_PRIMARY]
        )
        result = await bot._search("same prompt")
        assert result == GOOD_PRIMARY
        assert bot._invoke_research_route.await_count == 2

    asyncio.run(exercise())


def test_successful_research_is_cached(tmp_path):
    async def exercise():
        bot = make_bot(tmp_path)
        bot._invoke_research_route = AsyncMock(return_value=GOOD_PRIMARY)
        first = await bot._search("cache this prompt")
        second = await bot._search("cache this prompt")
        assert first == second == GOOD_PRIMARY
        assert bot._invoke_research_route.await_count == 1

    asyncio.run(exercise())


def test_primary_failure_uses_fallback(tmp_path):
    async def exercise():
        bot = make_bot(tmp_path)
        routes = []

        async def invoke(route, prompt):
            routes.append(str(route))
            if str(route).startswith("asknews/"):
                raise RuntimeError("503 temporary provider failure")
            return GOOD_FALLBACK

        bot._invoke_research_route = invoke
        result = await bot._search("fallback prompt")
        assert result.startswith("RESEARCH_PROVIDER_FALLBACK:")
        assert GOOD_FALLBACK in result
        assert len(routes) == 3  # two primary attempts, then one fallback

    asyncio.run(exercise())


def test_auth_failure_does_not_retry_primary(tmp_path):
    async def exercise():
        bot = make_bot(tmp_path)
        routes = []

        async def invoke(route, prompt):
            routes.append(str(route))
            if str(route).startswith("asknews/"):
                raise RuntimeError("401 unauthorized API key")
            return GOOD_FALLBACK

        bot._invoke_research_route = invoke
        result = await bot._search("auth failure prompt")
        assert result.startswith("RESEARCH_PROVIDER_FALLBACK:")
        assert len(routes) == 2  # primary once, fallback once

    asyncio.run(exercise())


def test_empty_primary_response_falls_back(tmp_path):
    async def exercise():
        bot = make_bot(tmp_path)
        bot._invoke_research_route = AsyncMock(
            side_effect=["empty", "still empty", GOOD_FALLBACK]
        )
        result = await bot._search("empty response prompt")
        assert GOOD_FALLBACK in result
        assert bot._invoke_research_route.await_count == 3

    asyncio.run(exercise())


def test_all_provider_failures_return_explicit_marker(tmp_path):
    async def exercise():
        bot = make_bot(tmp_path)
        bot._invoke_research_route = AsyncMock(
            side_effect=RuntimeError("503 all providers unavailable")
        )
        result = await bot._search("all fail prompt")
        assert result.startswith("RESEARCH_FAILED:")
        assert "503" in result

    asyncio.run(exercise())


def test_circuit_breaker_skips_repeated_primary_failures(tmp_path):
    async def exercise():
        bot = make_bot(
            tmp_path,
            BOT_RESEARCH_BREAKER_FAILURES="1",
            BOT_RESEARCH_BREAKER_SECONDS="300",
        )
        routes = []

        async def invoke(route, prompt):
            routes.append(str(route))
            if str(route).startswith("asknews/"):
                raise RuntimeError("503 temporary provider failure")
            return GOOD_FALLBACK

        bot._invoke_research_route = invoke
        await bot._search("first breaker prompt")
        await bot._search("different prompt bypasses cache")
        primary_calls = [
            route for route in routes if route.startswith("asknews/")
        ]
        fallback_calls = [
            route for route in routes if not route.startswith("asknews/")
        ]
        assert len(primary_calls) == 2  # retry on first prompt only
        assert len(fallback_calls) == 2

    asyncio.run(exercise())
