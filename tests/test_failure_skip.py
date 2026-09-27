import os
from unittest.mock import AsyncMock, patch

from adaptive_bot import AdaptiveForecastBot, is_provider_account_error
from forecasting_tools import BinaryQuestion


def test_rate_limit_is_not_a_provider_account_error():
    assert is_provider_account_error(RuntimeError("429 rate limit")) is False
    assert is_provider_account_error(RuntimeError("token limit exceeded")) is False
    assert is_provider_account_error(RuntimeError("maximum context length")) is False
    assert is_provider_account_error(RuntimeError("max_tokens is too large")) is False
    account = RuntimeError("invalid api key")
    account.status_code = 401
    assert is_provider_account_error(account) is True
    assert is_provider_account_error(RuntimeError("weekly limit reached")) is True
    assert is_provider_account_error(RuntimeError("insufficient credit for this request")) is True
    assert is_provider_account_error(RuntimeError("unauthorized")) is True


def test_repeated_failures_are_skipped(tmp_path):
    env = {
        "BOT_MODELS": "openrouter/openai/gpt-4o",
        "BOT_RESEARCHER": "no_research",
        "BOT_ENABLE_REVIEW": "false",
        "BOT_ENABLE_DURABLE_LOGGING": "false",
        "BOT_FAILURE_PATH": str(tmp_path / "failures.json"),
        "BOT_RETRY_FAILED": "",
    }
    with patch.dict(os.environ, env):
        bot = AdaptiveForecastBot(publish_reports_to_metaculus=False)
        bot._record_failure(9)
        bot._record_failure(9)
        question = BinaryQuestion(question_text="Stuck?", id_of_question=9)
        bot.forecast_question = AsyncMock()

        async def exercise():
            reports = await bot.forecast_questions([question])
            assert reports == []
            bot.forecast_question.assert_not_called()

        import asyncio
        asyncio.run(exercise())


def test_credit_error_is_not_recorded_and_stops_later_questions(tmp_path, caplog):
    env = {
        "BOT_MODELS": "openrouter/openai/gpt-4o",
        "BOT_RESEARCHER": "no_research",
        "BOT_ENABLE_REVIEW": "false",
        "BOT_ENABLE_DURABLE_LOGGING": "false",
        "BOT_FAILURE_PATH": str(tmp_path / "failures.json"),
        "BOT_QUESTION_CONCURRENCY": "1",
    }
    started = []

    async def fake_super(self, questions, return_exceptions=False):
        started.append(getattr(questions[0], "id_of_question", None))
        if questions[0].id_of_question == 1:
            return [RuntimeError("402 Payment Required: insufficient credits, weekly limit")]
        return [RuntimeError("should not start")]

    async def exercise():
        with patch.dict(os.environ, env):
            bot = AdaptiveForecastBot(publish_reports_to_metaculus=False)
            first = BinaryQuestion(question_text="First?", id_of_question=1)
            second = BinaryQuestion(question_text="Second?", id_of_question=2)
            with patch(
                "forecasting_tools.forecast_bots.forecast_bot.ForecastBot.forecast_questions",
                new=fake_super,
            ):
                reports = await bot.forecast_questions([first, second], return_exceptions=True)
            assert started == [1]
            assert len(reports) == 1
            assert is_provider_account_error(reports[0])
            assert bot._failure_count(1) == 0
            assert bot._provider_halt is True
            assert bot._run_stats["not_started_stop"] == 1
            again = await bot.forecast_questions([second], return_exceptions=True)
            assert again == []
            assert started == [1]

    import asyncio
    import logging
    with caplog.at_level(logging.ERROR):
        asyncio.run(exercise())
    assert "not recording a failure" in caplog.text
    assert "Not starting question 2" in caplog.text


def test_two_questions_run_together_and_a_third_waits_for_the_stop_flag(tmp_path):
    env = {
        "BOT_MODELS": "openrouter/openai/gpt-4o",
        "BOT_RESEARCHER": "no_research",
        "BOT_ENABLE_REVIEW": "false",
        "BOT_ENABLE_DURABLE_LOGGING": "false",
        "BOT_FAILURE_PATH": str(tmp_path / "failures.json"),
        "BOT_QUESTION_CONCURRENCY": "2",
    }
    started = []
    in_flight = 0
    peak = 0

    async def fake_super(self, questions, return_exceptions=False):
        nonlocal in_flight, peak
        question_id = questions[0].id_of_question
        started.append(question_id)
        in_flight += 1
        peak = max(peak, in_flight)
        if question_id == 1:
            await asyncio.sleep(0.05)
            in_flight -= 1
            return [RuntimeError("402 Payment Required: insufficient credits")]
        await asyncio.sleep(0.15)
        in_flight -= 1
        return [SimpleNamespace()]

    async def exercise():
        with patch.dict(os.environ, env):
            bot = AdaptiveForecastBot(publish_reports_to_metaculus=False)
            questions = [
                BinaryQuestion(question_text=f"Q{i}?", id_of_question=i)
                for i in (1, 2, 3)
            ]
            with patch(
                "forecasting_tools.forecast_bots.forecast_bot.ForecastBot.forecast_questions",
                new=fake_super,
            ):
                await bot.forecast_questions(questions, return_exceptions=True)
        assert peak == 2
        assert set(started) == {1, 2}
        assert bot._run_stats["not_started_stop"] == 1
        assert bot._failure_count(1) == 0

    import asyncio
    from types import SimpleNamespace
    asyncio.run(exercise())


def test_time_budget_does_not_start_questions(tmp_path):
    env = {
        "BOT_MODELS": "openrouter/openai/gpt-4o",
        "BOT_RESEARCHER": "no_research",
        "BOT_ENABLE_REVIEW": "false",
        "BOT_ENABLE_DURABLE_LOGGING": "false",
        "BOT_FAILURE_PATH": str(tmp_path / "failures.json"),
        "BOT_RUN_BUDGET_MINUTES": "0",
    }

    async def fake_super(self, questions, return_exceptions=False):
        raise AssertionError("question was started")

    async def exercise():
        with patch.dict(os.environ, env):
            bot = AdaptiveForecastBot(publish_reports_to_metaculus=False)
            questions = [
                BinaryQuestion(question_text="Late?", id_of_question=1),
                BinaryQuestion(question_text="Later?", id_of_question=2),
            ]
            with patch(
                "forecasting_tools.forecast_bots.forecast_bot.ForecastBot.forecast_questions",
                new=fake_super,
            ):
                reports = await bot.forecast_questions(questions, return_exceptions=True)
        assert reports == []
        assert bot._run_stats["not_started_budget"] == 2
        assert bot._run_stats["attempted"] == 0

    import asyncio
    asyncio.run(exercise())
