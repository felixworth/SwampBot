import os
from unittest.mock import AsyncMock, patch

from adaptive_bot import AdaptiveForecastBot
from forecasting_tools import BinaryQuestion


def test_repeated_failures_are_skipped(tmp_path):
    env = {
        "BOT_MODELS": "openrouter/openai/gpt-4o",
        "BOT_RESEARCHER": "no_research",
        "BOT_ENABLE_REVIEW": "false",
        "BOT_ENABLE_DURABLE_LOGGING": "false",
        "BOT_FAILURE_PATH": str(tmp_path / "failures.json"),
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
