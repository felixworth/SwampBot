"""Phase 0 smoke tests: every supported leaf type completes without API writes."""
import asyncio
import json
import os
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

from adaptive_bot import AdaptiveForecastBot
from forecasting_tools import (
    BinaryQuestion,
    DateQuestion,
    DiscreteQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecast_logic import QUANTILES
from upgrade_config import feature_flags


def _payload(kind: str) -> dict:
    data = {
        "rationale": "Canned offline evidence.",
        "baseline": "Canned reference class.",
        "key_uncertainty": "Canned uncertainty.",
        "needs_review": False,
    }
    if kind == "binary":
        data["probability"] = 0.6
    elif kind == "multiple_choice":
        data["probabilities"] = {"A": 0.6, "B": 0.4}
    elif kind in ("numeric", "discrete"):
        upper = 8 if kind == "discrete" else 100
        data["quantiles"] = [
            {"percentile": q, "value": q * upper} for q in QUANTILES
        ]
    else:
        start = datetime(2028, 1, 1, tzinfo=timezone.utc)
        data["quantiles"] = [
            {
                "percentile": q,
                "value": (start + timedelta(days=q * 365)).isoformat(),
            }
            for q in QUANTILES
        ]
    return data


QUESTIONS = [
    (
        "binary",
        BinaryQuestion(question_text="Test binary?", id_of_post=1, id_of_question=2),
    ),
    (
        "multiple_choice",
        MultipleChoiceQuestion(question_text="Which?", options=["A", "B"]),
    ),
    (
        "numeric",
        NumericQuestion(
            question_text="How much?",
            lower_bound=0,
            upper_bound=100,
            open_lower_bound=False,
            open_upper_bound=False,
            zero_point=None,
        ),
    ),
    (
        "discrete",
        DiscreteQuestion(
            question_text="How many?",
            lower_bound=0,
            upper_bound=8,
            open_lower_bound=False,
            open_upper_bound=False,
            zero_point=None,
            cdf_size=9,
        ),
    ),
    (
        "date",
        DateQuestion(
            question_text="When?",
            lower_bound=datetime(2028, 1, 1, tzinfo=timezone.utc),
            upper_bound=datetime(2029, 1, 1, tzinfo=timezone.utc),
            open_lower_bound=False,
            open_upper_bound=False,
            zero_point=None,
        ),
    ),
]


@pytest.mark.parametrize(("kind", "question"), QUESTIONS, ids=[q[0] for q in QUESTIONS])
def test_full_dry_run_pipeline_never_posts(kind, question):
    async def exercise():
        with patch.dict(
            os.environ,
            {
                "BOT_MODELS": "openrouter/openai/gpt-4o",
                "BOT_ENABLE_REVIEW": "false",
                "BOT_RESEARCHER": "no_research",
                "BOT_ENABLE_DURABLE_LOGGING": "false",
            },
        ):
            bot = AdaptiveForecastBot(
                publish_reports_to_metaculus=False,
                skip_previously_forecasted_questions=False,
                extra_metadata_in_explanation=True,
            )
            bot._engine.invoke = AsyncMock(
                return_value=json.dumps(_payload(kind))
            )
            bot._search = AsyncMock(
                return_value="Offline source, https://example.org, 2026-09-24"
            )

            method_names = (
                "post_binary_question_prediction",
                "post_multiple_choice_question_prediction",
                "post_numeric_question_prediction",
                "post_question_comment",
            )
            with ExitStack() as stack:
                post_mocks = [
                    stack.enter_context(
                        patch.object(bot.metaculus_client, method_name)
                    )
                    for method_name in method_names
                ]
                report = await bot.forecast_question(question)

            assert report.prediction is not None
            assert "Adaptive forecast" in report.explanation
            assert bot.get_config()["upgrade_flags"] == feature_flags()
            assert bot._engine.invoke.await_count == 1
            assert bot._search.await_count == 2
            for post_mock in post_mocks:
                post_mock.assert_not_called()

    asyncio.run(exercise())


def test_upgrade_features_are_loaded_but_inert():
    flags = feature_flags()
    assert flags["FF_CALIBRATION"] is True
    assert flags["FF_RESOLUTION_SPEC"] is True
    assert flags["FF_RESEARCH_V2"] is True
    assert flags["FF_NUMERIC_TAILS"] is True
    assert flags["FF_LOGIT_SHRINK"] is True
    assert flags["FF_CONDITIONAL_CAP"] is True
    assert flags["FF_SEPARATE_REVIEWER"] is True
    assert flags["FF_ROLE_ROTATION"] is True
    assert flags["FF_ENSEMBLE_V2"] is False
    assert flags["FF_STRUCTURED_MODELS"] is False
