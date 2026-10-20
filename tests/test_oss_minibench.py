import asyncio
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from adaptive_bot import AdaptiveForecastBot
from forecasting_tools import BinaryQuestion
from run_adaptive import minibench_id, run


ROOT = Path(__file__).resolve().parents[1]


def _load_coverage():
    path = ROOT / "scripts" / "coverage_check.py"
    spec = importlib.util.spec_from_file_location("coverage_check_oss", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_minibench_id_override(monkeypatch):
    client = SimpleNamespace(CURRENT_MINIBENCH_ID="minibench")
    monkeypatch.delenv("BOT_MINIBENCH_ID", raising=False)
    assert minibench_id(client) == "minibench"
    monkeypatch.setenv("BOT_MINIBENCH_ID", "fall-mini-2026")
    assert minibench_id(client) == "fall-mini-2026"
    monkeypatch.setenv("BOT_MINIBENCH_ID", "  ")
    assert minibench_id(client) == "minibench"


def test_is_minibench_accepts_configured_slug(monkeypatch):
    question = BinaryQuestion(question_text="Mini?")
    question.tournament_slugs = ["fall-mini-2026"]
    monkeypatch.delenv("BOT_MINIBENCH_ID", raising=False)
    assert AdaptiveForecastBot._is_minibench(question) is False
    monkeypatch.setenv("BOT_MINIBENCH_ID", "fall-mini-2026")
    assert AdaptiveForecastBot._is_minibench(question) is True
    question.tournament_slugs = ["something-minibench"]
    monkeypatch.delenv("BOT_MINIBENCH_ID", raising=False)
    assert AdaptiveForecastBot._is_minibench(question) is True


def test_coverage_merges_and_labels_minibench(monkeypatch, tmp_path):
    coverage = _load_coverage()
    now = datetime.now(timezone.utc)
    old = now - timedelta(hours=5)
    main_q = SimpleNamespace(
        id_of_question=1, already_forecasted=False, open_time=old,
        question_text="Main", page_url="https://example/1",
    )
    mini_q = SimpleNamespace(
        id_of_question=2, already_forecasted=False, open_time=old,
        question_text="Mini", page_url="https://example/2",
    )
    duplicate = SimpleNamespace(
        id_of_question=1, already_forecasted=False, open_time=old,
        question_text="Duplicate", page_url="https://example/1b",
    )

    class Client:
        CURRENT_MINIBENCH_ID = "minibench"

        def get_all_open_questions_from_tournament(self, tournament_id):
            if tournament_id == "fall-futureeval-2026":
                return [main_q]
            if tournament_id == "custom-mini":
                return [duplicate, mini_q]
            raise AssertionError(tournament_id)

    summary = tmp_path / "summary.md"
    monkeypatch.setenv("BOT_TOURNAMENT_ID", "fall-futureeval-2026")
    monkeypatch.setenv("BOT_INCLUDE_MINIBENCH", "true")
    monkeypatch.setenv("BOT_MINIBENCH_ID", "custom-mini")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(coverage, "MetaculusClient", lambda: Client())
    monkeypatch.setattr(coverage, "load_failures", lambda path="data/failures.json": {})
    assert coverage.main() == 1
    text = summary.read_text(encoding="utf-8")
    assert "| 1 | main | Main |" in text
    assert "| 2 | minibench | Mini |" in text
    assert "Duplicate" not in text


def test_run_does_not_call_log_report_summary(monkeypatch):
    called = MagicMock()

    class FakeBot:
        def __init__(self, **kwargs):
            self.log_report_summary = called
            self._run_stats = {
                "attempted": 0, "succeeded": 0, "failed": 0,
                "skipped_already": 0, "skipped_failures": 0,
                "not_started_stop": 0, "not_started_budget": 0,
                "review_triggers": 0, "forecast_cost_usd": 0.0,
            }
            self.skip_previously_forecasted_questions = True

        def get_config(self):
            return {"durable_logging": False, "model_roster": ["m"]}

        async def forecast_on_tournament(self, tournament_id, return_exceptions=False):
            return []

    class FakeClient:
        CURRENT_MINIBENCH_ID = "minibench"
        CURRENT_METACULUS_CUP_ID = "cup"

    monkeypatch.setattr("run_adaptive.check_environment", lambda strict=True: None)
    monkeypatch.setattr("run_adaptive.credits_below_floor", lambda analysts=None: False)
    monkeypatch.setattr("run_adaptive.AdaptiveForecastBot", FakeBot)
    monkeypatch.setattr("run_adaptive.MetaculusClient", FakeClient)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    args = SimpleNamespace(
        mode="tournament",
        tournament_id="fall-futureeval-2026",
        include_minibench=True,
        limit=None,
        publish=False,
        reports_dir="logs/adaptive",
    )
    monkeypatch.setenv("BOT_MINIBENCH_ID", "custom-mini")
    assert asyncio.run(run(args)) == 0
    called.assert_not_called()
