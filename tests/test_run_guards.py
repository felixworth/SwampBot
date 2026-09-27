import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from run_adaptive import credits_below_floor, repair_missing_comments


def test_credits_below_floor_stops_when_remaining_is_low(monkeypatch):
    monkeypatch.setenv("BOT_ENFORCE_SPEND_FLOOR", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("BOT_SPEND_FLOOR_USD", "1")
    response = MagicMock(status_code=200)
    response.json.return_value = {"data": {"limit_remaining": 0.2}}
    response.raise_for_status.return_value = None
    with patch("run_adaptive.requests.get", return_value=response):
        assert credits_below_floor() is True


def test_repair_posts_when_recent_forecast_has_no_comment(tmp_path, monkeypatch):
    log = tmp_path / "forecasts.jsonl"
    log.write_text(json.dumps({
        "event": "forecast_reasoning_complete",
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "post_id": 7,
        "explanation": "Recovered comment",
    }) + "\n", encoding="utf-8")
    monkeypatch.setenv("BOT_LOG_PATH", str(log))
    monkeypatch.setenv("METACULUS_TOKEN", "token")
    response = MagicMock(status_code=200)
    response.json.return_value = {"results": []}
    client = MagicMock()
    with patch("run_adaptive.requests.get", return_value=response):
        assert repair_missing_comments(client) == 1
    client.post_question_comment.assert_called_once_with(7, "Recovered comment")
