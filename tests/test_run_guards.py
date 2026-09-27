import json
import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from run_adaptive import credits_below_floor, key_remaining, repair_missing_comments


def test_credits_below_floor_stops_when_remaining_is_low(monkeypatch):
    monkeypatch.setenv("BOT_ENFORCE_SPEND_FLOOR", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("BOT_SPEND_FLOOR_USD", "1")
    response = MagicMock(status_code=200)
    response.json.return_value = {"data": {"limit_remaining": 0.2}}
    response.raise_for_status.return_value = None
    with patch("run_adaptive.requests.get", return_value=response):
        assert credits_below_floor() is True


def test_weekly_limit_stops_when_period_remaining_is_low(monkeypatch, caplog):
    monkeypatch.setenv("BOT_ENFORCE_SPEND_FLOOR", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("BOT_SPEND_FLOOR_USD", "1")
    monkeypatch.setenv("BOT_MODELS", "model-a,model-b")
    response = MagicMock(status_code=200)
    response.json.return_value = {"data": {
        "limit": 100,
        "usage": 10,
        "limit_remaining": 90,
        "limit_weekly": 20,
        "usage_weekly": 19.6,
        "limit_reset": "weekly",
    }}
    response.raise_for_status.return_value = None
    with caplog.at_level(logging.INFO):
        with patch("run_adaptive.requests.get", return_value=response):
            assert credits_below_floor() is True
    assert "limit_remaining" in caplog.text
    assert "usage_weekly" in caplog.text
    assert "limit_reset" in caplog.text


def test_reservation_stops_when_remaining_cannot_cover_analysts(monkeypatch):
    monkeypatch.setenv("BOT_ENFORCE_SPEND_FLOOR", "true")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("BOT_SPEND_FLOOR_USD", "1")
    response = MagicMock(status_code=200)
    response.json.return_value = {"data": {"limit_remaining": 1.2, "limit": 10, "usage": 8.8}}
    response.raise_for_status.return_value = None
    with patch("run_adaptive.requests.get", return_value=response):
        assert credits_below_floor(analysts=4) is True
        assert credits_below_floor(analysts=1) is False


def test_key_remaining_uses_limit_minus_usage_when_remaining_is_absent():
    remaining, used = key_remaining({"limit": 5, "usage": 4.2})
    assert abs(remaining - 0.8) < 1e-9
    assert used["limit"] == 5
    assert used["usage"] == 4.2


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
