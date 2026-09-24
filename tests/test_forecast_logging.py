import json
from datetime import datetime, timezone

from forecast_logging import (
    append_event,
    content_hash,
    safe_append_event,
    store_research,
    to_jsonable,
)


def test_jsonl_event_roundtrip(tmp_path):
    path = tmp_path / "logs" / "forecasts.jsonl"
    append_event(
        {
            "event": "test",
            "when": datetime(2026, 9, 24, tzinfo=timezone.utc),
            "prediction": [0.2, 0.8],
        },
        path,
    )
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["schema_version"] == 1
    assert rows[0]["event"] == "test"
    assert rows[0]["when"] == "2026-09-24T00:00:00+00:00"


def test_research_is_content_addressed_and_deduplicated(tmp_path):
    text = "Private research with dated sources."
    first = store_research(text, tmp_path)
    second = store_research(text, tmp_path)
    assert first == second == content_hash(text)
    assert (tmp_path / f"{first}.txt").read_text(encoding="utf-8") == text
    assert len(list(tmp_path.glob("*.txt"))) == 1


def test_logging_failure_never_raises(tmp_path):
    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("occupied", encoding="utf-8")
    safe_append_event({"event": "ignored"}, parent_file / "log.jsonl")


def test_unknown_objects_remain_json_serializable():
    class Unknown:
        pass

    json.dumps(to_jsonable({"value": Unknown()}))
