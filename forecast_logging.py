"""Failure-safe, replayable logging for adaptive forecasts.

Logs are private local files (and private GitHub Actions artifacts). Logging
errors are reported but never allowed to fail a forecast.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from upgrade_config import UPGRADE

logger = logging.getLogger(__name__)
_write_lock = threading.Lock()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if hasattr(value, "to_dict"):
        return to_jsonable(value.to_dict())
    return repr(value)


def question_fields(question: Any) -> dict[str, Any]:
    return {
        "question_id": getattr(question, "id_of_question", None),
        "post_id": getattr(question, "id_of_post", None),
        "url": getattr(question, "page_url", None),
        "title": getattr(question, "question_text", None),
        "question_type": getattr(question, "question_type", None),
        "close_time": to_jsonable(getattr(question, "close_time", None)),
        "tournaments": to_jsonable(getattr(question, "tournament_slugs", [])),
    }


def append_event(event: dict[str, Any], path: str | Path | None = None) -> None:
    destination = Path(path or UPGRADE["log_path"])
    payload = {
        "schema_version": 1,
        "logged_at": datetime.now(timezone.utc).isoformat(),
        **to_jsonable(event),
    }
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    with _write_lock:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def safe_append_event(event: dict[str, Any], path: str | Path | None = None) -> None:
    try:
        append_event(event, path)
    except Exception:
        logger.exception("Forecast logging failed; continuing without the log event")


def store_research(
    text: str, directory: str | Path | None = None
) -> str | None:
    try:
        digest = content_hash(text)
        root = Path(directory or UPGRADE["research_dir"])
        destination = root / f"{digest}.txt"
        if not destination.exists():
            root.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, destination)
        return digest
    except Exception:
        logger.exception("Research persistence failed; continuing without stored text")
        return None


def log_report_batch(
    reports: Iterable[Any],
    *,
    run_mode: str,
    published: bool,
    model_roster: list[str],
    review_triggers: int = 0,
    run_stats: dict[str, Any] | None = None,
) -> None:
    """Log final SDK reports, including total tracked cost and latency."""
    for report in reports:
        if isinstance(report, BaseException):
            safe_append_event(
                {
                    "event": "report_failed",
                    "run_mode": run_mode,
                    "published": published,
                    "model_roster": model_roster,
                    "error_type": type(report).__name__,
                    "error": str(report)[:4000],
                }
            )
            continue

        question = getattr(report, "question", None)
        safe_append_event(
            {
                "event": "report_complete",
                "run_mode": run_mode,
                "published": published,
                "model_roster": model_roster,
                **question_fields(question),
                "prediction": to_jsonable(getattr(report, "prediction", None)),
                "comment_hash": content_hash(
                    str(getattr(report, "explanation", ""))
                ),
                "cost_usd": getattr(report, "price_estimate", None),
                "latency_minutes": getattr(report, "minutes_taken", None),
                "minor_errors": to_jsonable(getattr(report, "errors", [])),
            }
        )
    safe_append_event(
        {
            "event": "run_summary",
            "run_mode": run_mode,
            "published": published,
            "model_roster": model_roster,
            "review_triggers": review_triggers,
            **(run_stats or {}),
        }
    )
