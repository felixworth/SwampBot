"""List Fall tournament questions this bot has not forecast.

Read-only. Exits 1 when a question is missing or stuck after repeated failures,
so the scheduled workflow goes red.

usage:
    python scripts/coverage_check.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecasting_tools import MetaculusClient


def _opened_at(question):
    opened = getattr(question, "open_time", None)
    if opened is None or opened == "":
        return None
    if isinstance(opened, datetime):
        parsed = opened
    else:
        try:
            parsed = datetime.fromisoformat(str(opened).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def load_failures(path: str | Path = "data/failures.json") -> dict[str, int]:
    destination = Path(path)
    if not destination.exists():
        return {}
    try:
        data = json.loads(destination.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    counts = {}
    for key, value in data.items():
        try:
            counts[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return counts


def coverage_lists(questions, failures: dict[str, int], now: datetime | None = None):
    """Return (missing, stuck). Missing means open, unforecast, and old enough."""
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    cutoff = current - timedelta(hours=2)
    missing = []
    stuck = []
    for question in questions:
        question_id = str(getattr(question, "id_of_question", "") or "")
        if failures.get(question_id, 0) >= 2:
            stuck.append(question)
        if getattr(question, "already_forecasted", False):
            continue
        opened = _opened_at(question)
        if opened is not None and opened > cutoff:
            continue
        missing.append(question)
    return missing, stuck


def _cell(question, field):
    value = getattr(question, field, None)
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(missing, stuck) -> str:
    lines = ["## Coverage", ""]
    lines.append(f"Missing forecasts: {len(missing)}")
    lines.append("")
    lines.append("| question id | title | url | opened |")
    lines.append("| --- | --- | --- | --- |")
    if missing:
        for question in missing:
            lines.append(
                "| {id} | {title} | {url} | {opened} |".format(
                    id=_cell(question, "id_of_question"),
                    title=_cell(question, "question_text"),
                    url=_cell(question, "page_url"),
                    opened=_cell(question, "open_time"),
                )
            )
    else:
        lines.append("| | none | | |")
    lines.append("")
    lines.append(f"Skipped after repeated failures: {len(stuck)}")
    lines.append("")
    lines.append("| question id | title | url |")
    lines.append("| --- | --- | --- |")
    if stuck:
        for question in stuck:
            lines.append(
                "| {id} | {title} | {url} |".format(
                    id=_cell(question, "id_of_question"),
                    title=_cell(question, "question_text"),
                    url=_cell(question, "page_url"),
                )
            )
    else:
        lines.append("| | none | |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    tournament_id = os.getenv("BOT_TOURNAMENT_ID", "").strip()
    if not tournament_id:
        print("BOT_TOURNAMENT_ID is not set")
        return 1
    questions = MetaculusClient().get_all_open_questions_from_tournament(tournament_id)
    missing, stuck = coverage_lists(questions, load_failures())
    markdown = render_markdown(missing, stuck)
    print(markdown)
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(markdown)
    return 1 if missing or stuck else 0


if __name__ == "__main__":
    raise SystemExit(main())
