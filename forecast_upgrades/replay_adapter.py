"""Bridge the live event log to the replay/calibration format.

The live logger appends events (e.g. "reasoning complete" with question_type,
initial_candidates, final_candidates, submitted value); replay_and_fit.py expects one
record per question with qtype, submitted_at, members[stage] and outcome.
Edit FIELD_MAP to match the logger's real field names; nothing else should need changing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

FIELD_MAP = {
    "event_name": "event",                   # field holding the event type
    "reasoning_event": "reasoning_complete",  # value marking the event with forecasts
    "question_id": "question_id",
    "qtype": "question_type",
    "timestamp": "timestamp",
    "initial": "initial_candidates",
    "final": "final_candidates",
    "submitted": "submitted_value",
    "tournament": "tournament",
    "archetype": "archetype",
}

_BINARY_KEYS = ("probability", "prediction", "forecast", "p", "value")


def _binary_value(candidate: Any) -> float | None:
    if isinstance(candidate, (int, float)):
        return float(candidate)
    if isinstance(candidate, Mapping):
        for k in _BINARY_KEYS:
            v = candidate.get(k)
            if isinstance(v, (int, float)):
                return float(v)
    return None


def load_jsonl(paths: Iterable[str | Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    return rows


def normalize_live_event(event: Mapping) -> dict:
    """Map this bot's forecast_reasoning_complete rows onto FIELD_MAP."""
    ev = dict(event)
    if ev.get("event") == "forecast_reasoning_complete":
        ev["event"] = FIELD_MAP["reasoning_event"]
        ev.setdefault("timestamp", ev.get("logged_at"))
        ev.setdefault("submitted_value", ev.get("final_submitted"))
        tournaments = ev.get("tournaments") or []
        if isinstance(tournaments, list) and tournaments:
            ev.setdefault("tournament", tournaments[0])
    return ev


def events_to_records(events: Iterable[Mapping], outcomes: Mapping[Any, Any], fmap: Mapping[str, str] = FIELD_MAP) -> list[dict]:
    """One record per question (latest reasoning event wins), outcomes joined by question id."""
    latest: dict[Any, Mapping] = {}
    for ev in events:
        if ev.get(fmap["event_name"]) != fmap["reasoning_event"]:
            continue
        qid = ev.get(fmap["question_id"])
        if qid is None:
            continue
        if qid not in latest or str(ev.get(fmap["timestamp"], "")) >= str(latest[qid].get(fmap["timestamp"], "")):
            latest[qid] = ev
    records = []
    for qid, ev in latest.items():
        qtype = ev.get(fmap["qtype"])
        members = []
        for stage, key in (("initial", fmap["initial"]), ("revised", fmap["final"])):
            for i, c in enumerate(ev.get(key) or []):
                members.append({"stage": stage, "sample_idx": i,
                                "model": c.get("model") if isinstance(c, Mapping) else None,
                                "binary": _binary_value(c) if qtype == "binary" else None})
        records.append({
            "question_id": qid,
            "qtype": qtype,
            "submitted_at": str(ev.get(fmap["timestamp"], "")),
            "segment": ev.get(fmap["archetype"]),
            "source": ev.get(fmap["tournament"]),
            "final_submitted": ev.get(fmap["submitted"]),
            "members": members,
            "outcome": outcomes.get(qid),
        })
    records.sort(key=lambda r: r["submitted_at"])
    return records
