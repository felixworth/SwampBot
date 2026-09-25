"""Fill outcomes on closed questions already present in the forecast log.

usage:
    python scripts/sync_resolutions.py [logs/forecasts.jsonl]

Reads every JSONL record, fetches the Metaculus resolution for closed questions
that still lack an outcome, then rewrites the log atomically. Annulled or
unresolved questions are left unchanged. Idempotent.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upgrade_config import UPGRADE

API_BASE = "https://www.metaculus.com/api2"


def _headers() -> dict[str, str]:
    token = os.getenv("METACULUS_TOKEN", "").strip()
    headers = {"User-Agent": "swampbot-sync-resolutions"}
    if token:
        headers["Authorization"] = f"Token {token}"
    return headers


def _fetch_question(question_id: int) -> dict:
    response = requests.get(
        f"{API_BASE}/questions/{question_id}/",
        headers=_headers(),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _outcome_from_payload(payload: dict) -> tuple[object | None, bool]:
    """Return (outcome, skip). skip=True means leave the record unchanged."""
    status = str(payload.get("status") or payload.get("question_status") or "").lower()
    if status in {"annulled", "ambiguous"}:
        return None, True
    if payload.get("resolution") in (None, "", "annulled", "ambiguous"):
        if status not in {"resolved"}:
            return None, True
        return None, True
    resolution = payload.get("resolution")
    qtype = str(payload.get("type") or payload.get("question_type") or "").lower()
    if qtype in {"binary", "binary_question"}:
        if resolution in (0, 0.0, "no", "0"):
            return 0, False
        if resolution in (1, 1.0, "yes", "1"):
            return 1, False
        return None, True
    if qtype in {"multiple_choice", "multiple_choice_question"}:
        options = payload.get("options") or []
        if isinstance(resolution, int):
            return resolution, False
        if resolution in options:
            return options.index(resolution), False
        return resolution, False
    return resolution, False


def sync_log(path: Path) -> dict[str, int]:
    if not path.exists():
        return {"records": 0, "updated": 0, "skipped": 0}
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cache: dict[int, tuple[object | None, bool]] = {}
    updated = 0
    skipped = 0
    outcomes = []
    for record in records:
        question_id = record.get("question_id")
        if question_id is None:
            skipped += 1
            continue
        if record.get("outcome") is not None:
            skipped += 1
            continue
        question_id = int(question_id)
        if question_id not in cache:
            try:
                cache[question_id] = _outcome_from_payload(_fetch_question(question_id))
            except Exception:
                cache[question_id] = (None, True)
        outcome, skip = cache[question_id]
        if skip or outcome is None:
            skipped += 1
            continue
        outcomes.append({
            "question_id": question_id,
            "outcome": outcome,
            "resolved_at": datetime.now(timezone.utc).isoformat(),
        })
        updated += 1
    destination = Path(UPGRADE["outcomes_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = destination.read_text(encoding="utf-8") if destination.exists() else ""
    known = set()
    for line in existing.splitlines():
        if line.strip():
            known.add(json.loads(line).get("question_id"))
    with destination.open("a", encoding="utf-8") as handle:
        for row in outcomes:
            if row["question_id"] not in known:
                handle.write(json.dumps(row) + "\n")
    return {"records": len(records), "updated": updated, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?", default=UPGRADE["log_path"])
    args = parser.parse_args()
    stats = sync_log(Path(args.log))
    print(
        f"synced {args.log}: {stats['records']} records, "
        f"{stats['updated']} updated, {stats['skipped']} unchanged"
    )


if __name__ == "__main__":
    main()
