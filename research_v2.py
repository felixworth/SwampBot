"""Resolution-first research: spec, source fetch, crux search, claims ledger."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from forecast_logic import parse_json
from upgrade_config import UPGRADE

ARCHETYPE_DEFINITIONS = {
    "deadline_event": "YES if a stated event happens before a deadline.",
    "threshold_quantity": "YES if a measured number crosses a threshold.",
    "scheduled_decision": "YES depends on a decision that already has a date.",
    "numeric_quantity": "The resolution is a number, not yes or no.",
    "multiple_choice": "The resolution is one of a fixed list of options.",
    "other": "None of the types above.",
}


SPEC_INSTRUCTION = (
    "Extract the resolution specification from the question. Quote the "
    "resolution criteria where precision matters. List anything ambiguous "
    "under ambiguities; do not guess. Return ONLY JSON."
)
CRUX_INSTRUCTION = (
    "List the 3-5 unknowns whose answers would most change the forecast. "
    "For each give a precise question, why it matters, and 1-2 search queries. "
    "Return ONLY JSON."
)
CLAIMS_INSTRUCTION = (
    "Extract distinct factual claims from the research. Assign the same "
    "independence_group to reports of the same underlying announcement or "
    "event. Return ONLY JSON: {\"claims\":[...]}."
)


def time_remaining_days(deadline, now=None) -> float | None:
    if deadline is None or deadline == "":
        return None
    current = now or datetime.now(timezone.utc)
    if isinstance(deadline, str):
        deadline = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return (deadline - current).total_seconds() / 86400


def _as_iso(value) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return value
    return value.isoformat()


def event_deadline(spec: dict[str, Any], scheduled_resolution_time=None):
    """Event horizon: spec deadline_utc, else scheduled resolution. Never close time."""
    deadline = spec.get("deadline_utc")
    if isinstance(deadline, str) and not deadline.strip():
        deadline = None
    if deadline is None:
        deadline = scheduled_resolution_time
    return deadline


def validate_resolution_spec(
    raw: str,
    scheduled_resolution_time=None,
    forecasts_close=None,
    now=None,
) -> dict[str, Any]:
    data = parse_json(raw)
    required = ("question_type", "archetype", "target", "resolution_sources")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError("Resolution spec missing: " + ", ".join(missing))
    if not isinstance(data.get("resolution_sources"), list):
        raise ValueError("resolution_sources must be a list")
    if data.get("archetype") not in ARCHETYPE_DEFINITIONS:
        raise ValueError(
            "archetype must be one of " + ", ".join(ARCHETYPE_DEFINITIONS)
        )
    horizon = event_deadline(data, scheduled_resolution_time)
    data["event_deadline"] = _as_iso(horizon)
    data["forecasts_close"] = _as_iso(forecasts_close)
    data["time_remaining_days"] = time_remaining_days(horizon, now)
    data.setdefault("ambiguities", [])
    data.setdefault("remaining_conditions", [])
    return data


def validate_cruxes(raw: str) -> list[dict[str, Any]]:
    data = parse_json(raw)
    cruxes = data.get("cruxes")
    if not isinstance(cruxes, list):
        raise ValueError("cruxes must be a list")
    minimum = int(UPGRADE["cruxes_min"])
    maximum = int(UPGRADE["cruxes_max"])
    if not minimum <= len(cruxes) <= maximum:
        raise ValueError(f"Need {minimum}-{maximum} cruxes, got {len(cruxes)}")
    cleaned = []
    for index, crux in enumerate(cruxes, start=1):
        if not isinstance(crux, dict):
            raise ValueError("Each crux must be an object")
        queries = crux.get("search_queries") or []
        if not isinstance(queries, list) or not queries:
            raise ValueError("Each crux needs search_queries")
        cleaned.append({
            "id": str(crux.get("id") or f"c{index}"),
            "question": str(crux.get("question") or "").strip(),
            "why_it_matters": str(crux.get("why_it_matters") or "").strip(),
            "search_queries": [str(query).strip() for query in queries if str(query).strip()][:2],
        })
        if not cleaned[-1]["question"] or not cleaned[-1]["search_queries"]:
            raise ValueError("Crux is missing a question or usable query")
    return cleaned


def validate_claims(raw: str) -> list[dict[str, Any]]:
    data = parse_json(raw)
    claims = data.get("claims")
    if not isinstance(claims, list):
        raise ValueError("claims must be a list")
    cleaned = []
    for index, claim in enumerate(claims, start=1):
        if not isinstance(claim, dict) or not str(claim.get("claim") or "").strip():
            continue
        cleaned.append({
            "claim": str(claim["claim"]).strip(),
            "source_url": str(claim.get("source_url") or ""),
            "published": claim.get("published"),
            "event_date": claim.get("event_date"),
            "independence_group": str(claim.get("independence_group") or f"g{index}"),
            "crux_ids": list(claim.get("crux_ids") or []),
            "direction": claim.get("direction") or "neutral",
        })
    return merge_claims(cleaned)


def merge_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim in claims:
        grouped[claim["independence_group"]].append(claim)
    merged = []
    for group, items in grouped.items():
        first = dict(items[0])
        first["independence_group"] = group
        if len(items) > 1:
            first["note"] = f"{len(items)} reports of the same event"
        merged.append(first)
    return merged


def similar_questions(title: str, records: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    tokens = set(re.findall(r"[a-z0-9]+", (title or "").lower()))
    if not tokens:
        return []
    scored = []
    seen = set()
    for record in records:
        other = str(record.get("title") or "")
        key = (record.get("question_id"), other)
        if key in seen or other == title:
            continue
        seen.add(key)
        other_tokens = set(re.findall(r"[a-z0-9]+", other.lower()))
        if not other_tokens:
            continue
        score = len(tokens & other_tokens) / len(tokens | other_tokens)
        if score >= 0.25 and record.get("outcome") is not None:
            scored.append((score, {
                "title": other,
                "outcome": record.get("outcome"),
                "prediction": record.get("final_submitted") or record.get("prediction"),
                "question_id": record.get("question_id"),
            }))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[:limit]]


def load_log_records(path: str | Path | None = None) -> list[dict[str, Any]]:
    paths = []
    if path:
        paths.append(Path(path))
    else:
        paths.append(Path(UPGRADE["log_path"]))
        paths.extend(Path("data/runs").glob("**/*.jsonl"))
    rows = []
    for destination in paths:
        if destination.exists() and destination.is_file():
            rows.extend(
                json.loads(line)
                for line in destination.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    return rows


def extract_source_urls(spec: dict[str, Any]) -> list[str]:
    urls = []
    for source in spec.get("resolution_sources") or []:
        if isinstance(source, dict) and source.get("url"):
            urls.append(str(source["url"]))
        elif isinstance(source, str) and source.startswith("http"):
            urls.append(source)
    text = json.dumps(spec, default=str)
    urls.extend(re.findall(r"https?://[^\s\"']+", text))
    unique = []
    for url in urls:
        if url not in unique:
            unique.append(url)
    return unique


def _relevant_excerpt(text: str, limit: int = 4000) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    scored = []
    for sentence in sentences:
        numbers = len(re.findall(r"\d", sentence))
        if numbers or len(sentence) > 80:
            scored.append((numbers, sentence))
    scored.sort(key=lambda item: item[0], reverse=True)
    chosen = " ".join(sentence for _score, sentence in scored[:12]) or text
    return chosen[:limit]


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_resolution_fact(url: str, timeout: float = 15.0) -> dict[str, Any]:
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "swampbot-research-v2"},
        )
        response.raise_for_status()
        text = _relevant_excerpt(_strip_html(response.text))
        if len(text) < 300:
            return {
                "source": urlparse(url).netloc or url,
                "value": text or None,
                "as_of": datetime.now(timezone.utc).date().isoformat(),
                "note": "needs_search",
                "url": url,
            }
        return {
            "source": urlparse(url).netloc or url,
            "value": text,
            "as_of": datetime.now(timezone.utc).date().isoformat(),
            "note": "fetched",
            "url": url,
        }
    except Exception as exc:
        return {
            "source": urlparse(url).netloc or url,
            "value": None,
            "as_of": datetime.now(timezone.utc).date().isoformat(),
            "note": f"fetch_failed: {type(exc).__name__}",
            "url": url,
        }


def format_research_brief(
    spec: dict[str, Any],
    facts: list[dict[str, Any]],
    outside_view: list[dict[str, Any]],
    crux_findings: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    uncertainties: list[str],
) -> str:
    sections = [
        "## Forecasts close\n" + str(spec.get("forecasts_close") or "unknown"),
        "## Event deadline\n" + str(spec.get("event_deadline") or "unknown"),
        "## Resolution spec\n" + json.dumps(spec, default=str, separators=(',', ':')),
        "## Resolution facts\n" + json.dumps(facts, default=str, separators=(',', ':')),
        "## Outside view\n" + (
            json.dumps(outside_view, default=str, separators=(',', ':'))
            if outside_view else "No similar resolved questions in the local log."
        ),
        "## Cruxes with findings\n" + json.dumps(crux_findings, default=str, separators=(',', ':')),
        "## Claims by independence group\n" + json.dumps(claims, default=str, separators=(',', ':')),
        "## Open uncertainties\n" + (
            "\n".join(f"- {item}" for item in uncertainties) or "- None recorded"
        ),
    ]
    return "\n\n".join(sections)


def spec_prompt(question_spec: dict[str, Any]) -> str:
    return (
        SPEC_INSTRUCTION
        + "\nQUESTION DATA:\n"
        + json.dumps(question_spec, default=str)
        + "\nArchetype, pick one:\n"
        + "\n".join(f"- {name}: {text}" for name, text in ARCHETYPE_DEFINITIONS.items())
        + '\nSchema: {"question_type":"","archetype":"","target":"",'
        '"resolves_yes_if":"","deadline_utc":"","resolution_sources":'
        '[{"name":"","url":"","kind":""}],"units":"","thresholds":[],'
        '"bounds":{},"already_happened":"","remaining_conditions":[],'
        '"ambiguities":[]}'
    )


def crux_prompt(question_spec: dict[str, Any], spec: dict[str, Any], facts: list[dict[str, Any]]) -> str:
    return (
        CRUX_INSTRUCTION
        + "\nQUESTION DATA:\n" + json.dumps(question_spec, default=str)
        + "\nRESOLUTION SPEC:\n" + json.dumps(spec, default=str)
        + "\nRESOLUTION FACTS:\n" + json.dumps(facts, default=str)[:12000]
        + '\nSchema: {"cruxes":[{"id":"c1","question":"","why_it_matters":"",'
        '"search_queries":["",""]}]}'
    )


def claims_prompt(crux_findings: list[dict[str, Any]]) -> str:
    return (
        CLAIMS_INSTRUCTION
        + "\nFINDINGS:\n" + json.dumps(crux_findings, default=str)[:20000]
        + '\nSchema: {"claims":[{"claim":"","source_url":"","published":"",'
        '"event_date":null,"independence_group":"g1","crux_ids":["c1"],'
        '"direction":"neutral"}]}'
    )
