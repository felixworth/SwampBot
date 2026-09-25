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


def time_remaining_days(close_time, now=None) -> float | None:
    if close_time is None:
        return None
    current = now or datetime.now(timezone.utc)
    if isinstance(close_time, str):
        close_time = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
    if close_time.tzinfo is None:
        close_time = close_time.replace(tzinfo=timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return (close_time - current).total_seconds() / 86400


def validate_resolution_spec(raw: str, close_time=None, now=None) -> dict[str, Any]:
    data = parse_json(raw)
    required = ("question_type", "archetype", "target", "resolution_sources")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError("Resolution spec missing: " + ", ".join(missing))
    if not isinstance(data.get("resolution_sources"), list):
        raise ValueError("resolution_sources must be a list")
    data["time_remaining_days"] = time_remaining_days(close_time, now)
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
    destination = Path(path or UPGRADE["log_path"])
    if not destination.exists():
        return []
    return [
        json.loads(line)
        for line in destination.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
        text = _strip_html(response.text)[:4000]
        if not text:
            raise ValueError("empty page")
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
        "## Resolution spec\n" + json.dumps(spec, indent=2, default=str),
        "## Resolution facts\n" + json.dumps(facts, indent=2, default=str),
        "## Outside view\n" + (
            json.dumps(outside_view, indent=2, default=str)
            if outside_view else "No similar resolved questions in the local log."
        ),
        "## Cruxes with findings\n" + json.dumps(crux_findings, indent=2, default=str),
        "## Claims by independence group\n" + json.dumps(claims, indent=2, default=str),
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
