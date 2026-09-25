"""Match a binary question to one liquid public market, or return nothing."""
from __future__ import annotations

import re
from datetime import datetime, timezone

import requests

from upgrade_config import UPGRADE


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _overlap(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _close_gap_days(question_close, market_close_ms) -> float | None:
    if question_close is None or not market_close_ms:
        return None
    if isinstance(question_close, str):
        question_close = datetime.fromisoformat(question_close.replace("Z", "+00:00"))
    if question_close.tzinfo is None:
        question_close = question_close.replace(tzinfo=timezone.utc)
    market_close = datetime.fromtimestamp(market_close_ms / 1000, tz=timezone.utc)
    return abs((question_close - market_close).total_seconds()) / 86400


def match_manifold(question_text: str, question_close, timeout: float = 15.0) -> dict | None:
    """Accept a Manifold market only when the title, deadline, and liquidity all agree."""
    response = requests.get(
        "https://api.manifold.markets/v0/search-markets",
        params={"term": question_text[:180], "limit": 5},
        timeout=timeout,
    )
    response.raise_for_status()
    best = None
    for market in response.json() or []:
        if market.get("outcomeType") != "BINARY":
            continue
        price = market.get("probability")
        if not isinstance(price, (int, float)) or not 0 < price < 1:
            continue
        overlap = _overlap(question_text, market.get("question") or "")
        gap = _close_gap_days(question_close, market.get("closeTime"))
        liquidity = float(market.get("volume") or 0)
        max_gap = float(UPGRADE.get("market_max_deadline_gap_days", 7))
        if overlap < 0.45 or gap is None or gap > max_gap or liquidity < 1000:
            continue
        candidate = {
            "source": "manifold",
            "price": float(price),
            "liquidity": liquidity,
            "title": market.get("question"),
            "deadline_gap_days": gap,
        }
        if best is None or overlap > best["overlap"]:
            candidate["overlap"] = overlap
            best = candidate
    if best:
        best.pop("overlap", None)
    return best
