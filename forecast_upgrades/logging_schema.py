"""What to log for every question so that later you can learn from it.

One JSON line per question (append-only). Store research text separately
(content-addressed by hash) to keep the log small; the hash lets you replay
two variants on IDENTICAL research, which removes most of the noise from A/B tests.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class ResearchBundle:
    bundle_id: str               # content_hash of the text
    provider: str                # "asknews", "perplexity", "openrouter:online", "resolution_source", ...
    queries: list[str] = field(default_factory=list)
    n_sources: int = 0
    newest_source_date: str | None = None
    cost_usd: float = 0.0


@dataclass
class MemberForecast:
    model: str                   # e.g. "openrouter/openai/..." (whatever you call)
    sample_idx: int
    stage: str                   # "initial" | "revised"
    research_bundle_ids: list[str]
    binary: float | None = None
    mc: list[float] | None = None
    percentiles: dict[str, float] | None = None   # {"0.1": value, ...}; str keys for JSON
    structured_inputs: dict[str, Any] | None = None  # e.g. {"k_events": 2, "exposure_years": 10}
    reasoning_hash: str | None = None
    cost_usd: float = 0.0
    latency_s: float = 0.0


@dataclass
class QuestionRecord:
    question_id: int
    post_id: int
    tournament: str
    qtype: str                   # "binary" | "multiple_choice" | "numeric" | "discrete" | "date"
    title: str
    close_time: str | None
    bot_version: str
    segment: str | None = None   # your router's archetype label, e.g. "before_deadline"
    research: list[ResearchBundle] = field(default_factory=list)
    members: list[MemberForecast] = field(default_factory=list)
    review_triggered: bool = False
    review_reasons: list[str] = field(default_factory=list)
    aggregate_raw: Any = None    # before calibration / caps
    final_submitted: Any = None  # exactly what was POSTed
    market_anchor: dict[str, Any] | None = None  # {"source": ..., "price": ..., "liquidity": ...}
    flags: list[str] = field(default_factory=list)  # "units_check_failed", "state_guard_tripped", ...
    submitted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    outcome: Any = None          # filled in later by a resolution sync job


def append_record(path: str | Path, record: QuestionRecord) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def load_records(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def members_at_stage(record: dict, stage: str = "initial") -> list[float]:
    """Binary member probabilities for replay (e.g. compare 'initial' vs 'revised')."""
    return [m["binary"] for m in record["members"] if m["stage"] == stage and m.get("binary") is not None]
