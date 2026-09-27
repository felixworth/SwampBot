"""Combine per-run event logs + outcomes into one replay-ready records file.

usage:
    python scripts/build_replay_records.py --runs "data/runs/**/*.jsonl" \
        --outcomes data/outcomes.jsonl --out logs/replay_records.jsonl
then:
    python scripts/replay_and_fit.py logs/replay_records.jsonl --out calibration.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere

from forecast_upgrades.replay_adapter import events_to_records, load_jsonl, normalize_live_event


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help='glob, e.g. "data/runs/**/*.jsonl"')
    ap.add_argument("--outcomes", required=True, help="jsonl of {question_id, outcome}")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run_files = sorted(glob.glob(a.runs, recursive=True))
    events = [normalize_live_event(ev) for ev in load_jsonl(run_files)]
    outcomes = {}
    if Path(a.outcomes).exists():
        for row in load_jsonl([a.outcomes]):
            outcomes[row["question_id"]] = row["outcome"]
    records = events_to_records(events, outcomes)
    if run_files and not records:
        raise SystemExit("run files exist but zero replay records were built")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    resolved = sum(r["outcome"] is not None for r in records)
    print(f"{len(events)} events -> {len(records)} questions ({resolved} with outcomes) -> {a.out}")


if __name__ == "__main__":
    main()
