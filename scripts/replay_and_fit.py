"""Replay post-processing variants on RESOLVED logged questions, then fit calibration.

usage:
    python scripts/replay_and_fit.py logs/forecasts.jsonl --out calibration.json

Expects records written with forecast_upgrades.logging_schema.append_record whose
"outcome" field has been filled in (0/1) by a resolution-sync job.
Uses a time-ordered split (fit on older questions, score on newer ones) so the
comparison never peeks at the future. Costs zero LLM calls.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere

from forecast_upgrades import aggregation as ag
from forecast_upgrades import calibration as cal
from forecast_upgrades import evaluation as ev
from forecast_upgrades.advanced import HierarchicalPlatt
from forecast_upgrades.logging_schema import load_records, members_at_stage


def build_rows(path: str) -> list[dict]:
    recs = [r for r in load_records(path) if r["qtype"] == "binary" and r.get("outcome") in (0, 1)]
    recs.sort(key=lambda r: r["submitted_at"])
    rows = []
    for r in recs:
        init = members_at_stage(r, "initial")
        if not init:
            continue
        rev = members_at_stage(r, "revised") or init
        rows.append({"members": init, "revised": rev, "outcome": int(r["outcome"]),
                     "segment": r.get("segment"), "source": r.get("source") or "unknown"})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--out", default="calibration.json")
    ap.add_argument("--min-n", type=int, default=100, help="minimum resolved questions before writing a fit")
    ap.add_argument("--cap", type=float, default=0.02)
    args = ap.parse_args()

    rows = build_rows(args.log)
    print(f"{len(rows)} resolved binary questions with member forecasts")
    if len(rows) < 40:
        print("Too few for comparisons; keep logging. (Plumbing looks fine if this ran.)")
        return

    half = len(rows) // 2
    train, test = rows[:half], rows[half:]
    agg = lambda ps: ag.aggregate_binary(ps, "median")  # noqa: E731
    prm, lam_scores = cal.fit_platt_cv([agg(r["members"]) for r in train], [r["outcome"] for r in train])
    lo, hi = args.cap, 1 - args.cap

    pipelines = {
        "median_initial": lambda r: agg(r["members"]),
        "median_after_review": lambda r: agg(r["revised"]),
        "mean_initial": lambda r: ag.aggregate_binary(r["members"], "mean"),
        "logodds_mean_initial": lambda r: ag.aggregate_binary(r["members"], "logodds_mean"),
        "median+cap": lambda r: cal.cap(agg(r["members"]), lo, hi),
        "median+platt+cap": lambda r: cal.cap(float(prm.apply(agg(r["members"]))), lo, hi),
    }
    if len({r["source"] for r in train}) >= 2:
        hp = HierarchicalPlatt().fit([agg(r["members"]) for r in train], [r["outcome"] for r in train],
                                     [r["source"] for r in train])
        pipelines["median+hier_platt+cap"] = lambda r: cal.cap(hp.apply(agg(r["members"]), r["source"]), lo, hi)
    res = ev.replay_binary(test, pipelines, baseline="median_initial")
    print(f"\nscored on the newest {len(test)} questions (fit on the older {len(train)}):")
    print(f"{'pipeline':<24}{'log score':>10}{'brier':>8}{'h2h vs baseline [95% CI]':>32}")
    for name, r in res.items():
        h = r["h2h_vs_baseline"]
        print(f"{name:<24}{r['mean_log_score']:>10.4f}{r['mean_brier']:>8.4f}   {h['mean']:+7.2f} [{h['lo']:+.2f}, {h['hi']:+.2f}]")

    print("\ncalibration of median_initial on all rows:")
    for b in ev.calibration_table([agg(r["members"]) for r in rows], [r["outcome"] for r in rows]):
        print(f"  {b['bin']}: n={b['n']:<4} forecast={b['mean_forecast']:.2f} observed={b['observed']:.2f}")

    if len(rows) >= args.min_n:
        final, scores = cal.fit_platt_cv([agg(r["members"]) for r in rows], [r["outcome"] for r in rows])
        identity_loss = cal.log_loss([agg(r["members"]) for r in rows], [r["outcome"] for r in rows])
        best_cv = min(scores.values())
        use = best_cv < identity_loss
        payload = {"a": final.a if use else 1.0, "b": final.b if use else 0.0, "n": final.n,
                   "cv_log_loss": best_cv, "identity_log_loss": identity_loss, "applied": use}
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.out}: {payload}")
    else:
        print(f"\nnot writing a fit: need >= {args.min_n} resolved questions")


if __name__ == "__main__":
    main()
