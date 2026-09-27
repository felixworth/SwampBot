"""New entry point. The original main.py is an unchanged rollback baseline."""
import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from adaptive_bot import AdaptiveForecastBot
from bot_helpers import check_environment, print_startup_banner, print_run_summary_banner
from forecast_logging import log_report_batch
from forecasting_tools import ForecastReport, MetaculusClient
from upgrade_config import UPGRADE

logger = logging.getLogger(__name__)


def log_dry_run_payloads(reports):
    """Expose the forecast and private-comment payloads without making API writes."""
    try:
        for report in reports:
            if not isinstance(report, ForecastReport):
                continue
            logger.info(
                "DRY RUN forecast payload for %s:\n%s",
                report.question.page_url,
                report.prediction,
            )
            logger.info(
                "DRY RUN private comment payload for %s:\n%s",
                report.question.page_url,
                report.explanation,
            )
    except Exception:
        # Diagnostics must never turn a valid forecast run into a failed run.
        logger.exception("Could not render dry-run payloads")


def _analyst_count() -> int:
    names = [part.strip() for part in os.getenv("BOT_MODELS", "").split(",") if part.strip()]
    if UPGRADE.get("FF_ENSEMBLE_V2"):
        samples = int(UPGRADE.get("samples_per_model", 2))
        names = [name for name in names for _ in range(samples)]
    return len(names)


def _as_amount(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def key_remaining(data: dict) -> tuple[float | None, dict]:
    """Remaining dollars, plus the response fields that decision used."""
    used = {}
    for key in ("limit", "limit_remaining", "usage"):
        if key in data:
            used[key] = data.get(key)
    for key, value in data.items():
        lowered = str(key).lower()
        if any(word in lowered for word in ("reset", "period", "weekly", "monthly", "daily")):
            used[key] = value
    remainings = []
    direct = _as_amount(data.get("limit_remaining"))
    if direct is not None:
        remainings.append(direct)
    elif _as_amount(data.get("limit")) is not None and _as_amount(data.get("usage")) is not None:
        remainings.append(_as_amount(data.get("limit")) - _as_amount(data.get("usage")))
    for key, value in list(data.items()):
        lowered = str(key).lower()
        if isinstance(value, dict) and any(
            word in lowered for word in ("reset", "period", "week", "month", "day")
        ):
            inner = _as_amount(value.get("limit_remaining"))
            if inner is None:
                inner = _as_amount(value.get("remaining"))
            if inner is None and _as_amount(value.get("limit")) is not None and _as_amount(value.get("usage")) is not None:
                inner = _as_amount(value.get("limit")) - _as_amount(value.get("usage"))
            if inner is not None:
                remainings.append(inner)
                used[key] = value
        if any(word in lowered for word in ("weekly", "monthly", "daily", "period")) and "remaining" in lowered:
            amount = _as_amount(value)
            if amount is not None:
                remainings.append(amount)
    grouped = {}
    for key, value in data.items():
        lowered = str(key).lower()
        for word in ("weekly", "monthly", "daily"):
            if word in lowered:
                grouped.setdefault(word, {})[lowered] = (key, value)
    for word, fields in grouped.items():
        remaining_item = next((item for name, item in fields.items() if "remaining" in name), None)
        if remaining_item and _as_amount(remaining_item[1]) is not None:
            continue
        limit_item = next(
            (item for name, item in fields.items() if "limit" in name and "remaining" not in name),
            None,
        )
        usage_item = next((item for name, item in fields.items() if "usage" in name), None)
        if (
            limit_item
            and usage_item
            and _as_amount(limit_item[1]) is not None
            and _as_amount(usage_item[1]) is not None
        ):
            remainings.append(_as_amount(limit_item[1]) - _as_amount(usage_item[1]))
    if not remainings:
        return None, used
    return min(remainings), used


def credits_below_floor(analysts: int | None = None) -> bool:
    """Stop a run when the key cannot cover the floor or one forecast round."""
    if os.getenv("BOT_ENFORCE_SPEND_FLOOR", "true").lower() != "true":
        return False
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        return False
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {key}"},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json().get("data") or {}
        remaining, used = key_remaining(data)
        roster = _analyst_count() if analysts is None else analysts
        floor = float(os.getenv("BOT_SPEND_FLOOR_USD", "1"))
        reservation = 0.50 * roster
        logger.info(
            "OpenRouter key check fields=%s remaining=%s floor=%s analyst_reservation=%s analysts=%s",
            used,
            remaining,
            floor,
            reservation,
            roster,
        )
        if remaining is None:
            return False
        return remaining < floor or remaining < reservation
    except Exception:
        logger.exception("Could not read OpenRouter balance; continuing")
        return False


def repair_missing_comments(client, hours: int = 24) -> int:
    """Post a private comment when a recent forecast has none."""
    path = Path(os.getenv("BOT_LOG_PATH", "logs/forecasts.jsonl"))
    if not path.exists():
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    posted = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") != "forecast_reasoning_complete":
            continue
        logged_at = event.get("logged_at")
        if not logged_at:
            continue
        when = datetime.fromisoformat(logged_at.replace("Z", "+00:00"))
        if when < cutoff:
            continue
        post_id = event.get("post_id")
        if not post_id:
            continue
        try:
            response = requests.get(
                "https://www.metaculus.com/api2/comments/",
                params={"post": post_id, "page_size": 20},
                headers={"Authorization": f"Token {os.environ.get('METACULUS_TOKEN', '')}"},
                timeout=20,
            )
            if response.status_code != 200:
                continue
            payload = response.json()
            comments = payload.get("results", payload if isinstance(payload, list) else [])
            if comments:
                continue
            text = event.get("explanation") or "Forecast comment recovered from the local log."
            client.post_question_comment(post_id, text)
            posted += 1
        except Exception:
            logger.exception("Comment repair skipped for post %s", post_id)
    return posted


async def run(args):
    check_environment(strict=True)
    if args.mode == 'tournament' and not args.tournament_id:
        raise ValueError('Set --tournament-id or BOT_TOURNAMENT_ID to the tournament you entered. '
                         'The original SDK seasonal default may be stale.')
    if args.limit is not None and args.mode != 'test_questions':
        raise ValueError('--limit is only supported in test_questions mode')
    if args.limit is not None and args.limit < 1:
        raise ValueError('--limit must be at least 1')
    print_startup_banner(args.mode, will_publish=args.publish)
    if credits_below_floor():
        print("STOP: OpenRouter balance is below the spend floor or the forecast reservation. No questions were started.")
        return 0
    client = MetaculusClient()
    bot = AdaptiveForecastBot(
        publish_reports_to_metaculus=args.publish,
        folder_to_save_reports_to=args.reports_dir,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        metaculus_client=client,
    )
    if args.mode == 'test_questions':
        targets = ['bot-testing-area']
        bot.skip_previously_forecasted_questions = False
    elif args.mode == 'metaculus_cup':
        targets = [client.CURRENT_METACULUS_CUP_ID]
        bot.skip_previously_forecasted_questions = False
    else:
        targets = [args.tournament_id]
        if args.include_minibench and args.tournament_id != str(client.CURRENT_MINIBENCH_ID):
            targets.append(client.CURRENT_MINIBENCH_ID)
    reports = []
    # Use one event loop: the SDK and model concurrency locks are reused safely.
    for target in targets:
        if args.limit is not None:
            questions = client.get_all_open_questions_from_tournament(target)
            reports.extend(await bot.forecast_questions(
                questions[:args.limit], return_exceptions=True))
        else:
            reports.extend(await bot.forecast_on_tournament(target, return_exceptions=True))
    if not args.publish:
        log_dry_run_payloads(reports)
    elif args.publish:
        repaired = repair_missing_comments(client)
        if repaired:
            logger.info("Repaired %s missing private comments", repaired)
    config = bot.get_config()
    if config.get('durable_logging'):
        log_report_batch(
            reports,
            run_mode=args.mode,
            published=args.publish,
            model_roster=list(config['model_roster']),
            review_triggers=bot._run_stats.get('review_triggers', 0),
            run_stats=dict(bot._run_stats),
        )
    bot.log_report_summary(reports, raise_errors=False)
    print_run_summary_banner(reports, will_publish=args.publish)
    _write_step_summary(bot._run_stats)
    return 1 if any(isinstance(r, BaseException) for r in reports) else 0


def _write_step_summary(stats):
    """Append run counts to the GitHub job page. Never fail the forecast run."""
    path = os.getenv('GITHUB_STEP_SUMMARY')
    if not path:
        return
    try:
        lines = [
            '## Forecast run',
            '',
            '| | |',
            '| --- | ---: |',
            f"| Questions attempted | {stats.get('attempted', 0)} |",
            f"| Succeeded | {stats.get('succeeded', 0)} |",
            f"| Failed | {stats.get('failed', 0)} |",
            f"| Skipped as already forecast | {stats.get('skipped_already', 0)} |",
            f"| Skipped for repeated failures | {stats.get('skipped_failures', 0)} |",
            f"| Not started after provider stop | {stats.get('not_started_stop', 0)} |",
            f"| Not started after time budget | {stats.get('not_started_budget', 0)} |",
            f"| Review triggers | {stats.get('review_triggers', 0)} |",
            f"| Tracked forecast-event cost (USD) | {stats.get('forecast_cost_usd', 0):.4f} |",
            '',
        ]
        with open(path, 'a', encoding='utf-8') as handle:
            handle.write('\n'.join(lines))
    except Exception:
        logger.exception('Could not write the GitHub step summary')


def main():
    parser = argparse.ArgumentParser(description='Research ensemble using original Metaculus SDK integration')
    parser.add_argument('--mode', choices=['tournament', 'metaculus_cup', 'test_questions'], default='test_questions')
    parser.add_argument('--tournament-id', default=os.getenv('BOT_TOURNAMENT_ID'))
    parser.add_argument('--include-minibench', action='store_true')
    parser.add_argument('--limit', type=int, help='Testing area only: forecast at most this many questions')
    submission = parser.add_mutually_exclusive_group()
    submission.add_argument('--publish', dest='publish', action='store_true',
                            help='Submit forecasts and private comments')
    submission.add_argument('--dry-run', dest='publish', action='store_false',
                            help='Run the full pipeline without submitting (the default)')
    parser.set_defaults(publish=False)
    parser.add_argument('--reports-dir', default='logs/adaptive')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    return asyncio.run(run(args))


if __name__ == '__main__':
    raise SystemExit(main())
