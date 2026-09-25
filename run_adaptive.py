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


def credits_below_floor() -> bool:
    """Stop a run when OpenRouter reports less than BOT_SPEND_FLOOR_USD remaining."""
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
        remaining = (response.json().get("data") or {}).get("limit_remaining")
        if remaining is None:
            return False
        floor = float(os.getenv("BOT_SPEND_FLOOR_USD", "1"))
        return float(remaining) < floor
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
        print("STOP: OpenRouter balance is below BOT_SPEND_FLOOR_USD. No questions were started.")
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
        )
    bot.log_report_summary(reports, raise_errors=False)
    print_run_summary_banner(reports, will_publish=args.publish)
    return 1 if any(isinstance(r, BaseException) for r in reports) else 0


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
