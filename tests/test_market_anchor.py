from datetime import datetime, timedelta, timezone

from market_anchor import _close_gap_days, _overlap


def test_overlap_rejects_unrelated_titles():
    assert _overlap("Will France leave the EU?", "Bitcoin price above 100k") < 0.45


def test_close_gap_rejects_distant_deadlines():
    close = datetime(2026, 10, 1, tzinfo=timezone.utc)
    far = int((close + timedelta(days=30)).timestamp() * 1000)
    near = int((close + timedelta(days=2)).timestamp() * 1000)
    assert _close_gap_days(close, far) > 7
    assert _close_gap_days(close, near) < 7
