"""A same-period credit reading can still be stale, and stale reads too HIGH.

    SAME PERIOD IS NOT THE SAME AS CURRENT.

The period check added on 2026-09-11 asks "has the quota reset since this was
written". Necessary, not sufficient. On 2026-09-16 the file read
``{"remaining": 154, "updated": "2026-09-10"}`` while the provider reported 100
remaining and the durable ledger reported 0 spendable: **same month, six days
stale, wrong in the direction of SPENDING.**

WHY STALENESS CORRELATES WITH THE DANGER. The file is written only when a run
actually spends. A gate that refuses every request therefore freezes the last
figure — so the staler the file, the more likely something has stopped the
pipeline from updating it, which is exactly the condition under which a
too-high number must not be believed.

The answer to no reading is to PROBE, never to assume. `/v4/sports` is free.
"""

import json
from datetime import date, timedelta

import src.scrapers.theodds_scraper as ts


def _write(tmp_path, monkeypatch, blob):
    p = tmp_path / "theodds_credits.json"
    p.write_text(json.dumps(blob))
    monkeypatch.setattr(ts, "_CREDITS_STATE_PATH", p)
    return p


def test_todays_reading_is_used(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch,
           {"remaining": 220, "updated": date.today().isoformat()})
    assert ts._load_persisted_credits() == 220


def test_yesterdays_reading_is_still_used(tmp_path, monkeypatch):
    """One day of tolerance, because the pipeline writes daily.

    Refusing a one-day-old figure would probe on every ordinary run, which
    trades a real defect for constant noise.
    """
    y = (date.today() - timedelta(days=1)).isoformat()
    _write(tmp_path, monkeypatch, {"remaining": 220, "updated": y})
    assert ts._load_persisted_credits() == 220


def test_a_SIX_DAY_OLD_reading_is_NO_READING(tmp_path, monkeypatch):
    """THE REGRESSION — the exact file observed on 2026-09-16."""
    old = (date.today() - timedelta(days=6)).isoformat()
    _write(tmp_path, monkeypatch, {"remaining": 154, "updated": old})
    assert ts._load_persisted_credits() is None, (
        "a six-day-old figure was returned as a current credit reading — that "
        "is a permission to spend describing a state that ended")


def test_the_stale_figure_is_not_quietly_rounded_down(tmp_path, monkeypatch):
    """None, not 0. 'We do not know' must not arrive as 'there are none'.

    Returning 0 would trip the hard gate and skip the fetch — the same
    substitution, in the opposite direction, and just as wrong.
    """
    old = (date.today() - timedelta(days=30)).isoformat()
    _write(tmp_path, monkeypatch, {"remaining": 154, "updated": old})
    got = ts._load_persisted_credits()
    assert got is None and got != 0


def test_a_PRIOR_PERIOD_reading_is_still_no_reading(tmp_path, monkeypatch):
    """The original check must survive the new one."""
    last_month = (date.today().replace(day=1) - timedelta(days=1)).isoformat()
    _write(tmp_path, monkeypatch, {"remaining": 15, "updated": last_month})
    assert ts._load_persisted_credits() is None


def test_an_unparseable_date_is_no_reading(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, {"remaining": 400, "updated": "not-a-date"})
    assert ts._load_persisted_credits() is None


def test_an_undated_reading_is_no_reading(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, {"remaining": 400})
    assert ts._load_persisted_credits() is None
