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

BOUNDED 2026-10-01. `test_yesterdays_reading_is_still_used` used to compute its
"yesterday" from the real clock, so on the FIRST OF ANY MONTH it wrote
`today - 1 day` — a PRIOR-PERIOD date — and asserted the reading was admitted,
while `test_a_PRIOR_PERIOD_reading_is_still_no_reading` wrote the *same date* and
asserted it was refused. The pair contradicted itself one day in thirty, and on
2026-10-01 it failed in CI and took `--update`, settlement and the whole card
with it.

THE DEFECT WAS IN THE TESTS, NOT THE LOADER. Production is correct and its
docstring named the date in advance. The fix is to bound the test's claim to what
it always meant — one day of tolerance WITHIN a billing period — and to stop
depending on the real calendar at all, so the suite's behaviour is a property of
the code rather than of the day it runs on.

NO GRACE WAS GRANTED TO THE LOADER. Admitting a one-day-old reading across a
period boundary would widen a tolerance on a spending guard and change which
readings are admitted — prediction-affecting, and the exact relaxation this file
exists to refuse.
"""

import json
from datetime import date, timedelta

import pytest

import src.scrapers.theodds_scraper as ts


def _write(tmp_path, monkeypatch, blob):
    p = tmp_path / "theodds_credits.json"
    p.write_text(json.dumps(blob))
    monkeypatch.setattr(ts, "_CREDITS_STATE_PATH", p)
    return p


def _pin_today(monkeypatch, d: date):
    """Pin `date.today()` inside the module under test.

    The loader reads `date.today()` twice — once for the period check, once for
    the age — so pinning it is what makes every case below decidable on any
    calendar day. Everything else about `date` is inherited.
    """
    class _PinnedDate(date):
        @classmethod
        def today(cls):
            return d
    monkeypatch.setattr(ts, "date", _PinnedDate)


def _same_period_yesterday(d: date) -> date:
    """The newest date that is both <= d - 1 day AND inside d's period.

    On the 2nd or later this is literally yesterday. On the 1st there is no
    same-period yesterday, so it is `d` itself — which is the honest answer:
    the one-day tolerance has nothing to reach back to without crossing the
    boundary, and crossing it is what must not be forgiven.
    """
    y = d - timedelta(days=1)
    return y if ts._credits_period(y) == ts._credits_period(d) else d


def test_todays_reading_is_used(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch,
           {"remaining": 220, "updated": date.today().isoformat()})
    assert ts._load_persisted_credits() == 220


def test_yesterdays_reading_is_still_used(tmp_path, monkeypatch):
    """One day of tolerance WITHIN A PERIOD, because the pipeline writes daily.

    Refusing a one-day-old figure would probe on every ordinary run, which
    trades a real defect for constant noise. But the tolerance stops at the
    billing boundary: see the module docstring for what unbounding it cost.

    `today` is pinned so this asserts a property of the loader rather than of
    the day the suite runs on.
    """
    today = date(2026, 10, 15)          # any day whose yesterday is same-period
    _pin_today(monkeypatch, today)
    y = _same_period_yesterday(today)
    assert y == date(2026, 10, 14), "the fixture no longer exercises a real yesterday"
    _write(tmp_path, monkeypatch, {"remaining": 220, "updated": y.isoformat()})
    assert ts._load_persisted_credits() == 220


def test_the_tolerance_STOPS_at_the_period_boundary(tmp_path, monkeypatch):
    """The case that used to be asserted both ways. Now asserted once, correctly.

    On the 1st, `today - 1 day` is last month. It is NO READING — the quota has
    reset and the figure describes a month that is over.
    """
    _pin_today(monkeypatch, date(2026, 10, 1))
    _write(tmp_path, monkeypatch,
           {"remaining": 220, "updated": date(2026, 9, 30).isoformat()})
    assert ts._load_persisted_credits() is None, (
        "a one-day-old reading from the PREVIOUS billing period was admitted — "
        "that is the cross-boundary grace this file refuses")


@pytest.mark.parametrize("ordinal", range(366))
def test_THE_PAIR_NEVER_WRITES_ONE_DATE_WITH_TWO_VERDICTS(ordinal, tmp_path,
                                                          monkeypatch):
    """THE BOUND, PROVED OVER A FULL YEAR — not by asserting today passes.

    For every day of 2026, the two cases must write DIFFERENT dates and the
    loader must agree with each of them:

        same-period yesterday  -> ADMITTED
        last day of prior month -> REFUSED

    The old pair collided on all twelve first-of-months. If this parameterisation
    ever reports a collision again, the two assertions have started describing
    one date and one of them is wrong.
    """
    today = date(2026, 1, 1) + timedelta(days=ordinal)
    _pin_today(monkeypatch, today)

    admitted_date = _same_period_yesterday(today)
    refused_date = today.replace(day=1) - timedelta(days=1)

    assert admitted_date != refused_date, (
        f"COLLISION on {today}: both cases write {admitted_date} and assert "
        f"opposite outcomes — the contradiction is back")
    assert ts._credits_period(admitted_date) == ts._credits_period(today)
    assert ts._credits_period(refused_date) != ts._credits_period(today)

    _write(tmp_path, monkeypatch,
           {"remaining": 220, "updated": admitted_date.isoformat()})
    assert ts._load_persisted_credits() == 220, (
        f"same-period reading dated {admitted_date} refused on {today}")

    _write(tmp_path, monkeypatch,
           {"remaining": 220, "updated": refused_date.isoformat()})
    assert ts._load_persisted_credits() is None, (
        f"prior-period reading dated {refused_date} admitted on {today}")


def test_POSITIVE_CONTROL_the_period_rule_is_load_bearing(tmp_path, monkeypatch):
    """Remove the prior-period rule and the refusal must stop holding.

    THE 1st IS THE ONLY DAY WHERE THE PERIOD RULE IS THE SOLE PROTECTION. Last
    month's final day is then exactly ONE day old, so `age > 1` is False and the
    staleness check lets it through — the period check is all that stands between
    a finished month's figure and a spending decision.

    Neutralising `_credits_period` (every date in one period) must therefore
    flip the refusal to an admission. If it does not, this file's central
    assertion is being carried by the age check and would survive deleting the
    rule it claims to test.
    """
    _pin_today(monkeypatch, date(2026, 10, 1))
    _write(tmp_path, monkeypatch,
           {"remaining": 220, "updated": date(2026, 9, 30).isoformat()})

    assert ts._load_persisted_credits() is None          # rule present: refused

    monkeypatch.setattr(ts, "_credits_period", lambda d: "ONE-PERIOD")
    assert ts._load_persisted_credits() == 220, (
        "with the period rule neutralised the prior-month reading was STILL "
        "refused, so something other than that rule is doing the work and the "
        "test does not prove what it claims")


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
