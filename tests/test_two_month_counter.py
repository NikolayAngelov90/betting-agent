"""A window spanning two credit-reset months, accounted for in both.

The stop used to refuse ANY window touching two ledger month keys. That
conflated two different facts, and refusing both threw away the only
candidate window with a date margin because of a READ bug:

  ACCOUNTING SPANS TWO KEYS — the claims already wrote under their own
  month; only `h1_credits_used` took a single row. Expressible, and now
  summed.

  THE RESERVATION CANNOT BE CARVED — a day of the window with no declared
  reservation, or a month whose pool cannot hold it. Then the bound does not
  exist for part of the window and a stop compared against it can never
  fire. Still refused.

BOTH CONTROLS ARE HERE, because one of them alone proves half of it: a test
that only showed a straddling window permitted would also pass if the stop
had simply stopped refusing everything.
"""

import datetime as dt

import pytest
from sqlalchemy import text

from src.data import h1_window as hw
from src.data.database import DatabaseManager
from src.data.models import Base

import scripts.h1_collection_check as hcc


def _mgr(tmp_path):
    mgr = DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / "q.db")}})())
    Base.metadata.create_all(mgr.engine)
    assert not mgr.is_postgres
    return mgr


def _spend(mgr, month_first: dt.date, used: int):
    """Write the attributed counter for one ledger month, as a claim would."""
    from src.data.odds_quota import PROVIDER_H1
    with mgr.get_session() as s:
        # `limit_` is NOT NULL on api_budget, so a claim always writes one.
        s.execute(text("INSERT INTO api_budget (day, provider, used, limit_) "
                       "VALUES (:d, :p, :u, :l)"),
                  {"d": month_first, "p": PROVIDER_H1, "u": used, "l": 500})
        s.commit()


def _straddle(monkeypatch, a=dt.date(2026, 11, 25), b=dt.date(2026, 12, 3),
              declare=True):
    """Move the window across a reset, through the ONE source.

    `declare` ALSO moves the reservation dates. They are separate module
    constants in `odds_quota`, bound to the window at IMPORT, so moving the
    window alone leaves the reservation behind — which is precisely the
    failure control 2 provokes: a window re-positioned without re-declaring
    its reservation.
    """
    monkeypatch.setattr(hw, "FROM", a)
    monkeypatch.setattr(hw, "UNTIL", b)
    if declare:
        import src.data.odds_quota as oq
        monkeypatch.setattr(oq, "H1_RESERVATION_FROM", a)
        monkeypatch.setattr(oq, "H1_RESERVATION_UNTIL", b)


# ═════════════ CONTROL 1 — a straddling window IS permitted ═════════════════

def test_a_STRADDLING_WINDOW_SUMS_BOTH_MONTHS(tmp_path, monkeypatch):
    """The positive half: two keys, both read, and the total is their sum."""
    _straddle(monkeypatch)
    assert [str(k) for k in hw.month_keys()] == ["2026-11-01", "2026-12-01"], \
        "the window is not straddling, so this test proves nothing"
    mgr = _mgr(tmp_path)
    _spend(mgr, dt.date(2026, 11, 1), 90)
    _spend(mgr, dt.date(2026, 12, 1), 30)
    with mgr.get_session() as s:
        assert hcc.h1_credits_used(s) == 120, (
            "the counter read one month and understated the spend — silently, "
            "in the direction that keeps collecting")
        assert hcc.h1_spend_by_month(s) == {"2026-11-01": 90, "2026-12-01": 30}


def test_the_KNOWN_WRONG_single_key_read_UNDERSTATES_it(tmp_path, monkeypatch):
    """What the old read returned, so the fix is shown to change something.

    Without this, summing could be asserted against a fixture where both
    months happened to be equal, or where only one row existed.
    """
    _straddle(monkeypatch)
    mgr = _mgr(tmp_path)
    _spend(mgr, dt.date(2026, 11, 1), 90)
    _spend(mgr, dt.date(2026, 12, 1), 30)
    from src.data.odds_quota import PROVIDER_H1, month_key
    with mgr.get_session() as s:
        row = s.execute(text(
            "SELECT used FROM api_budget WHERE day = :d AND provider = :p"),
            {"d": month_key(dt.date(2026, 11, 25)), "p": PROVIDER_H1}).fetchone()
        single = int(row[0])
    assert single == 90
    assert single < 120, "the single-key read did not understate, so the sum " \
                         "is not demonstrably an improvement"


def test_a_STRADDLING_WINDOW_IS_NOT_REFUSED_ANY_MORE(tmp_path, monkeypatch):
    """The stop must not halt on the mere fact of two keys."""
    _straddle(monkeypatch)
    ok, why = hcc.reservation_is_expressible()
    assert ok, f"a straddling window with a declared reservation was refused: {why}"


# ═════════ CONTROL 2 — an unexpressible reservation IS still refused ════════

def test_a_WINDOW_OUTSIDE_THE_DECLARED_RESERVATION_IS_REFUSED(monkeypatch):
    """The negative half, and the original concern stated correctly.

    `h1_reservation` returns 0 outside H1_RESERVATION_FROM..UNTIL. Those are
    bound to the window module, so moving the window alone keeps them in
    step — which is exactly what makes this hard to provoke. The reservation
    constants are therefore pinned to the ORIGINAL dates while the window
    moves, which is the real failure: a window re-positioned without
    re-declaring its reservation.
    """
    _straddle(monkeypatch, declare=False)      # window moved, reservation NOT
    ok, why = hcc.reservation_is_expressible()
    assert not ok, (
        "a window with NO declared reservation for any of its days was "
        "permitted. The bound does not exist for those days, so a stop "
        "compared against it can never fire.")
    assert "NO declared reservation" in why
    assert "not" in why and "two months" in why, (
        "the refusal must say WHICH fact it is, or it repeats the conflation "
        "it was written to end")


def test_the_REFUSAL_REACHES_THE_STOP(tmp_path, monkeypatch):
    """A predicate nothing consults is not a guard."""
    _straddle(monkeypatch, declare=False)
    stop = hcc.collection_stop(n_fixtures=0, credits_spent=0, raw_rows=0,
                               reservation=168)
    assert stop.halt and stop.state == "BUDGET_UNREADABLE", (
        f"the stop did not refuse an unexpressible reservation: {stop}")
    assert "no expressible reservation" in stop.reason
    assert "not the same as its accounting spanning two months" in stop.reason, (
        "the refusal reached the stop but did not carry WHICH fact it is")


def test_CONTROL_the_STOP_DOES_NOT_REFUSE_A_VALID_STRADDLE(monkeypatch):
    """And the same stop must pass a straddling window that IS declared."""
    _straddle(monkeypatch, declare=True)
    stop = hcc.collection_stop(n_fixtures=0, credits_spent=0, raw_rows=0,
                               reservation=168)
    assert stop.state != "BUDGET_UNREADABLE", (
        f"a declared straddling window was still refused as unreadable: {stop}")


# ═══════════════════════ the window today is NOT straddling ═════════════════

def test_THE_LIVE_WINDOW_TOUCHES_ONE_KEY_and_that_is_asserted():
    """The property holds by construction now; nothing would notice if it
    stopped, which is why it is asserted rather than assumed."""
    assert hw.straddles_a_month() is False
    assert len(hw.month_keys()) == 1
    assert len(hw.days_list()) == hw.DAYS == 8
    ok, why = hcc.reservation_is_expressible()
    assert ok, why
