"""`--resend` writes nothing, PROVED BY A ROW CENSUS over every table.

Built the way `test_fixture_horizon_probe.py` proves the probe's
no-persistence claim: count rows before and after, across every table the
schema defines, and keep a POSITIVE CONTROL that injects a write so the
census is shown to be capable of noticing one. A no-write test with no such
control is the vacuous shape - `after == before` also holds when the census
counts the wrong thing, or nothing.

THE MARKER GETS ITS OWN OBSERVATION. `picks_sent_date.txt` is a FILE, so a
row census cannot see it, and it is the specific thing a resend must not
touch: moving it changes `_picks_sent_today()`, which gates the review
freeze. Counting rows and declaring the path clean would have missed exactly
the write that matters, so the file is read before and after too.
"""

import datetime as dt
import pathlib

import pytest
from sqlalchemy import inspect as sa_inspect, text

from src.data.database import DatabaseManager
from src.data.models import Base, Match, SavedPick, Team
from src.reporting import pick_resend as pr

DAY = dt.date(2026, 11, 2)


# ════════════════════════════════ fixtures ══════════════════════════════════

def _mgr(tmp_path):
    # Built the way the probe's own fixture is, so the two no-persistence
    # proofs rest on the same kind of database rather than on two.
    mgr = DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / "t.db")}})())
    Base.metadata.create_all(mgr.engine)
    assert not mgr.is_postgres, "the fixture must be SQLite"
    return mgr


class _PgFacade:
    """A session source that DECLARES PostgreSQL over the SQLite fixture.

    The dialect assertion in `load_card` is a production guard, and a test
    that bypassed it by weakening it would remove the guard. This stands in
    front of it instead, and the guard itself is exercised by the known-wrong
    control below - which passes the REAL SQLite manager and requires a
    refusal.
    """

    is_postgres = True

    def __init__(self, mgr):
        self._mgr = mgr
        self.engine = mgr.engine

    def get_session(self):
        return self._mgr.get_session()


def _seed(mgr, n=3, agreements=("solo", "unanimous", "split")):
    with mgr.get_session() as s:
        s.add(Team(id=1, name="Alpha"))
        s.add(Team(id=2, name="Beta"))
        for i in range(n):
            s.add(Match(id=100 + i, league="england/premier-league",
                        match_date=dt.datetime(2026, 11, 2, 20, 0),
                        is_fixture=True, home_team_id=1, away_team_id=2))
            s.add(SavedPick(
                id=500 + i, match_id=100 + i,
                pick_date=dt.datetime(2026, 11, 2, 7, 0),
                match_name=f"Alpha vs Beta {i}",
                league="england/premier-league", market="Over 2.5",
                selection="Over 2.5 Goals", odds=1.80,
                predicted_probability=0.60, expected_value=0.08,
                confidence=0.65, kelly_stake_percentage=1.5,
                risk_level="low", model_agreement=agreements[i % len(agreements)],
                used_fallback_odds=False, is_paper=True))
        s.commit()
    return mgr


def _census(mgr):
    with mgr.get_session() as s:
        return pr.row_census(s, engine=mgr.engine)


# ═══════════════ PART A — the census, and that it can fail ══════════════════

def test_the_CENSUS_ENUMERATES_EVERY_TABLE_IN_THE_SCHEMA(tmp_path):
    """A census over a hand-kept list cannot see a table added later."""
    mgr = _seed(_mgr(tmp_path))
    seen = set(_census(mgr))
    schema = set(sa_inspect(mgr.engine).get_table_names())
    assert seen == schema, (
        "the census and the schema disagree, so a write to a table outside "
        f"the census would be invisible. only-in-schema={schema - seen} "
        f"only-in-census={seen - schema}")
    # NOT a count comparison against CENSUS_TABLES: the fixture's schema and
    # production's need not have the same number of tables, and asserting
    # they do would fail for a reason that has nothing to do with the census.
    # What matters is that the census is non-trivial and covers the table a
    # resend could plausibly write to.
    assert "saved_picks" in seen and len(seen) >= 8, seen


def test_a_RESEND_WRITES_NO_ROW_TO_ANY_TABLE(tmp_path):
    """Counted before and after, over every table."""
    mgr = _seed(_mgr(tmp_path))
    db = _PgFacade(mgr)

    before = _census(mgr)
    assert sum(before.values()) > 0, "the census observed nothing - it is vacuous"

    picks, meta = pr.load_card(db, DAY)
    cert = pr.certify(picks, meta)
    pr.format_verdict(cert, DAY)

    after = _census(mgr)
    assert after == before, (
        "the resend path changed the database. Deltas: "
        + repr({k: (before[k], after[k]) for k in before if before[k] != after[k]}))


def test_the_POSITIVE_CONTROL_the_census_catches_an_injected_write(tmp_path):
    """PROVE THE TEST ABOVE CAN FAIL.

    Without this, `after == before` would also pass against a path that wrote
    freely, if the census counted the wrong thing.
    """
    mgr = _seed(_mgr(tmp_path))
    before = _census(mgr)
    with mgr.get_session() as s:                     # the injected write
        s.add(Match(id=999, league="france/ligue-1",
                    match_date=dt.datetime(2026, 11, 3, 16, 0),
                    is_fixture=True, home_team_id=1, away_team_id=2))
        s.commit()
    after = _census(mgr)
    assert after != before, (
        "the census did NOT notice an injected write, so the no-write test "
        "above proves nothing")
    assert after["matches"] == before["matches"] + 1


# ═════════════ PART B — the marker, which a row census cannot see ═══════════

def test_the_RESEND_DOES_NOT_TOUCH_THE_PICKS_SENT_MARKER(tmp_path, monkeypatch):
    """The file whose move would unfreeze the review path."""
    from src.reporting import telegram_bot as tb
    marker = tmp_path / "picks_sent_date.txt"
    marker.write_text("2026-10-31")
    monkeypatch.setattr(tb, "_PICKS_SENT_STATE", marker)

    mgr = _seed(_mgr(tmp_path))
    picks, meta = pr.load_card(_PgFacade(mgr), DAY)

    sent_bodies = []
    n = tb.TelegramNotifier.__new__(tb.TelegramNotifier)
    n.enabled = True
    monkeypatch.setattr(n, "_get_bot", lambda: object(), raising=False)

    async def _chunked(message, header, report=""):
        sent_bodies.append(message)
    monkeypatch.setattr(n, "_send_chunked", _chunked, raising=False)

    import asyncio
    before = marker.read_text()
    assert asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        n.resend_daily_picks(picks, paper_mode=True, reconstructed=True)) is True
    assert marker.read_text() == before, (
        "the resend moved the picks-sent marker. That marker gates the "
        "review freeze; moving it on a resend can unfreeze reviews.")
    assert sent_bodies, "nothing was formatted, so the test proved nothing"
    assert "RECONSTRUCTED COPY" in sent_bodies[0]


def test_the_KNOWN_WRONG_send_daily_picks_DOES_move_the_marker(tmp_path, monkeypatch):
    """THE CONTROL FOR THE TEST ABOVE.

    If neither path moved the marker the assertion above would hold for the
    wrong reason - the marker never moves, rather than the resend declining
    to move it. This shows the ordinary send does move it.
    """
    from src.reporting import telegram_bot as tb
    marker = tmp_path / "picks_sent_date.txt"
    marker.write_text("2026-10-31")
    monkeypatch.setattr(tb, "_PICKS_SENT_STATE", marker)

    mgr = _seed(_mgr(tmp_path))
    picks, _meta = pr.load_card(_PgFacade(mgr), DAY)

    n = tb.TelegramNotifier.__new__(tb.TelegramNotifier)
    n.enabled = True
    monkeypatch.setattr(n, "_get_bot", lambda: object(), raising=False)

    async def _chunked(message, header, report=""):
        return None
    monkeypatch.setattr(n, "_send_chunked", _chunked, raising=False)

    import asyncio
    # SEEDED WITH A DATE THAT IS NOT TODAY, AND COMPARED TO ITSELF. Asserting
    # the marker now equals `date.today()` would compare a clock-derived
    # value against a clock-derived value across a possible midnight, which
    # is the shape that has cost cards; "it changed from what we wrote" needs
    # no clock and proves the same thing.
    before = marker.read_text()
    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        n.send_daily_picks(picks, force=True, paper_mode=True))
    assert marker.read_text() != before, (
        "the ordinary send did NOT move the marker, so the resend test's "
        "assertion is vacuous")


# ═══════════════ PART C — the refusal, and what it is derived from ══════════

def test_the_CARD_IS_REFUSED_WHEN_IT_CANNOT_BE_REPRODUCED(tmp_path):
    mgr = _seed(_mgr(tmp_path))
    picks, meta = pr.load_card(_PgFacade(mgr), DAY)
    cert = pr.certify(picks, meta)
    assert cert["reproducible"] is False
    # solo + unanimous imply a non-empty models_for; split does not.
    assert cert["certain"]["models_for"] == 2
    assert cert["undetermined"]["models_for"] == 1
    v = pr.format_verdict(cert, DAY)
    assert "RESEND REFUSED" in v and "models_for" in v
    assert "--accept-reconstructed" in v, "the refusal must name its remedy"


def test_CERTAIN_TRACKS_THE_AGREEMENT_DEFINITION_rather_than_a_literal(tmp_path):
    """KNOWN-WRONG INPUT: no agreement recorded means no suffix was rendered.

    `certain` counts picks whose `model_agreement` IMPLIES a non-empty
    `models_for`. With the agreement blank the formatter prints no Models
    line at all, so nothing is lost and `certain` must be empty. If this
    returned 3 the count would be a hard-coded "all picks" rather than a
    reading of the definition.
    """
    mgr = _seed(_mgr(tmp_path), agreements=("", "", ""))
    picks, meta = pr.load_card(_PgFacade(mgr), DAY)
    cert = pr.certify(picks, meta)
    assert cert["certain"] == {}, cert["certain"]
    assert "models_for" not in cert["undetermined"]


def test_CONTRARIAN_VALUE_IS_RECOVERED_EXACTLY_not_estimated(tmp_path):
    """`divergence = prob / (1/odds) = prob * odds` - an identity."""
    mgr = _seed(_mgr(tmp_path))
    picks, _ = pr.load_card(_PgFacade(mgr), DAY)
    for p in picks:
        assert p.contrarian_value == round(
            p.predicted_probability * p.odds, 2)
    assert picks[0].contrarian_value == round(0.60 * 1.80, 2) == 1.08


def test_OPENING_ODDS_IS_LEFT_ABSENT_rather_than_guessed(tmp_path):
    """A field rebuilt to the wrong value is worse than one left out.

    The pick took ONE book's price and the book is not stored, so any
    cross-book summary would cross or miss the formatter's 3% movement
    threshold arbitrarily. Absent is visible; wrong is not.
    """
    mgr = _seed(_mgr(tmp_path))
    picks, _ = pr.load_card(_PgFacade(mgr), DAY)
    # `all()` over an empty list is True, which is how this test passed once
    # while `load_card` was silently returning nothing.
    assert len(picks) == 3, "no picks loaded - the assertions below are vacuous"
    assert all(p.opening_odds == 0.0 for p in picks)
    assert all(p.models_for == "" and p.predicted_xg == "" for p in picks)


# ═══════════════ PART D — the dialect guard is a guard ══════════════════════

def test_a_SQLITE_SESSION_IS_REFUSED(tmp_path):
    """An empty SQLite file answers the pick query with a plausible zero."""
    mgr = _seed(_mgr(tmp_path))
    with pytest.raises(AssertionError) as e:
        pr.load_card(mgr, DAY)            # the REAL manager, not the facade
    assert "PostgreSQL" in str(e.value)


def test_AN_EMPTY_DATE_IS_A_MEASURED_ZERO_not_an_error(tmp_path):
    mgr = _seed(_mgr(tmp_path))
    picks, meta = pr.load_card(_PgFacade(mgr), dt.date(2026, 10, 1))
    assert picks == [] and meta["n"] == 0
