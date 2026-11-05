"""The client settlement's COUNT and PERIOD must describe one population.

SPLIT OUT OF `test_client_chat_is_sealed.py` on 2026-11-04. These tests are
DB-backed: they seed a database and drive `experiment_record.build`, so they
import `DatabaseManager` and the models and are therefore on the capture
path.

THAT IS THE RIGHT CLASS FOR THEM AND THE WRONG ONE FOR THE OTHER FILE. A test
about WHICH CHAT a message reaches must not be able to stop price capture; a
test about whether a reported population is the one that was counted is data
correctness and should. Leaving both in one file silently moved the
notification tests into the gate — the same import-pulls-a-file-into-the-gate
trap this project has hit three times now.

THE DEFECT THEY GUARD, measured on production 2026-11-04: `settled` filtered
`result IS NOT NULL` while `min/max(pick_date)` did not, so 1,645 settled
picks were paired with a period ending 2026-11-04 that 13 UNSETTLED picks had
widened. And 929 + 663 = 1,592 against 1,645 left 53 VOID picks a client
adding the two numbers could not account for.

A client cannot check either and will not know to ask.
"""

import datetime as dt

import pytest


# ══════ A2. the count and the period describe the SAME population ══════════

def _seeded_record(tmp_path, settled_days, unsettled_days):
    """Build a real ExperimentRecord over a seeded database."""
    import datetime as _dt

    from src.data.database import DatabaseManager
    from src.data.models import Base, Match, SavedPick, Team
    from src.reporting.experiment_record import build
    mgr = DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / "e.db")}})())
    Base.metadata.create_all(mgr.engine)
    with mgr.get_session() as s:
        s.add(Team(id=1, name="A")); s.add(Team(id=2, name="B"))
        pid = 1
        for d, res in ([(d, "win") for d in settled_days]
                       + [(d, None) for d in unsettled_days]):
            s.add(Match(id=100 + pid, league="x",
                        match_date=_dt.datetime.combine(d, _dt.time(20, 0)),
                        is_fixture=False, home_team_id=1, away_team_id=2))
            s.add(SavedPick(
                id=pid, match_id=100 + pid, pick_date=d,
                match_name="A vs B", league="x", market="Over 2.5",
                selection="Over 2.5 Goals", odds=1.80,
                predicted_probability=0.6, expected_value=0.08,
                confidence=0.65, kelly_stake_percentage=1.5,
                risk_level="low", model_agreement="solo",
                used_fallback_odds=False, is_paper=True, result=res))
            pid += 1
        s.commit()
    return build(mgr), mgr


def test_the_COUNT_AND_THE_PERIOD_COME_FROM_THE_SAME_POPULATION(tmp_path):
    """An UNSETTLED pick must not widen the period it is not counted in.

    MEASURED 2026-11-04 on production before the fix: 1,645 settled picks
    paired with a period ending 2026-11-04, which 13 unsettled picks had
    widened by a day. A client cannot check that and will not know to ask.
    """
    settled = [dt.date(2026, 8, 11), dt.date(2026, 9, 1), dt.date(2026, 11, 3)]
    unsettled = [dt.date(2026, 11, 20)]          # later, and NOT counted
    rec, _ = _seeded_record(tmp_path, settled, unsettled)
    assert rec.settled == len(settled), rec.settled
    assert str(rec.last_date) == "2026-11-03", (
        f"the period ends {rec.last_date}, but the last SETTLED pick is "
        f"2026-11-03. An unsettled pick widened a period it does not enter.")
    assert str(rec.first_date) == "2026-08-11"
    body = "\n".join(__import__(
        "src.reporting.experiment_record", fromlist=["client_block"]
    ).client_block(rec, html=False))
    assert f"from {len(settled)} settled picks" in body, body
    assert "3 Nov 2026" in body and "20 Nov" not in body, body


def test_KNOWN_WRONG_an_UNFILTERED_MAX_WOULD_DIFFER(tmp_path):
    """Prove the test above is checking something.

    If the seeded unsettled pick did not extend past the settled ones, the
    assertion would hold whether or not the query were filtered.
    """
    from sqlalchemy import text
    settled = [dt.date(2026, 8, 11), dt.date(2026, 11, 3)]
    unsettled = [dt.date(2026, 11, 20)]
    _rec_, mgr = _seeded_record(tmp_path, settled, unsettled)
    with mgr.get_session() as s:
        unf = s.execute(text(
            "SELECT max(pick_date) FROM saved_picks WHERE is_paper = 1")).scalar()
        fil = s.execute(text(
            "SELECT max(pick_date) FROM saved_picks "
            "WHERE is_paper = 1 AND result IS NOT NULL")).scalar()
    assert str(unf) != str(fil), (
        f"unfiltered {unf} equals filtered {fil}, so this fixture cannot "
        f"distinguish the two queries and the test above proves nothing")
    assert str(unf).startswith("2026-11-20") and str(fil).startswith("2026-11-03")


def test_EVERY_NUMBER_IN_THE_CLIENT_BLOCK_RECONCILES(tmp_path):
    """wins + losses + void == settled, and the rate is of the decided.

    MEASURED on production: 929 + 663 = 1,592 against 1,645 settled, leaving
    53 VOID picks that a client adding the two numbers could not account for.
    """
    from src.reporting.experiment_record import ExperimentRecord, client_block
    import inspect
    sig = inspect.signature(ExperimentRecord)
    kw = {k: v for k, v in (
        ("wins", 929), ("losses", 663), ("settled", 1645), ("cohorts", 15),
        ("win_rate", 929 / 1592), ("flat_roi", -0.057), ("avg_odds", 1.62),
        ("pl_units", -93.0), ("first_date", "2026-08-11"),
        ("last_date", "2026-11-03")) if k in sig.parameters}
    body = "\n".join(client_block(ExperimentRecord(**kw), html=False))
    assert "929 wins" in body and "663 losses" in body
    assert "53 void" in body, (
        f"the 53 void picks are invisible, so 929 + 663 does not reach the "
        f"1645 stated: {body}")
    assert "1592 decided" in body, "the percentage's base is not stated"
    assert "58.4%" in body
