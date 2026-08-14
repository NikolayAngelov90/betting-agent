"""Stage 10 — `pick_observations`: dual CLV observation storage.

The contract this file defends, in one sentence: **the frozen model's taken
price must survive a Claude CHANGE**, because after the review
`saved_picks.odds` holds the new price and nothing else in the system holds the
old one. The odds table keeps one row per (match, bookmaker, market, selection)
and overwrites it on every refresh, so a price not captured at pick time is
gone permanently.

Temp SQLite throughout; conftest strips DATABASE_URL.
"""

from datetime import date, datetime, timedelta

import pytest
from sqlalchemy.exc import IntegrityError

import src.data.database as db_mod
from src.data.models import (Base, Match, Odds, PickObservation, SavedPick,
                             Team)


def _mgr(tmp_path, name):
    from sqlalchemy import event

    mgr = db_mod.DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / name)}})())

    # SQLite enforces foreign keys only when asked, and only per connection —
    # the pool hands out several, so this has to be a connect listener rather
    # than a one-off PRAGMA. Postgres always enforces them; without this the
    # cascade test would pass vacuously on SQLite.
    @event.listens_for(mgr.engine, "connect")
    def _fk_on(dbapi_conn, _rec):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON")
        cur.close()

    Base.metadata.create_all(mgr.engine)
    with mgr.get_session() as s:
        s.add(Team(id=1, name="Home FC"))
        s.add(Team(id=2, name="Away FC"))
        s.commit()
    return mgr


def _obs(pick_id, attribution, market="1X2", selection="Home Win",
         taken_odds=2.0, taken_at=None):
    return PickObservation(
        pick_id=pick_id, attribution=attribution, market=market,
        selection=selection, taken_odds=taken_odds,
        taken_at=taken_at or datetime(2026, 9, 1, 12, 0),
        closing_status="pending")


def _pick_row(pid=1, **kw):
    base = dict(id=pid, match_id=pid, pick_date=date(2026, 9, 1),
                market="1X2", selection="Home Win", odds=2.0,
                predicted_probability=0.5, expected_value=0.1, confidence=0.5,
                kelly_stake_percentage=1.0, closing_capture_status="pending",
                created_at=datetime(2026, 9, 1, 12, 0))
    base.update(kw)
    return SavedPick(**base)


# ═══════════════════════════════ §17 — database integrity

def test_unique_pick_attribution(tmp_path):
    mgr = _mgr(tmp_path, "u.db")
    with mgr.get_session() as s:
        s.add(Match(id=1, home_team_id=1, away_team_id=2,
                    match_date=datetime(2026, 9, 1, 18, 0), league="x/y"))
        s.add(_pick_row(1))
        s.add(_obs(1, "model"))
        s.commit()

    with pytest.raises(IntegrityError):
        with mgr.get_session() as s:
            s.add(_obs(1, "model", selection="Away Win"))
            s.commit()


def test_both_attributions_coexist_on_one_pick(tmp_path):
    mgr = _mgr(tmp_path, "u2.db")
    with mgr.get_session() as s:
        s.add(Match(id=1, home_team_id=1, away_team_id=2,
                    match_date=datetime(2026, 9, 1, 18, 0), league="x/y"))
        s.add(_pick_row(1))
        s.add(_obs(1, "model"))
        s.add(_obs(1, "final"))
        s.commit()
    with mgr.get_session() as s:
        assert s.query(PickObservation).count() == 2


def test_attribution_is_constrained_to_model_and_final(tmp_path):
    """A third value would silently create a series nothing reports on."""
    mgr = _mgr(tmp_path, "c.db")
    with mgr.get_session() as s:
        s.add(Match(id=1, home_team_id=1, away_team_id=2,
                    match_date=datetime(2026, 9, 1, 18, 0), league="x/y"))
        s.add(_pick_row(1))
        s.commit()

    with pytest.raises(IntegrityError):
        with mgr.get_session() as s:
            s.add(_obs(1, "original"))
            s.commit()


def test_deleting_a_pick_cascades_to_its_observations(tmp_path):
    """The review's consolidation branch deletes a pick; its observations must
    not survive as orphans that the report would still count.

    Stage 13, defect A6 — THIS TEST USED TO PASS VACUOUSLY.

    It previously deleted via ``session.execute(SavedPick.__table__.delete())``,
    a Core statement that goes straight to the database and lets its
    ON DELETE CASCADE fire. Production does not do that. It calls
    ``session.delete(obj)`` (match_briefing.py), which routes through the ORM's
    unit of work — and that path was broken for four days: the default cascade
    on the ``observations`` relationship emitted
    ``UPDATE pick_observations SET pick_id = NULL`` against a NOT NULL column,
    so every consolidation rolled back. The test asserted the schema while the
    code path it was named after could not run at all.

    It now deletes the way production deletes. Verified to FAIL against the
    pre-fix relationship.
    """
    mgr = _mgr(tmp_path, "fk.db")
    with mgr.get_session() as s:
        s.add(Match(id=1, home_team_id=1, away_team_id=2,
                    match_date=datetime(2026, 9, 1, 18, 0), league="x/y"))
        s.add(_pick_row(1))
        s.add(_obs(1, "model"))
        s.add(_obs(1, "final"))
        s.commit()

    with mgr.get_session() as s:
        s.delete(s.get(SavedPick, 1))          # the production path
        s.commit()
    with mgr.get_session() as s:
        assert s.query(SavedPick).count() == 0
        assert s.query(PickObservation).count() == 0, (
            "observations survived an ORM delete of their pick")


def test_orm_delete_of_a_pick_does_not_null_the_observation_fk(tmp_path):
    """The precise failure mode, pinned separately.

    A cascade misconfiguration does not announce itself — it surfaces as an
    IntegrityError from somewhere else entirely, which match_briefing then
    swallowed as "Could not apply briefing decision". Assert that no UPDATE to
    NULL is attempted, by asserting the delete simply succeeds.
    """
    mgr = _mgr(tmp_path, "fk2.db")
    with mgr.get_session() as s:
        for pid in (1, 2):
            s.add(Match(id=pid, home_team_id=1, away_team_id=2,
                        match_date=datetime(2026, 9, 1, 18, 0), league="x/y"))
            s.add(_pick_row(pid, match_id=pid))
            s.flush()
            s.add(_obs(pid, "model"))
            s.add(_obs(pid, "final"))
        s.commit()

    # Delete one of two picks — the sibling and its observations must survive.
    with mgr.get_session() as s:
        s.delete(s.get(SavedPick, 1))
        s.commit()

    with mgr.get_session() as s:
        assert s.query(SavedPick).count() == 1
        remaining = s.query(PickObservation).all()
        assert len(remaining) == 2
        assert {o.pick_id for o in remaining} == {2}
        assert all(o.pick_id is not None for o in remaining)


def test_migration_sql_is_additive_and_does_not_backfill():
    """§4 and §13. Read the migration, not a description of it."""
    import pathlib

    sql = pathlib.Path("migrations/006_pick_observations.sql").read_text(
        encoding="utf-8")
    lowered = sql.lower()
    assert "create table if not exists pick_observations" in lowered
    assert "check (attribution in ('model', 'final'))" in lowered
    assert "unique (pick_id, attribution)" in lowered
    assert "on delete cascade" in lowered
    # Nothing that would touch saved_picks or seed rows.
    # Scan STATEMENTS, not prose: the header comment legitimately uses words
    # like "renamed" while explaining what the migration does not do.
    statements = "\n".join(
        ln for ln in lowered.splitlines() if not ln.strip().startswith("--"))
    for forbidden in ("alter table saved_picks", "drop column", "rename",
                      "insert into pick_observations", "update saved_picks",
                      "delete from"):
        assert forbidden not in statements, (
            f"migration 006 contains {forbidden!r}")
    assert pathlib.Path(
        "migrations/006_pick_observations.rollback.sql").exists()


# ═════════════════════ §5/§6 — MODEL is written before the review

def test_save_picks_writes_both_observations_before_any_review(tmp_path,
                                                               monkeypatch):
    """The timing rule. `_save_picks` runs inside get_daily_picks; the review
    runs after it. If the MODEL row were written later it could not exist."""
    from types import SimpleNamespace

    from src.agent.betting_agent import FootballBettingAgent

    mgr = _mgr(tmp_path, "save.db")
    with mgr.get_session() as s:
        s.add(Match(id=1, home_team_id=1, away_team_id=2,
                    match_date=datetime(2026, 9, 1, 18, 0), league="x/y"))
        s.commit()

    agent = FootballBettingAgent.__new__(FootballBettingAgent)
    agent.db = mgr
    agent.config = SimpleNamespace(get=lambda k, d=None: d)

    pick = SimpleNamespace(
        match_id=1, match="A vs B", league="x/y", market="Over 2.5",
        selection="Over 2.5 Goals", odds=1.85, predicted_probability=0.6,
        expected_value=0.11, confidence=0.6, kelly_stake_percentage=1.0,
        risk_level="medium", used_fallback_odds=False,
        model_agreement="unanimous", market_probability=0.55, market_books=8)

    agent._save_picks([pick], date(2026, 9, 1))

    with mgr.get_session() as s:
        rows = {o.attribution: o for o in s.query(PickObservation).all()}
    assert set(rows) == {"model", "final"}
    for o in rows.values():
        assert o.market == "Over 2.5"
        assert o.selection == "Over 2.5 Goals"
        assert o.taken_odds == pytest.approx(1.85)
        assert o.closing_status == "pending"


def test_observation_failure_rolls_the_pick_back(tmp_path, monkeypatch):
    """Stage 10.3 replaced Stage 10's behaviour here.

    Stage 10 wrapped the observation write in a SAVEPOINT and swallowed errors,
    so a bookkeeping failure could not cost a pick. Stage 10.1 measured what
    that bought: picks committed with no observations. Such a pick is
    indistinguishable in the report from one that was never measurable, and the
    model's taken price is unrecoverable once the review overwrites it — so
    losing the pick is strictly better than keeping an unattributable one.
    """
    from types import SimpleNamespace

    import src.agent.betting_agent as ba
    from src.agent.betting_agent import FootballBettingAgent

    mgr = _mgr(tmp_path, "fail.db")
    with mgr.get_session() as s:
        s.add(Match(id=1, home_team_id=1, away_team_id=2,
                    match_date=datetime(2026, 9, 1, 18, 0), league="x/y"))
        s.commit()

    def _boom(*a, **k):
        raise RuntimeError("observation write failed")

    monkeypatch.setattr(ba, "_write_pick_observations", _boom)

    agent = FootballBettingAgent.__new__(FootballBettingAgent)
    agent.db = mgr
    agent.config = SimpleNamespace(get=lambda k, d=None: d)
    pick = SimpleNamespace(
        match_id=1, match="A vs B", league="x/y", market="1X2",
        selection="Home Win", odds=2.0, predicted_probability=0.55,
        expected_value=0.1, confidence=0.55, kelly_stake_percentage=1.0,
        risk_level="medium", used_fallback_odds=False,
        model_agreement="unanimous", market_probability=0.5, market_books=8)

    with pytest.raises(RuntimeError):
        agent._save_picks([pick], date(2026, 9, 1))

    # The invariant is the database state, not the exception.
    with mgr.get_session() as s:
        assert s.query(SavedPick).count() == 0, (
            "a pick survived a failed observation write")
        assert s.query(PickObservation).count() == 0


# ══════════════ §9/§18 — THE MANDATORY CHANGE ACCEPTANCE TEST

def test_stage10_acceptance_change_preserves_both_taken_prices(tmp_path,
                                                               monkeypatch):
    """Stage 10 §18, end to end. This is the acceptance test.

    Reproduces the exact Stage 9 blocker: the frozen model takes Over 2.5 @1.85,
    Claude switches to Home Win @2.10, and the model's price must still be 1.85
    afterwards even though `saved_picks.odds` now reads 2.10.
    """
    from src.reporting.match_briefing import _update_final_observation

    mgr = _mgr(tmp_path, "acc.db")
    now = datetime.utcnow()
    kickoff = now + timedelta(minutes=45)
    taken_at = now - timedelta(hours=3)

    # 1-2. Frozen model produces the pick; both observations are written.
    with mgr.get_session() as s:
        s.add(Match(id=1, home_team_id=1, away_team_id=2, match_date=kickoff,
                    league="x/y", is_fixture=True))
        s.add(_pick_row(1, market="Over 2.5", selection="Over 2.5 Goals",
                        odds=1.85, model_market="Over 2.5",
                        model_selection="Over 2.5 Goals",
                        pick_date=kickoff.date(), created_at=taken_at))
        s.add(_obs(1, "model", "Over 2.5", "Over 2.5 Goals", 1.85, taken_at))
        s.add(_obs(1, "final", "Over 2.5", "Over 2.5 Goals", 1.85, taken_at))
        s.commit()

    # 3-4. Claude CHANGEs to Home Win @2.10 — saved_picks.odds is overwritten
    # and the FINAL observation moves with it.
    with mgr.get_session() as s:
        pick = s.get(SavedPick, 1)
        pick.market, pick.selection, pick.odds = "1X2", "Home Win", 2.10
        pick.review_action = "CHANGE"
        _update_final_observation(s, 1, market="1X2", selection="Home Win",
                                  taken_odds=2.10)
        s.commit()

    # 5-6. The prices.
    with mgr.get_session() as s:
        rows = {o.attribution: o for o in s.query(PickObservation).all()}
        assert rows["model"].taken_odds == pytest.approx(1.85)
        assert rows["model"].selection == "Over 2.5 Goals"
        assert rows["model"].market == "Over 2.5"
        assert rows["final"].taken_odds == pytest.approx(2.10)
        assert rows["final"].selection == "Home Win"
        assert rows["final"].market == "1X2"
        # saved_picks carries the FINAL price; MODEL survives only here.
        assert s.get(SavedPick, 1).odds == pytest.approx(2.10)
        # taken_at is NOT moved — both series share one causal boundary.
        assert rows["final"].taken_at == taken_at

    # 7. Closing capture resolves the two selections independently, from odds
    # observed after the pick was taken.
    observed = now - timedelta(minutes=5)
    with mgr.get_session() as s:
        for sel, o in (("Home", 2.00), ("Draw", 3.50), ("Away", 4.20)):
            s.add(Odds(match_id=1, bookmaker="Pinnacle", market_type="1X2",
                       selection=sel, odds_value=o, timestamp=observed))
        for sel, o in (("Over 2.5", 1.70), ("Under 2.5", 2.20)):
            s.add(Odds(match_id=1, bookmaker="Pinnacle",
                       market_type="over_under", selection=sel, odds_value=o,
                       timestamp=observed))
        s.commit()

    from scripts import capture_closing_lines as cap
    monkeypatch.setattr(cap, "get_db", lambda: mgr)
    stats = cap.capture(within_minutes=90)

    # 8-9. Each series uses its OWN taken price and its OWN close.
    with mgr.get_session() as s:
        rows = {o.attribution: o for o in s.query(PickObservation).all()}

    m, f = rows["model"], rows["final"]
    assert m.closing_status == "captured" and f.closing_status == "captured"
    assert m.closing_odds == pytest.approx(1.70), "model close came from 1X2"
    assert f.closing_odds == pytest.approx(2.00), "final close came from totals"
    assert m.closing_odds != f.closing_odds, "one close served both series"

    model_clv = m.taken_odds / m.closing_odds - 1
    final_clv = f.taken_odds / f.closing_odds - 1
    assert model_clv == pytest.approx(1.85 / 1.70 - 1)
    assert final_clv == pytest.approx(2.10 / 2.00 - 1)

    # 10. Two distinct selections → two resolutions. Not more.
    assert stats["observations_considered"] == 2
    assert stats["observations_resolved"] == 2


def test_unchanged_pick_resolves_one_underlying_observation(tmp_path,
                                                            monkeypatch):
    """§8 and §11: KEEP yields two attribution records off ONE resolution."""
    mgr = _mgr(tmp_path, "keep.db")
    now = datetime.utcnow()
    kickoff = now + timedelta(minutes=45)
    taken_at = now - timedelta(hours=3)
    observed = now - timedelta(minutes=5)

    with mgr.get_session() as s:
        s.add(Match(id=1, home_team_id=1, away_team_id=2, match_date=kickoff,
                    league="x/y", is_fixture=True))
        s.add(_pick_row(1, market="Over 2.5", selection="Over 2.5 Goals",
                        odds=1.85, review_action="KEEP",
                        model_market="Over 2.5",
                        model_selection="Over 2.5 Goals",
                        pick_date=kickoff.date(), created_at=taken_at))
        s.add(_obs(1, "model", "Over 2.5", "Over 2.5 Goals", 1.85, taken_at))
        s.add(_obs(1, "final", "Over 2.5", "Over 2.5 Goals", 1.85, taken_at))
        for sel, o in (("Over 2.5", 1.75), ("Under 2.5", 2.15)):
            s.add(Odds(match_id=1, bookmaker="Pinnacle",
                       market_type="over_under", selection=sel, odds_value=o,
                       timestamp=observed))
        s.commit()

    from scripts import capture_closing_lines as cap
    monkeypatch.setattr(cap, "get_db", lambda: mgr)
    stats = cap.capture(within_minutes=90)

    assert stats["observations_considered"] == 2
    assert stats["observations_resolved"] == 1, (
        "an unchanged pick was resolved twice — MODEL and FINAL name the same "
        "bet and must share ONE underlying observation")

    with mgr.get_session() as s:
        rows = {o.attribution: o for o in s.query(PickObservation).all()}
    assert rows["model"].closing_odds == rows["final"].closing_odds == 1.75


# ══════════════════════════ §12 — the Stage 8 rules still bind

@pytest.mark.parametrize("offset_minutes,expect", [
    (-5, "missing"),   # observed before the pick
    (0, "missing"),    # observed AT the pick — the same-snapshot rule
    (+5, "captured"),  # observed after the pick
])
def test_observations_obey_the_same_snapshot_rule(tmp_path, monkeypatch,
                                                  offset_minutes, expect):
    mgr = _mgr(tmp_path, f"snap{offset_minutes}.db")
    now = datetime.utcnow()
    kickoff = now + timedelta(minutes=45)
    taken_at = now - timedelta(minutes=30)
    observed = taken_at + timedelta(minutes=offset_minutes)

    with mgr.get_session() as s:
        s.add(Match(id=1, home_team_id=1, away_team_id=2, match_date=kickoff,
                    league="x/y", is_fixture=True))
        s.add(_pick_row(1, pick_date=kickoff.date(), created_at=taken_at))
        s.add(_obs(1, "model", "1X2", "Home Win", 2.0, taken_at))
        s.add(_obs(1, "final", "1X2", "Home Win", 2.0, taken_at))
        for sel, o in (("Home", 1.95), ("Draw", 3.50), ("Away", 4.20)):
            s.add(Odds(match_id=1, bookmaker="Pinnacle", market_type="1X2",
                       selection=sel, odds_value=o, timestamp=observed))
        s.commit()

    from scripts import capture_closing_lines as cap
    monkeypatch.setattr(cap, "get_db", lambda: mgr)
    cap.capture(within_minutes=90)

    with mgr.get_session() as s:
        for o in s.query(PickObservation).all():
            assert o.closing_status == expect, f"{o.attribution}: {o.closing_status}"


def test_a_change_invalidates_a_close_taken_on_the_old_selection(tmp_path):
    """The old close priced a different bet. Keeping it would attribute one
    selection's market movement to another."""
    from src.reporting.match_briefing import _update_final_observation

    mgr = _mgr(tmp_path, "inval.db")
    with mgr.get_session() as s:
        s.add(Match(id=1, home_team_id=1, away_team_id=2,
                    match_date=datetime(2026, 9, 1, 18, 0), league="x/y"))
        s.add(_pick_row(1))
        o = _obs(1, "final")
        o.closing_odds, o.closing_status, o.closing_book_count = 1.9, "captured", 5
        s.add(o)
        s.add(_obs(1, "model"))
        s.commit()

    with mgr.get_session() as s:
        _update_final_observation(s, 1, market="Over 2.5",
                                  selection="Over 2.5 Goals", taken_odds=1.7)
        s.commit()

    with mgr.get_session() as s:
        rows = {x.attribution: x for x in s.query(PickObservation).all()}
    assert rows["final"].closing_odds is None
    assert rows["final"].closing_status == "pending"
    assert rows["model"].taken_odds == pytest.approx(2.0), "MODEL was touched"


# ═══════════════════════ §14/§16 — reporting reads the new table

def test_report_prefers_observations_and_keeps_series_separate(tmp_path,
                                                               monkeypatch):
    import scripts.paper_trading_report as rep
    import src.data.database as dbm

    mgr = _mgr(tmp_path, "rep.db")
    taken_at = datetime(2026, 9, 1, 12, 0)
    with mgr.get_session() as s:
        s.add(Match(id=1, home_team_id=1, away_team_id=2,
                    match_date=datetime(2026, 9, 1, 18, 0), league="x/y"))
        # saved_picks carries the FINAL price only.
        s.add(_pick_row(1, market="1X2", selection="Home Win", odds=2.10,
                        closing_odds=2.00, closing_capture_status="captured",
                        review_action="CHANGE", model_market="Over 2.5",
                        model_selection="Over 2.5 Goals", created_at=taken_at))
        mo = _obs(1, "model", "Over 2.5", "Over 2.5 Goals", 1.85, taken_at)
        mo.closing_odds, mo.closing_status = 1.70, "captured"
        fo = _obs(1, "final", "1X2", "Home Win", 2.10, taken_at)
        fo.closing_odds, fo.closing_status = 2.00, "captured"
        s.add_all([mo, fo])
        s.commit()

    monkeypatch.setattr(dbm, "get_db", lambda: mgr)
    picks = rep.load_picks(days=3650, include_live=True, model_version=None)
    assert len(picks) == 1
    p = picks[0]

    # Stage 9 alone could not measure MODEL here; Stage 10 can.
    assert rep.series_clv(p, rep.MODEL) == pytest.approx(1.85 / 1.70 - 1)
    assert rep.series_clv(p, rep.FINAL) == pytest.approx(2.10 / 2.00 - 1)


def test_report_falls_back_to_saved_picks_without_observations(tmp_path,
                                                               monkeypatch):
    """Historical picks have no observations and must still report a FINAL
    series — and no MODEL series where the selection changed."""
    import scripts.paper_trading_report as rep
    import src.data.database as dbm

    mgr = _mgr(tmp_path, "fb.db")
    with mgr.get_session() as s:
        s.add(Match(id=1, home_team_id=1, away_team_id=2,
                    match_date=datetime(2026, 9, 1, 18, 0), league="x/y"))
        s.add(_pick_row(1, odds=2.10, closing_odds=2.00,
                        closing_capture_status="captured",
                        model_market=None, model_selection=None))
        s.commit()

    monkeypatch.setattr(dbm, "get_db", lambda: mgr)
    p = rep.load_picks(days=3650, include_live=True, model_version=None)[0]
    assert p.observations == {}
    assert rep.series_clv(p, rep.FINAL) == pytest.approx(2.10 / 2.00 - 1)
    assert rep.series_clv(p, rep.MODEL) is None


def test_model_version_and_code_revision_unchanged():
    """Stage 10 is storage + evaluation. It must not bump either."""
    from src.models.model_version import CODE_REVISION, model_version
    from src.utils.config import Config

    from tests.test_config_identity import FROZEN_MODEL_VERSION

    # Stage 10.2: the DEPLOYED config is the frozen subject. The previous
    # version read the gitignored local file.
    assert CODE_REVISION == "s5.2"
    assert model_version(Config("config/config.example.yaml")) == \
        FROZEN_MODEL_VERSION


# ═════ Stage 13 Step 1b — consolidation must never destroy the model's evidence

def _consolidation_env(tmp_path, name, other_selection):
    """A match with two picks, each with both observations."""
    mgr = _mgr(tmp_path, name)
    with mgr.get_session() as s:
        s.add(Match(id=1, home_team_id=1, away_team_id=2,
                    match_date=datetime(2026, 9, 1, 18, 0), league="x/y"))
        # primary = highest EV, the one the review binds to
        s.add(_pick_row(1, match_id=1, market="Team Goals",
                        selection="Home Over 0.5", odds=1.72,
                        expected_value=0.20, model_market="Team Goals",
                        model_selection="Home Over 0.5"))
        s.add(_pick_row(2, match_id=1, market="1X2",
                        selection=other_selection, odds=2.10,
                        expected_value=0.05, model_market="1X2",
                        model_selection=other_selection))
        s.flush()
        s.add(_obs(1, "model", "Team Goals", "Home Over 0.5", 1.72))
        s.add(_obs(1, "final", "Team Goals", "Home Over 0.5", 1.72))
        s.add(_obs(2, "model", "1X2", other_selection, 2.10))
        s.add(_obs(2, "final", "1X2", other_selection, 2.10))
        s.commit()
    return mgr


def _model_obs_count(mgr):
    with mgr.get_session() as s:
        return s.query(PickObservation).filter(
            PickObservation.attribution == "model").count()


def test_consolidation_never_reduces_model_observations(tmp_path):
    """The invariant Step 1b exists to enforce.

    Fixing the ORM cascade (Step 1) made `session.delete(primary)` in the
    review's consolidation branch actually work — and with the cascade it would
    have taken the primary's MODEL observation with it. That row is the frozen
    model's only record of what it selected and at what price; the odds table
    keeps one row per (match, bookmaker, market, selection) and overwrites it,
    so nothing can rebuild it.

    The branch must mark the pick superseded, never remove it.
    """
    from src.reporting.match_briefing import MatchBriefingService

    mgr = _consolidation_env(tmp_path, "cons.db", "Home Win")
    before = _model_obs_count(mgr)
    assert before == 2

    # Reproduce the branch's effect exactly as _apply_decision now performs it.
    with mgr.get_session() as s:
        primary = s.get(SavedPick, 1)
        other = s.get(SavedPick, 2)
        other.review_action = "CHANGE"
        primary.disposition = "consolidated"
        s.commit()

    assert _model_obs_count(mgr) == before, (
        "consolidation destroyed a MODEL observation — the frozen model's "
        "record of its own selection")

    with mgr.get_session() as s:
        assert s.query(SavedPick).count() == 2, "a pick was deleted"
        p = s.get(SavedPick, 1)
        assert p.disposition == "consolidated"
        assert p.result is None, "disposition must not be written to `result`"
        assert p.review_action != "CONSOLIDATED", (
            "disposition must not be overloaded onto review_action")


def test_consolidation_branch_does_not_delete_in_source():
    """Structural: the branch must not call session.delete at all.

    Asserting behaviour is not enough here — a future edit could reintroduce
    the delete on a path this test does not drive.
    """
    import inspect

    from src.reporting.match_briefing import MatchBriefingService

    src = inspect.getsource(MatchBriefingService._apply_decision)
    # Scan CODE, not prose: the branch's comment legitimately quotes the call it
    # replaced, to explain why it must never come back.
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    assert "session.delete" not in code, (
        "_apply_decision deletes a SavedPick again — that destroys its "
        "pick_observations, including the MODEL row")
    assert 'disposition = "consolidated"' in code


def test_superseded_pick_is_excluded_from_the_live_record(tmp_path):
    """It was never a bet. Its sibling carries the stake; counting both would
    double-weight one wager in ROI and in the EV-threshold calibrator."""
    from src.agent.betting_agent import _live_only

    mgr = _consolidation_env(tmp_path, "cons2.db", "Home Win")
    with mgr.get_session() as s:
        s.get(SavedPick, 1).disposition = "consolidated"
        s.query(SavedPick).update({SavedPick.result: "win"})
        s.commit()

    with mgr.get_session() as s:
        live = s.query(SavedPick).filter(_live_only()).all()
        assert [p.id for p in live] == [2], (
            "a superseded pick reached the live record")


def test_superseded_pick_keeps_its_model_series_but_not_final(tmp_path,
                                                              monkeypatch):
    """The asymmetry, stated as a test: the frozen model really did select it
    at a real price (MODEL counts); it was never taken (FINAL does not)."""
    import scripts.paper_trading_report as rep
    import src.data.database as dbm

    mgr = _consolidation_env(tmp_path, "cons3.db", "Home Win")
    with mgr.get_session() as s:
        s.get(SavedPick, 1).disposition = "consolidated"
        for o in s.query(PickObservation).filter(
                PickObservation.pick_id == 1).all():
            o.closing_odds, o.closing_status = 1.60, "captured"
        s.commit()

    monkeypatch.setattr(dbm, "get_db", lambda: mgr)
    picks = {p.id: p for p in rep.load_picks(days=3650, include_live=True,
                                             model_version=None)}
    superseded = picks[1]
    assert superseded.disposition == "consolidated"
    assert rep.series_clv(superseded, rep.MODEL) is not None, (
        "the MODEL series lost a selection the frozen model genuinely made")
    assert rep.series_clv(superseded, rep.FINAL) is None, (
        "a superseded pick was counted as a taken bet in the FINAL series")
