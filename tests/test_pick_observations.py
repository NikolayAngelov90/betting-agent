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

from tests.experiment_pins import CODE_REVISION_PIN
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
    assert CODE_REVISION == CODE_REVISION_PIN
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


def test_no_bulk_core_delete_of_a_parent_entity():
    """Stage 13 Step 1c — the half the session.delete guard does not cover.

    `session.query(SavedPick).filter(...).delete()` is a bulk Core DELETE. It
    loads no objects, so the ORM's cascade never runs. Measured, with SQLite's
    foreign_keys pragma OFF (the state this repo was in until Step 1c):

        before: picks=1 observations=2
        bulk delete: SUCCEEDED
        after : picks=0 observations=2 ORPHANS=2

    No exception, no rollback — silently orphaned rows pointing at a pick_id
    that no longer exists. Strictly worse than the crash Step 1 fixed, and
    invisible to any invariant that counts orphans only after a delete raises.
    Turning the pragma on closes it (the DDL cascade fires), but that is one
    line of connection setup away from regressing, so the call shape is banned
    on parent entities as well.

    Child-table bulk deletes stay allowed and are used deliberately —
    flashscore_scraper drops a match's Flashscore odds rows before rewriting
    them, injury_scraper clears a team's stale injuries. Nothing hangs off
    `odds` or `injuries`, so there is no cascade to skip: the danger is
    specific to entities that OWN dependent rows.
    """
    import pathlib
    import re

    PARENTS = ("SavedPick", "Match", "Team")
    offenders = []
    sources = (list(pathlib.Path("src").rglob("*.py"))
               + list(pathlib.Path("scripts").rglob("*.py")))
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            code = line.split("#", 1)[0]
            for parent in PARENTS:
                if re.search(rf"query\(\s*{parent}\s*\).*\.delete\(", code):
                    offenders.append(f"{path}:{i}: {line.strip()}")
                if re.search(rf"{parent}\.__table__\.delete\(", code):
                    offenders.append(f"{path}:{i}: {line.strip()}")

    assert not offenders, (
        "bulk Core delete on a parent entity — the ORM cascade is skipped and "
        "dependent rows are orphaned without raising:\n  "
        + "\n  ".join(offenders))


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

# ═══════════════════════════════ Step 1c item 3 — the disposition contract

def test_disposition_is_write_once(tmp_path):
    """Like is_paper and model_version — but enforced, not merely arranged.

    Those two are write-once because exactly one site writes them, at insert.
    `disposition` is NULL at insert and set later, so the same guarantee has to
    be a validator. Overwriting it would let a supersession be relabelled a
    void (or the reverse) with the original reason unrecoverable.
    """
    mgr = _mgr(tmp_path, "wo.db")
    with mgr.get_session() as s:
        s.add(Match(id=1, home_team_id=1, away_team_id=2,
                    match_date=datetime(2026, 9, 1, 18, 0), league="x/y"))
        s.add(_pick_row(1))
        s.commit()

    with mgr.get_session() as s:
        p = s.get(SavedPick, 1)
        p.disposition = "consolidated"
        s.commit()

    with mgr.get_session() as s:
        p = s.get(SavedPick, 1)
        p.disposition = "consolidated"          # idempotent replay: allowed
        with pytest.raises(ValueError, match="write-once"):
            p.disposition = "void_wrong_fixture"
        with pytest.raises(ValueError, match="write-once"):
            p.disposition = None                # nor back to "it was a bet"


def test_paired_clv_pairs_by_pick_not_by_fixture():
    """Step 1c item 3. Both halves of a pair must come from ONE pick row.

    After a consolidation a fixture carries 2 MODEL observations and 1 FINAL.
    Pairing by match_id would marry the orphaned MODEL to the survivor's FINAL
    and report the difference between two DIFFERENT bets as the review's
    within-pick effect — a number with no meaning, in the direction that
    flatters whichever selection happened to close better.
    """
    import inspect
    import re

    from scripts.paper_trading_report import section_clv

    src = inspect.getsource(section_clv)
    body = chr(10).join(ln.split("#", 1)[0] for ln in src.splitlines())

    # The pair is appended inside `for p in picks:` and both members are
    # resolved from that same `p`. Assert the shape, since a later refactor to
    # a match_id-keyed dict is exactly the mistake this guards.
    loop = body[body.index("for p in picks"):body.index("paired.append")]
    assert re.search(r"mc\s*=\s*series_clv\(\s*p\s*,", loop)
    assert re.search(r"fc\s*=\s*series_clv\(\s*p\s*,", loop)
    assert "paired.append((p.match_id, mc, fc))" in body, (
        "the paired subset no longer appends one pick's own two measurements")
    # Guarded by both: the append is conditional on BOTH being present.
    assert "if mc is not None and fc is not None:" in body


def test_clustering_survives_two_model_and_one_final_on_one_fixture():
    """Neither double-counted nor dropped.

    A consolidated fixture contributes 2 MODEL observations and 1 FINAL. The
    MODEL series must treat them as ONE cluster (paying a design effect for
    the doubling), and the cluster bootstrap must still accept the sample.
    """
    from scripts.paper_trading_report import _boot, _effective_n

    model_fx = [77, 77] + list(range(100, 108))   # 8 ordinary one-pick fixtures
    final_fx = [77] + list(range(100, 108))

    n, k, deff, n_eff = _effective_n(model_fx)
    assert (n, k) == (10, 9), "fixture 77 must be one cluster, not two"
    assert deff > 1.0 and n_eff < n, (
        "two picks on one fixture must cost effective sample size")

    n2, k2, deff2, n_eff2 = _effective_n(final_fx)
    assert (n2, k2) == (9, 9) and deff2 == 1.0 and n_eff2 == 9

    lo, hi = _boot([0.01, 0.02] + [0.0] * 8, clusters=model_fx)
    assert lo is not None, "the doubled cluster was dropped from the bootstrap"
