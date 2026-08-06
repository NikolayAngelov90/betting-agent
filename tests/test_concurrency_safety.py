"""Races that fire the moment the pipeline runs more than one worker.

Today `concurrency: cancel-in-progress: false` serialises workflow runs, which is
the only reason these have not bitten. Each test here drives the interleaving
directly rather than hoping to catch it by luck.
"""

from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import IntegrityError

from src.data.database import DatabaseManager
from src.data.models import Match, Odds, SavedPick, Team


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    mgr = DatabaseManager(config=SimpleNamespace(
        database={"sqlite_path": str(tmp_path / "race.db")}))
    assert not mgr.is_postgres, "test DB must be SQLite, not production Postgres"
    mgr.create_tables()
    return mgr


def _fixture(db):
    with db.get_session() as s:
        h, a = Team(name="Home"), Team(name="Away")
        s.add_all([h, a]); s.flush()
        m = Match(home_team_id=h.id, away_team_id=a.id,
                  match_date=datetime(2026, 8, 10, 18, 0),
                  league="test/league", is_fixture=True)
        s.add(m); s.flush()
        return m.id


def _pick_values(match_id, selection="Home Win", **kw):
    base = dict(
        match_id=match_id, pick_date=date(2026, 8, 10), match_name="Home vs Away",
        league="test/league", market="1X2", selection=selection, odds=2.0,
        predicted_probability=0.55, expected_value=0.1, confidence=0.55,
        kelly_stake_percentage=2.0, risk_level="medium", used_fallback_odds=False,
        model_agreement="unanimous", model_market="1X2", model_selection=selection,
        model_probability=0.55, created_at=datetime(2026, 8, 10, 9, 0),
    )
    base.update(kw)
    return base


class TestSavedPickDedup:
    """S2-1: dedup was a read-then-write both workers passed."""

    def test_the_database_now_rejects_a_duplicate(self, db):
        mid = _fixture(db)
        with db.get_session() as s:
            s.add(SavedPick(**_pick_values(mid)))
        with pytest.raises(IntegrityError):
            with db.get_session() as s:
                s.add(SavedPick(**_pick_values(mid)))

    def test_interleaved_workers_produce_exactly_one_pick(self, db):
        """The real race, driven explicitly.

        Both sessions read "no duplicate" BEFORE either writes — precisely what
        two workers do. Without the constraint both inserts land and every
        downstream statistic (win rate, ROI, Brier, stake sizing) is skewed.
        """
        from src.agent.betting_agent import _insert_pick_if_absent
        mid = _fixture(db)
        values = _pick_values(mid)

        s1 = db.SessionLocal()
        s2 = db.SessionLocal()
        try:
            # Both check first — both see nothing.
            assert s1.query(SavedPick).filter_by(
                match_id=mid, selection="Home Win").first() is None
            assert s2.query(SavedPick).filter_by(
                match_id=mid, selection="Home Win").first() is None

            first = _insert_pick_if_absent(s1, values)
            s1.commit()
            second = _insert_pick_if_absent(s2, values)
            s2.commit()

            assert first is not None, "the first writer should have inserted"
            assert second is None, "the second writer must be told it lost"
        finally:
            s1.close()
            s2.close()

        with db.get_session() as s:
            assert s.query(SavedPick).count() == 1

    def test_a_different_selection_on_the_same_match_is_allowed(self, db):
        """The constraint must not block the legitimate two-picks-per-match case."""
        from src.agent.betting_agent import _insert_pick_if_absent
        mid = _fixture(db)
        with db.get_session() as s:
            assert _insert_pick_if_absent(s, _pick_values(mid, selection="Home Win"))
            assert _insert_pick_if_absent(s, _pick_values(mid, selection="Over 2.5 Goals"))
        with db.get_session() as s:
            assert s.query(SavedPick).count() == 2

    def test_the_same_pick_on_a_later_date_is_allowed(self, db):
        from src.agent.betting_agent import _insert_pick_if_absent
        mid = _fixture(db)
        with db.get_session() as s:
            assert _insert_pick_if_absent(s, _pick_values(mid))
            assert _insert_pick_if_absent(
                s, _pick_values(mid, pick_date=date(2026, 8, 11)))
        with db.get_session() as s:
            assert s.query(SavedPick).count() == 2

    def test_losing_the_race_does_not_re_notify(self, db):
        """`new_picks` drives the Telegram message.

        A worker that loses the race must not report the pick as new, or the
        user gets the same pick twice.
        """
        from src.agent.betting_agent import _insert_pick_if_absent
        mid = _fixture(db)
        with db.get_session() as s:
            _insert_pick_if_absent(s, _pick_values(mid))
        with db.get_session() as s:
            assert _insert_pick_if_absent(s, _pick_values(mid)) is None


class TestOddsUpsert:
    """S2-2: the odds writer read first, then inserted.

    `odds` DOES have a unique index. The old code built an in-memory snapshot of
    existing rows and inserted whatever was missing — so a concurrent writer
    caused a unique violation, and on PostgreSQL that aborts the entire
    transaction: every odds row for the fixture is lost and the step dies.
    """

    def _rows(self, match_id, odds_value=2.0, n=3):
        sels = ["Home", "Draw", "Away"][:n]
        return [{
            "match_id": match_id, "bookmaker": "Bet365", "market_type": "1X2",
            "selection": s, "odds_value": odds_value, "opening_odds": odds_value,
            "timestamp": datetime(2026, 8, 10, 9, 0),
        } for s in sels]

    def test_upsert_inserts_when_absent(self, db):
        from src.scrapers.apifootball_scraper import APIFootballScraper
        mid = _fixture(db)
        with db.get_session() as s:
            APIFootballScraper._upsert_odds_rows(s, self._rows(mid))
        with db.get_session() as s:
            assert s.query(Odds).count() == 3

    def test_upsert_over_a_concurrent_insert_does_not_raise(self, db):
        """The exact interleaving that used to lose the whole batch."""
        from src.scrapers.apifootball_scraper import APIFootballScraper
        mid = _fixture(db)

        # A rival worker got there first with one of the three rows.
        with db.get_session() as s:
            s.add(Odds(match_id=mid, bookmaker="Bet365", market_type="1X2",
                       selection="Draw", odds_value=3.30, opening_odds=3.30))

        with db.get_session() as s:
            APIFootballScraper._upsert_odds_rows(s, self._rows(mid, odds_value=2.5))

        with db.get_session() as s:
            rows = {o.selection: o for o in s.query(Odds).all()}
        assert set(rows) == {"Home", "Draw", "Away"}, "the batch must survive intact"
        assert rows["Home"].odds_value == 2.5
        assert rows["Draw"].odds_value == 2.5, "conflicting row should be updated"

    def test_upsert_preserves_first_seen_opening_odds(self, db):
        """opening_odds is the CLV baseline — an update must never overwrite it."""
        from src.scrapers.apifootball_scraper import APIFootballScraper
        mid = _fixture(db)
        with db.get_session() as s:
            s.add(Odds(match_id=mid, bookmaker="Bet365", market_type="1X2",
                       selection="Home", odds_value=2.20, opening_odds=2.20))

        with db.get_session() as s:
            APIFootballScraper._upsert_odds_rows(s, self._rows(mid, odds_value=1.80, n=1))

        with db.get_session() as s:
            row = s.query(Odds).filter_by(selection="Home").one()
        assert row.odds_value == 1.80, "current odds should move"
        assert row.opening_odds == 2.20, "opening odds must stay at first-seen"

    def test_upsert_backfills_opening_odds_when_missing(self, db):
        """Mirrors `if existing.opening_odds is None: existing.opening_odds = ...`."""
        from src.scrapers.apifootball_scraper import APIFootballScraper
        mid = _fixture(db)
        with db.get_session() as s:
            s.add(Odds(match_id=mid, bookmaker="Bet365", market_type="1X2",
                       selection="Home", odds_value=2.20, opening_odds=None))

        with db.get_session() as s:
            APIFootballScraper._upsert_odds_rows(s, self._rows(mid, odds_value=1.80, n=1))

        with db.get_session() as s:
            row = s.query(Odds).filter_by(selection="Home").one()
        assert row.opening_odds == 2.20, "should fall back to the prior odds_value"


class TestHistoryMirrorLock:
    """S2-3: two atomic renames are not one atomic update.

    Without a lock, process A can write parquet_A, B can write parquet_B, and
    then A can write meta_A — pairing A's watermark with B's rows. The next sync
    resumes past rows it never stored. The row-count reconcile only notices if
    those rows were inserts; for UPDATES the count matches and the stale values
    live forever.
    """

    @pytest.fixture
    def mirror_dir(self, tmp_path):
        return tmp_path / "locked"

    def _mirror(self, mirror_dir):
        from src.data.history_mirror import HistoryMirror
        return HistoryMirror(directory=mirror_dir, require_postgres=False)

    def _seed(self, db, n=30):
        base = datetime(2024, 1, 1)
        with db.get_session() as s:
            t = [Team(name=f"L{i}") for i in range(4)]
            s.add_all(t); s.flush()
            ids = [x.id for x in t]
            for i in range(n):
                s.add(Match(home_team_id=ids[i % 4], away_team_id=ids[(i + 1) % 4],
                            match_date=base + timedelta(days=i), league="lock/league",
                            home_goals=i % 3, away_goals=1, home_xg=1.0,
                            is_fixture=False))

    def test_lock_serialises_two_syncs(self, db, mirror_dir):
        """A second sync must wait rather than interleave."""
        from src.data.history_mirror import _SyncLock
        self._seed(db)
        m = self._mirror(mirror_dir)
        m.sync(db)

        lock_path = mirror_dir / "match_history.parquet.lock"
        held = _SyncLock(lock_path, timeout=0.5)
        with held:
            # A rival holding the lock forces the next attempt to time out and
            # proceed rather than deadlock — the pipeline must never hang.
            rival = _SyncLock(lock_path, timeout=0.5)
            with rival:
                pass   # returns after the timeout warning, does not raise

    def test_a_concurrent_update_is_never_lost(self, db, mirror_dir):
        """The failure the lock exists to prevent, end to end.

        An UPDATE (not an insert) is the dangerous case: row counts stay equal,
        so the reconcile cannot save us. The mirror must still converge.
        """
        self._seed(db)
        m = self._mirror(mirror_dir)
        m.sync(db)

        with db.get_session() as s:
            row = s.query(Match).order_by(Match.id.asc()).first()
            target = row.id
            row.home_xg = 9.5

        frame = m.sync(db)
        assert frame.loc[frame["id"] == target, "home_xg"].iloc[0] == pytest.approx(9.5)

        with db.get_session() as s:
            db_count = s.query(Match).filter(
                Match.is_fixture == False, Match.home_goals.isnot(None)).count()  # noqa: E712
        assert len(frame) == db_count, "row counts must still agree"

    def test_lock_file_does_not_break_a_cold_start(self, db, mirror_dir):
        """A stale lock file left by a killed process must not block a sync."""
        mirror_dir.mkdir(parents=True, exist_ok=True)
        (mirror_dir / "match_history.parquet.lock").write_bytes(b"")
        self._seed(db)
        frame = self._mirror(mirror_dir).sync(db)
        assert len(frame) == 30


class TestApiBudget:
    """S1-2: the request budget was per-process, so it did not exist.

    `self._requests_today = 0` in __init__, never persisted. Seven CLI processes
    a day meant the "100/day" cap was really ~700, and with sharding it becomes
    N_workers x 100.
    """

    def _store(self, db, limit=10):
        from src.data.api_budget import ApiBudgetStore
        return ApiBudgetStore(db, "test-provider", limit)

    def test_claims_are_visible_to_another_process(self, db):
        """Two stores = two processes sharing one database."""
        a, b = self._store(db), self._store(db)
        assert a.claim(4) is True
        assert b.used() == 4, "the second process must see the first's spend"
        assert b.remaining() == 6

    def test_the_cap_is_global_not_per_process(self, db):
        a, b = self._store(db, limit=10), self._store(db, limit=10)
        assert a.claim(6) is True
        assert b.claim(6) is False, "6+6 exceeds the shared limit of 10"
        assert b.claim(4) is True, "the remaining 4 should still be claimable"
        assert a.used() == 10

    def test_an_oversized_claim_is_all_or_nothing(self, db):
        s = self._store(db, limit=10)
        assert s.claim(11) is False
        assert s.used() == 0, "a refused claim must not consume anything"

    def test_reserve_lowers_the_effective_ceiling(self, db):
        s = self._store(db, limit=10)
        assert s.claim(8, reserve=5) is False, "8 > 10-5"
        assert s.claim(5, reserve=5) is True

    def test_release_returns_an_unused_claim(self, db):
        s = self._store(db, limit=10)
        s.claim(5)
        s.release(2)
        assert s.used() == 3

    def test_release_cannot_go_negative(self, db):
        s = self._store(db, limit=10)
        s.claim(1)
        s.release(5)
        assert s.used() == 0

    def test_budgets_are_scoped_per_day(self, db):
        from datetime import date as _d
        s = self._store(db, limit=10)
        s.claim(10, day=_d(2026, 8, 10))
        assert s.claim(10, day=_d(2026, 8, 11)) is True, "a new day resets the budget"

    def test_missing_table_degrades_instead_of_raising(self, db, monkeypatch):
        """Migration 002 rolled back: fall back to per-process accounting."""
        s = self._store(db)
        monkeypatch.setattr(type(s), "available", lambda self: False)
        assert s.claim(999) is True, "must fail open, not block the scraper"
        assert s.used() == 0

    def test_scraper_reads_the_shared_counter(self, db, monkeypatch):
        from src.scrapers.apifootball_scraper import APIFootballScraper
        monkeypatch.setattr("src.scrapers.apifootball_scraper.get_db", lambda: db)
        one = APIFootballScraper()
        two = APIFootballScraper()
        before = one.remaining_budget()
        assert one._budget.claim(5) is True
        assert two.remaining_budget() == before - 5, (
            "a second scraper instance must see the first's spend")


class TestBatchedHotPaths:
    """S1-3, S1-4, S3-1: loops that became the wall clock at scale."""

    def _history(self, db, n_teams=6, n_matches=40):
        base = datetime(2024, 1, 1)
        with db.get_session() as s:
            teams = [Team(name=f"B{i}", league="batch/league") for i in range(n_teams)]
            s.add_all(teams); s.flush()
            ids = [t.id for t in teams]
            for i in range(n_matches):
                s.add(Match(home_team_id=ids[i % n_teams],
                            away_team_id=ids[(i + 1) % n_teams],
                            match_date=base + timedelta(days=i),
                            league="batch/league", home_goals=1, away_goals=0,
                            is_fixture=False))
        return ids

    def test_batched_team_counts_match_the_per_team_query(self, db):
        """S1-3 must produce identical numbers to the loop it replaced."""
        from sqlalchemy import or_ as _or
        ids = self._history(db)
        from src.data.sql_helpers import id_in
        from sqlalchemy import func

        with db.get_session() as s:
            reference = {
                tid: s.query(Match.id).filter(
                    Match.is_fixture == False,  # noqa: E712
                    Match.home_goals.isnot(None),
                    _or(Match.home_team_id == tid, Match.away_team_id == tid),
                ).count()
                for tid in ids
            }

            batched = {tid: 0 for tid in ids}
            for column in (Match.home_team_id, Match.away_team_id):
                for tid, n in s.query(column, func.count(Match.id)).filter(
                    Match.is_fixture == False,  # noqa: E712
                    Match.home_goals.isnot(None),
                    id_in(s, column, set(ids)),
                ).group_by(column).all():
                    if tid in batched:
                        batched[tid] += n

        assert batched == reference
        assert sum(reference.values()) == 80, "each match counts for both teams"

    def test_unanalyzable_today_still_flags_zero_history_teams(self, db, monkeypatch):
        """End-to-end check of the rewritten _unanalyzable_today."""
        from src.agent.betting_agent import FootballBettingAgent
        ids = self._history(db)

        with db.get_session() as s:
            newcomer = Team(name="Newcomer", league="batch/league")
            s.add(newcomer); s.flush()
            today = datetime.combine(date.today(), datetime.min.time())
            known = Match(home_team_id=ids[0], away_team_id=ids[1],
                          match_date=today + timedelta(hours=18),
                          league="batch/league", is_fixture=True)
            unknown = Match(home_team_id=ids[0], away_team_id=newcomer.id,
                            match_date=today + timedelta(hours=20),
                            league="batch/league", is_fixture=True)
            s.add_all([known, unknown]); s.flush()
            known_id, unknown_id = known.id, unknown.id

        agent = FootballBettingAgent.__new__(FootballBettingAgent)
        agent.db = db
        skip = agent._unanalyzable_today()

        assert unknown_id in skip, "a team with zero history must be flagged"
        assert known_id not in skip, "a well-known fixture must not be flagged"

    def test_prune_is_batched_and_spares_picked_matches(self, db):
        """S3-1: bounded batches, NOT EXISTS, and betting history preserved."""
        from src.utils.logger import utcnow
        old = utcnow() - timedelta(days=500)
        with db.get_session() as s:
            t1, t2 = Team(name="P1"), Team(name="P2")
            s.add_all([t1, t2]); s.flush()
            prunable = Match(home_team_id=t1.id, away_team_id=t2.id,
                             match_date=old, league="p/l",
                             home_goals=1, away_goals=1, is_fixture=False)
            picked = Match(home_team_id=t2.id, away_team_id=t1.id,
                           match_date=old, league="p/l",
                           home_goals=0, away_goals=0, is_fixture=False)
            recent = Match(home_team_id=t1.id, away_team_id=t2.id,
                           match_date=utcnow() - timedelta(days=5), league="p/l",
                           home_goals=2, away_goals=0, is_fixture=False)
            s.add_all([prunable, picked, recent]); s.flush()
            for m in (prunable, picked, recent):
                s.add(Odds(match_id=m.id, bookmaker="B", market_type="1X2",
                           selection="Home", odds_value=2.0))
            s.add(SavedPick(match_id=picked.id, pick_date=date(2024, 1, 1),
                            match_name="P2 vs P1", league="p/l", market="1X2",
                            selection="Home Win", odds=2.0, result="win"))
            keep_ids = {picked.id, recent.id}
            drop_id = prunable.id

        deleted = db.prune_old_odds(keep_days=400, batch_size=1)  # 1 match per batch

        assert deleted == 1, f"expected exactly one prunable row, got {deleted}"
        with db.get_session() as s:
            remaining = {o.match_id for o in s.query(Odds).all()}
        assert remaining == keep_ids
        assert drop_id not in remaining

    def test_prune_is_a_noop_when_nothing_qualifies(self, db):
        """The common case: it used to seq-scan the whole table to delete zero."""
        self._history(db)          # all matches are recent
        assert db.prune_old_odds(keep_days=400) == 0

    def test_prune_cursor_advances_past_matches_with_no_odds(self, db):
        """Progress guarantee.

        Old matches whose odds are already gone delete 0 rows. Without an id
        cursor the loop would re-select the same batch forever (or, with a
        `deleted < batch_size: break`, stop before reaching matches that DO
        still have odds). Here the prunable row sits behind 5 odds-less matches.
        """
        from src.utils.logger import utcnow
        old = utcnow() - timedelta(days=500)
        with db.get_session() as s:
            t1, t2 = Team(name="C1"), Team(name="C2")
            s.add_all([t1, t2]); s.flush()
            for _ in range(5):                      # old, no odds at all
                s.add(Match(home_team_id=t1.id, away_team_id=t2.id,
                            match_date=old, league="c/l",
                            home_goals=0, away_goals=0, is_fixture=False))
            s.flush()
            last = Match(home_team_id=t1.id, away_team_id=t2.id,
                         match_date=old, league="c/l",
                         home_goals=1, away_goals=1, is_fixture=False)
            s.add(last); s.flush()
            s.add(Odds(match_id=last.id, bookmaker="B", market_type="1X2",
                       selection="Home", odds_value=2.0))

        # One match per batch: the cursor must walk past all five empties.
        assert db.prune_old_odds(keep_days=400, batch_size=1) == 1
        with db.get_session() as s:
            assert s.query(Odds).count() == 0

    def test_prune_respects_the_batch_cap(self, db):
        """max_batches bounds the work per run; the rest waits for tomorrow."""
        from src.utils.logger import utcnow
        old = utcnow() - timedelta(days=500)
        with db.get_session() as s:
            t1, t2 = Team(name="D1"), Team(name="D2")
            s.add_all([t1, t2]); s.flush()
            for _ in range(6):
                m = Match(home_team_id=t1.id, away_team_id=t2.id,
                          match_date=old, league="d/l",
                          home_goals=0, away_goals=0, is_fixture=False)
                s.add(m); s.flush()
                s.add(Odds(match_id=m.id, bookmaker="B", market_type="1X2",
                           selection="Home", odds_value=2.0))

        assert db.prune_old_odds(keep_days=400, batch_size=1, max_batches=2) == 2
        with db.get_session() as s:
            assert s.query(Odds).count() == 4, "the rest must survive to the next run"
