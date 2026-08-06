"""Tests for the local Parquet history mirror.

The mirror only earns its place if it is *never* stale. These tests drive every
way a match row can change — inserted, completed, un-completed, restated,
deleted — plus the failure modes (interrupted write, corrupt file, schema bump)
and check both that the mirror tracks the database and that it does so
incrementally rather than by redownloading.

They run on SQLite with ``require_postgres=False``. The Postgres-only part of
the design is the trigger; the sync logic under test is dialect-independent, and
SQLAlchemy's ``onupdate=utcnow`` maintains ``updated_at`` for the ORM writes
these tests perform.
"""

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

pytest.importorskip("pyarrow", reason="Parquet engine required for the mirror")

from src.data.database import DatabaseManager
from src.data.history_mirror import HistoryMirror, MirrorUnavailable, SCHEMA_VERSION
from src.data.models import Match, Team


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    mgr = DatabaseManager(config=SimpleNamespace(
        database={"sqlite_path": str(tmp_path / "mirror.db")}))
    assert not mgr.is_postgres, "test DB must be SQLite, not production Postgres"
    mgr.create_tables()
    return mgr


@pytest.fixture
def mirror(tmp_path):
    return HistoryMirror(directory=tmp_path / "mirror", require_postgres=False)


def _seed(db, n=40):
    base = datetime(2024, 1, 1)
    with db.get_session() as s:
        teams = [Team(name=f"M{i}", league="mirror/league") for i in range(4)]
        s.add_all(teams)
        s.flush()
        ids = [t.id for t in teams]
        for i in range(n):
            s.add(Match(
                home_team_id=ids[i % 4], away_team_id=ids[(i + 1) % 4],
                match_date=base + timedelta(days=i),
                league="mirror/league", home_goals=i % 4, away_goals=(i + 1) % 3,
                home_xg=1.0 + (i % 3) * 0.2, away_xg=0.8,
                is_fixture=False,
            ))
    return ids


def _db_truth(db):
    """What the mirror should contain, read straight from the database."""
    with db.get_session() as session:
        rows = session.query(
            Match.id, Match.match_date, Match.home_team_id, Match.away_team_id,
            Match.home_goals, Match.away_goals, Match.home_xg, Match.away_xg,
            Match.league,
        ).filter(
            Match.is_fixture == False,  # noqa: E712
            Match.home_goals.isnot(None),
        ).order_by(Match.match_date.asc(), Match.id.asc()).all()
    return [tuple(r) for r in rows]


def _mirror_tuples(frame):
    frame = frame.sort_values(["match_date", "id"], kind="mergesort")
    out = []
    for r in frame.itertuples(index=False, name=None):
        out.append((
            int(r[0]), r[1].to_pydatetime() if hasattr(r[1], "to_pydatetime") else r[1],
            int(r[2]), int(r[3]), int(r[4]), int(r[5]),
            float(r[6]) if r[6] == r[6] else None,
            float(r[7]) if r[7] == r[7] else None,
            r[8],
        ))
    return out


class TestMirrorTracksTheDatabase:

    def test_first_sync_matches_the_database(self, db, mirror):
        _seed(db)
        frame = mirror.sync(db)
        assert _mirror_tuples(frame) == _db_truth(db)

    def test_new_completed_match_appears(self, db, mirror):
        ids = _seed(db)
        mirror.sync(db)

        with db.get_session() as s:
            s.add(Match(home_team_id=ids[0], away_team_id=ids[2],
                        match_date=datetime(2024, 6, 1), league="mirror/league",
                        home_goals=5, away_goals=0, is_fixture=False))

        assert _mirror_tuples(mirror.sync(db)) == _db_truth(db)

    def test_fixture_gaining_a_result_enters_the_mirror(self, db, mirror):
        ids = _seed(db)
        with db.get_session() as s:
            s.add(Match(home_team_id=ids[0], away_team_id=ids[1],
                        match_date=datetime(2024, 7, 1), league="mirror/league",
                        is_fixture=True))
        before = mirror.sync(db)

        with db.get_session() as s:
            row = s.query(Match).filter(Match.is_fixture == True).one()  # noqa: E712
            row.is_fixture = False
            row.home_goals = 2
            row.away_goals = 2

        after = mirror.sync(db)
        assert len(after) == len(before) + 1
        assert _mirror_tuples(after) == _db_truth(db)

    def test_cleared_result_leaves_the_mirror(self, db, mirror):
        _seed(db)
        before = mirror.sync(db)

        with db.get_session() as s:
            row = s.query(Match).order_by(Match.id.desc()).first()
            row.home_goals = None
            row.away_goals = None

        after = mirror.sync(db)
        assert len(after) == len(before) - 1
        assert _mirror_tuples(after) == _db_truth(db)

    def test_restated_stats_are_picked_up(self, db, mirror):
        """xG backfill rewrites rows that are already old and completed —
        the exact case created_at cannot see."""
        _seed(db)
        mirror.sync(db)

        with db.get_session() as s:
            row = s.query(Match).order_by(Match.match_date.asc()).first()
            target_id = row.id
            row.home_xg = 3.75

        frame = mirror.sync(db)
        got = frame.loc[frame["id"] == target_id, "home_xg"].iloc[0]
        assert got == pytest.approx(3.75)
        assert _mirror_tuples(frame) == _db_truth(db)

    def test_deleted_row_triggers_a_reconcile(self, db, mirror):
        """updated_at cannot record a delete; the row-count check must."""
        _seed(db)
        before = mirror.sync(db)

        with db.get_session() as s:
            s.query(Match).filter(
                Match.id == s.query(Match.id).order_by(Match.id.desc()).first()[0]
            ).delete(synchronize_session=False)

        after = mirror.sync(db)
        assert len(after) == len(before) - 1
        assert _mirror_tuples(after) == _db_truth(db)


class TestSyncIsIncremental:

    @staticmethod
    def _instrument(monkeypatch, forbid_full=True):
        """Count delta rows; optionally assert no full resync happens."""
        seen = {"delta_rows": 0, "delta_calls": 0}
        real_delta = HistoryMirror._fetch_delta

        def counting_delta(d, since):
            rows = real_delta(d, since)
            seen["delta_calls"] += 1
            seen["delta_rows"] += len(rows)
            return rows

        monkeypatch.setattr(HistoryMirror, "_fetch_delta", staticmethod(counting_delta))
        if forbid_full:
            monkeypatch.setattr(
                HistoryMirror, "_fetch_full",
                staticmethod(lambda d: (_ for _ in ()).throw(
                    AssertionError("unexpected full resync"))))
        return seen

    # The watermark is re-queried with `>=`, not `>`, so a row written in the
    # same clock tick as the previous watermark can never be skipped. The price
    # is a deliberate overlap: every row sharing the boundary timestamp comes
    # back once more. That overlap is NOT one row — `utcnow()` has sub-
    # microsecond call overhead and genuinely collides, and on PostgreSQL
    # `now()` is transaction-time so an entire write batch shares a value. It is
    # bounded by the largest single write batch, and merges idempotently by
    # primary key: cheap insurance against silent loss.
    #
    # So these tests assert the property that actually matters — the delta is
    # proportional to CHANGES, not to history size — rather than pinning an
    # exact overlap that depends on clock resolution.
    _COLLISION_SLACK = 10

    def test_unchanged_database_refetches_almost_nothing(self, db, mirror, monkeypatch):
        _seed(db, n=400)
        mirror.sync(db)

        seen = self._instrument(monkeypatch)
        mirror.sync(db)

        assert seen["delta_calls"] == 1
        assert seen["delta_rows"] <= self._COLLISION_SLACK, (
            f"unchanged database fetched {seen['delta_rows']} of 400 rows")

    def test_one_change_fetches_about_one_row(self, db, mirror, monkeypatch):
        _seed(db, n=400)
        mirror.sync(db)

        with db.get_session() as s:
            row = s.query(Match).order_by(Match.id.asc()).first()
            row.home_xg = 2.5

        seen = self._instrument(monkeypatch)
        mirror.sync(db)

        assert seen["delta_rows"] <= 1 + self._COLLISION_SLACK, (
            f"one changed row pulled {seen['delta_rows']} of 400")
        assert _mirror_tuples(mirror.sync(db)) == _db_truth(db)

    def test_delta_cost_does_not_grow_with_table_size(self, db, monkeypatch, tmp_path):
        """The whole point: sync cost tracks changes, not history size.

        Same three edits against a 100-row and a 900-row history. If the delta
        scaled with the table, the second number would be ~9x the first.

        Note the counter is installed once and reset between measurements — a
        monkeypatch applied inside a helper persists for the whole test, so
        patching per measurement would leave the second mirror's cold start
        wired to the first measurement's assertions.
        """
        seen = self._instrument(monkeypatch, forbid_full=False)

        def measure(n, tag):
            _seed(db, n=n)
            m = HistoryMirror(directory=tmp_path / tag, require_postgres=False)
            m.sync(db)                       # cold: full resync, not counted below
            with db.get_session() as s:
                for row in s.query(Match).order_by(Match.id.desc()).limit(3):
                    row.home_xg = 4.25
            seen["delta_rows"] = 0
            m.sync(db)                       # the measurement
            return seen["delta_rows"]

        small = measure(100, "scale_small")
        big = measure(800, "scale_big")      # cumulative: the DB now holds 900 rows

        with db.get_session() as s:
            total = s.query(Match).count()
        assert total >= 900, f"expected a ~900-row history, got {total}"

        assert big <= small + self._COLLISION_SLACK, (
            f"delta grew with history size: {small} rows at 100, {big} at {total}")
        assert big <= 3 + 2 * self._COLLISION_SLACK, (
            f"3 changes in a {total}-row history pulled {big} rows")

    def test_watermark_is_the_newest_row_received_not_now(self, db, mirror):
        """Using now() would silently skip rows committed mid-sync."""
        _seed(db)
        mirror.sync(db)
        meta = json.loads(mirror.meta_path.read_text())

        with db.get_session() as s:
            newest = s.query(Match.updated_at).order_by(
                Match.updated_at.desc()).first()[0]

        assert meta["watermark"] is not None
        assert datetime.fromisoformat(meta["watermark"]) == newest


class TestFailureModes:

    def test_missing_metadata_after_a_crash_is_recoverable(self, db, mirror):
        """Simulates dying between the Parquet write and the metadata write."""
        _seed(db)
        mirror.sync(db)
        mirror.meta_path.unlink()

        frame = mirror.sync(db)
        assert _mirror_tuples(frame) == _db_truth(db)

    def test_corrupt_parquet_is_rebuilt(self, db, mirror):
        _seed(db)
        mirror.sync(db)
        mirror.parquet_path.write_bytes(b"not a parquet file")

        frame = mirror.sync(db)
        assert _mirror_tuples(frame) == _db_truth(db)

    def test_schema_version_bump_forces_a_resync(self, db, mirror):
        _seed(db)
        mirror.sync(db)
        meta = json.loads(mirror.meta_path.read_text())
        meta["schema_version"] = SCHEMA_VERSION + 1
        mirror.meta_path.write_text(json.dumps(meta))

        frame = mirror.sync(db)
        assert _mirror_tuples(frame) == _db_truth(db)
        assert json.loads(mirror.meta_path.read_text())["schema_version"] == SCHEMA_VERSION

    def test_partial_write_never_leaves_a_torn_file(self, db, mirror):
        _seed(db)
        mirror.sync(db)
        assert not list(mirror.dir.glob("*.tmp")), "temp files left behind"

    def test_sqlite_is_skipped_by_policy(self, db, tmp_path):
        strict = HistoryMirror(directory=tmp_path / "strict", require_postgres=True)
        assert strict.supports(db) is False
        with pytest.raises(MirrorUnavailable):
            strict.sync(db)

    def test_missing_updated_at_column_is_unavailable(self, db, mirror, monkeypatch):
        """A rolled-back migration 001 must degrade, not raise."""
        monkeypatch.setattr(HistoryMirror, "supports", lambda self, d: False)
        with pytest.raises(MirrorUnavailable):
            mirror.sync(db)

    def test_invalidate_removes_the_files(self, db, mirror):
        _seed(db)
        mirror.sync(db)
        assert mirror.parquet_path.exists()
        mirror.invalidate()
        assert not mirror.parquet_path.exists()
        assert not mirror.meta_path.exists()


class TestModelsSeeIdenticalRowsThroughTheMirror:

    def test_match_history_via_mirror_equals_via_database(self, db, tmp_path, monkeypatch):
        """The property that matters: swapping in the mirror changes nothing."""
        from src.data import match_history

        _seed(db, n=60)

        monkeypatch.setenv("HISTORY_MIRROR_DISABLED", "1")
        match_history.invalidate()
        from_db = match_history.get_completed_matches(db)

        monkeypatch.delenv("HISTORY_MIRROR_DISABLED", raising=False)
        monkeypatch.setattr(
            "src.data.history_mirror.HistoryMirror.__init__",
            lambda self, directory=None, require_postgres=True: HistoryMirror.__init__(
                self, directory=tmp_path / "m2", require_postgres=False),
        )
        match_history.invalidate()
        from_mirror = match_history.get_completed_matches(db)

        assert len(from_mirror) == len(from_db)
        for a, b in zip(from_mirror, from_db):
            assert (a.id, a.match_date, a.home_team_id, a.away_team_id,
                    a.home_goals, a.away_goals, a.home_xg, a.away_xg, a.league) == \
                   (b.id, b.match_date, b.home_team_id, b.away_team_id,
                    b.home_goals, b.away_goals, b.home_xg, b.away_xg, b.league)
        match_history.invalidate()

    def test_elo_and_poisson_are_identical_through_the_mirror(self, db, tmp_path, monkeypatch):
        """Explicit model-level equality, not just row-level.

        Row equality already implies this, but the models are the thing that
        must not move — so assert on the ratings and strengths themselves.
        """
        from src.data import match_history
        from src.models.elo_system import EloRatingSystem
        from src.models.poisson_model import PoissonModel

        _seed(db, n=120)
        monkeypatch.setattr("src.models.elo_system.get_db", lambda: db)

        def fit_both():
            elo = EloRatingSystem()
            elo.fit()
            poisson = PoissonModel()
            monkeypatch.setattr(poisson, "db", db)
            poisson.fit()
            return elo.ratings, poisson._team_strengths, poisson.league_avg_home_goals

        monkeypatch.setenv("HISTORY_MIRROR_DISABLED", "1")
        match_history.invalidate()
        db_elo, db_poisson, db_avg = fit_both()

        monkeypatch.delenv("HISTORY_MIRROR_DISABLED", raising=False)
        monkeypatch.setattr(
            "src.data.history_mirror.HistoryMirror.__init__",
            lambda self, directory=None, require_postgres=True: HistoryMirror.__init__(
                self, directory=tmp_path / "m3", require_postgres=False),
        )
        match_history.invalidate()
        mirror_elo, mirror_poisson, mirror_avg = fit_both()
        match_history.invalidate()

        assert db_elo, "reference fit produced no Elo ratings"
        assert mirror_elo.keys() == db_elo.keys()
        for tid in db_elo:
            assert mirror_elo[tid] == pytest.approx(db_elo[tid], abs=1e-12)

        assert mirror_poisson.keys() == db_poisson.keys()
        for tid, vals in db_poisson.items():
            assert mirror_poisson[tid]["attack"] == pytest.approx(vals["attack"], abs=1e-12)
            assert mirror_poisson[tid]["defense"] == pytest.approx(vals["defense"], abs=1e-12)

        assert mirror_avg == pytest.approx(db_avg, abs=1e-12)
