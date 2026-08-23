"""Local Parquet mirror of completed-match history, kept fresh incrementally.

    PostgreSQL  →  incremental sync  →  local Parquet  →  Arrow/pandas  →  models

Why
---
``Elo.fit()`` and ``PoissonModel.fit()`` need the whole completed-match history.
After the column-projection work that is a 3.8 MB read — but it happens once per
*process*, and the daily GitHub Actions job runs seven of them, so the same
rows crossed the network ~5 times a day for data that barely changes.

The seven CLI steps share one runner filesystem, and ``data/models/`` is already
persisted between runs by ``actions/cache``. So a Parquet file there is shared
by every step of a run *and* survives to the next day. A sync then costs one
watermark query plus only the rows that actually changed — typically a few
hundred, not 38,000.

Correctness
-----------
The mirror is only ever as good as its change detection, so it is built around
``matches.updated_at`` (migration 001: column + BEFORE UPDATE trigger). Rules:

* **Inserts and updates** both move ``updated_at``, and the trigger fires for
  every writer — ORM, bulk ``query().update()``, raw SQL, psql.
* **The watermark is the newest ``updated_at`` actually received**, never
  ``now()``. A row committed while the sync was running therefore cannot be
  skipped; it simply arrives next time.
* **The next sync re-asks with ``>=``**, so a row sharing the watermark's exact
  timestamp is re-fetched rather than missed. Re-fetching is free of
  consequence because rows are merged by primary key.
* **Membership is re-evaluated on every synced row.** A fixture that gains a
  result enters the mirror; a row whose result is cleared leaves it. This is why
  the delta query does *not* filter on ``is_fixture``/``home_goals``.
* **Deletes are caught by a row-count reconcile.** ``updated_at`` cannot record
  a deletion, so each sync compares the mirror's row count against the
  database's and does a full resync on any mismatch. That costs one aggregate.
* **Interrupted syncs are safe.** The Parquet file is written to a temp path and
  atomically renamed; the metadata file is written only after. A crash in
  between leaves an older watermark, so the next run re-fetches a small overlap
  and merges it idempotently.

Anything missing — no ``updated_at`` column, no pyarrow, SQLite, a corrupt file
— disables the mirror and the caller falls back to reading from the database.
Degrading costs egress; it never costs correctness.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from sqlalchemy import func, inspect

from src.data.models import Match
from src.utils.logger import get_logger

logger = get_logger()

# Bump when the column set changes — a mismatch forces a full resync rather than
# silently serving a frame that is missing a column the models now read.
SCHEMA_VERSION = 1


def filter_generation() -> str:
    """Fingerprint of the rows this mirror is allowed to contain.

    Stage 13 (s5.3). The mirror is a cache of "completed matches", and what
    counts as one is now narrower: contaminated matches are excluded. A Parquet
    built before that change contains rows the current code must never see —
    and it does not announce itself, because a cache that answers is
    indistinguishable from a cache that answers correctly.

    A one-time invalidation would not have been enough. The daily workflow has
    `Restore database and models` / `Save database and models` steps, so a stale
    mirror can be restored into a run whose code has the filter and be trusted
    on arrival. Local deletion would never touch it.

    Deriving the marker from the SOURCE of the predicate makes invalidation
    automatic: change the filter, and every existing mirror stops matching
    without anyone remembering to do anything. A comment-only edit also
    invalidates, which over-rebuilds — one database read, the safe direction.

    BOUNDARY — what this does NOT cover.

    The digest covers `_base_filter`'s own text and nothing else. If the
    predicate ever referenced a module-level constant, an enum of exclusion
    reasons, or a helper function, changing THAT would alter the filter's
    behaviour without moving this digest, and a stale mirror would be served as
    valid. That is exactly the shape found inside `_base_filter` itself during
    Stage 13: a guard that looks complete because nobody asked what it excludes.

    Today the predicate is closed — it references only `Match` — and
    `test_mirror_generation_stamp.py` pins that with an AST check rather than
    leaving it as an assumption. If that test ever fails, this function must be
    extended to digest the symbols the predicate closes over, not merely
    relaxed.
    """
    import hashlib
    import inspect

    from src.data.match_history import _HistoryCache

    try:
        src = inspect.getsource(_HistoryCache._base_filter)
    except (OSError, TypeError):  # pragma: no cover — source unavailable
        return "unknown"
    return hashlib.blake2s(src.encode("utf-8"), digest_size=6).hexdigest()



# Must stay in step with src/data/match_history.py's projection.
COLUMNS = (
    "id", "match_date", "home_team_id", "away_team_id",
    "home_goals", "away_goals", "home_xg", "away_xg", "league",
)

_DEFAULT_DIR = Path("data/models")
_PARQUET_NAME = "match_history.parquet"
_META_NAME = "match_history_meta.json"


class MirrorUnavailable(Exception):
    """The mirror cannot be used; the caller should read from the database."""


class _SyncLock:
    """Cross-process exclusive lock around a whole sync.

    Atomic renames make each file write crash-safe, but they do not make the
    *pair* of writes atomic with respect to another process:

        A: writes parquet_A (watermark_A)
        B: writes parquet_B (watermark_B, older)
        A: writes meta_A   (watermark_A)
        -> meta_A + parquet_B

    The next sync then resumes from watermark_A and never re-fetches the rows
    that existed only in parquet_A. The row-count reconcile catches that only if
    those rows were inserts; if they were *updates* — a settled result, an xG
    backfill — the count still matches and the stale values persist silently.

    Serialising the sync removes the interleaving entirely, and as a bonus stops
    N workers each doing a full resync against a cold cache. Held only for the
    duration of a sync, which is a delta fetch plus a file write.

    Best-effort by design: if locking is unavailable the sync still runs, which
    is exactly the behaviour that existed before.
    """

    def __init__(self, path: Path, timeout: float = 120.0):
        self.path = path
        self.timeout = timeout
        self._handle = None

    def __enter__(self):
        import time
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self._handle = open(self.path, "a+b")
                self._acquire(self._handle)
                return self
            except MirrorUnavailable:
                raise
            except Exception:
                if self._handle is not None:
                    self._handle.close()
                    self._handle = None
                if time.monotonic() >= deadline:
                    logger.warning(
                        f"history mirror lock busy for {self.timeout:.0f}s — "
                        f"syncing without it"
                    )
                    return self
                time.sleep(0.2)

    def __exit__(self, *exc):
        if self._handle is not None:
            try:
                self._release(self._handle)
            except Exception:
                pass
            self._handle.close()
            self._handle = None
        return False

    @staticmethod
    def _acquire(handle):
        try:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except ImportError:
            pass
        import msvcrt
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)

    @staticmethod
    def _release(handle):
        try:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            return
        except ImportError:
            pass
        import msvcrt
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


class HistoryMirror:
    """Parquet-backed mirror of the completed-match history."""

    def __init__(self, directory: Optional[Path] = None,
                 require_postgres: bool = True):
        """
        Args:
            directory: Where the mirror lives. Defaults to ``data/models``,
                which GitHub Actions already persists via ``actions/cache`` —
                that is what makes the mirror survive between workflow steps
                and between days.
            require_postgres: Skip SQLite. This is a *policy*, not a
                correctness requirement: mirroring a local SQLite file to
                another local file saves no egress. Correctness needs only a
                maintained ``updated_at``, which SQLAlchemy's onupdate provides
                on SQLite — so the tests turn this off to exercise the sync.
        """
        self.dir = Path(directory or os.environ.get("HISTORY_MIRROR_DIR", _DEFAULT_DIR))
        self.parquet_path = self.dir / _PARQUET_NAME
        self.meta_path = self.dir / _META_NAME
        self.require_postgres = require_postgres

    # ------------------------------------------------------------- capability

    @staticmethod
    def _pandas():
        try:
            import pandas as pd
            return pd
        except Exception as e:  # pragma: no cover - dependency shape
            raise MirrorUnavailable(f"pandas unavailable: {e}")

    @staticmethod
    def _require_parquet_engine():
        try:
            import pyarrow  # noqa: F401
        except Exception as e:
            raise MirrorUnavailable(f"pyarrow unavailable: {e}")

    def supports(self, db) -> bool:
        """True when this database can drive an incremental sync.

        Probes for ``matches.updated_at`` rather than assuming migration 001 is
        applied, so a rolled-back schema degrades to a database read instead of
        raising.
        """
        try:
            if self.require_postgres and not db.is_postgres:
                return False
            cols = {c["name"] for c in inspect(db.engine).get_columns("matches")}
            return "updated_at" in cols
        except Exception as e:
            logger.debug(f"history mirror capability probe failed: {e}")
            return False

    # ------------------------------------------------------------------ state

    def _read_meta(self) -> dict:
        try:
            with open(self.meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            if meta.get("schema_version") != SCHEMA_VERSION:
                logger.info(
                    f"history mirror schema {meta.get('schema_version')} != "
                    f"{SCHEMA_VERSION} — full resync"
                )
                return {}
            # Stage 13 (s5.3): refuse, do not serve. A mirror built under a
            # different exclusion predicate holds rows this code must not see.
            _want = filter_generation()
            if meta.get("filter_generation") != _want:
                logger.warning(
                    "history mirror was built under exclusion filter "
                    f"{meta.get('filter_generation')!r}, current is {_want!r} "
                    "— discarding it and reading from the database"
                )
                return {}
            return meta
        except FileNotFoundError:
            return {}
        except Exception as e:
            logger.warning(f"history mirror metadata unreadable ({e}) — full resync")
            return {}

    def _write_atomic(self, frame, watermark) -> None:
        """Write Parquet then metadata, both atomically and in that order.

        Ordering matters: metadata written first would claim a watermark for
        rows that are not in the file yet, and a crash between the two would
        make the mirror permanently miss them. This way a crash only costs a
        re-fetch of the overlap.
        """
        self.dir.mkdir(parents=True, exist_ok=True)

        tmp_parquet = self.parquet_path.with_suffix(".parquet.tmp")
        frame.to_parquet(tmp_parquet, index=False)
        os.replace(tmp_parquet, self.parquet_path)

        tmp_meta = self.meta_path.with_suffix(".json.tmp")
        with open(tmp_meta, "w", encoding="utf-8") as fh:
            json.dump({
                "schema_version": SCHEMA_VERSION,
                "filter_generation": filter_generation(),
                "watermark": watermark.isoformat() if watermark is not None else None,
                "row_count": int(len(frame)),
            }, fh)
        os.replace(tmp_meta, self.meta_path)

    def invalidate(self) -> None:
        """Drop the mirror from disk (next sync rebuilds it)."""
        for path in (self.parquet_path, self.meta_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.debug(f"could not remove {path}: {e}")

    # ------------------------------------------------------------------- sync

    @staticmethod
    def _completed_count(db) -> int:
        with db.get_session() as session:
            return session.query(func.count(Match.id)).filter(
                Match.is_fixture == False,  # noqa: E712
                Match.home_goals.isnot(None),
                Match.training_exclusion_reason.is_(None),  # s5.3 — learns/measures
            ).scalar() or 0

    @staticmethod
    def _fetch_full(db):
        with db.get_session() as session:
            return session.query(
                Match.id, Match.match_date,
                Match.home_team_id, Match.away_team_id,
                Match.home_goals, Match.away_goals,
                Match.home_xg, Match.away_xg, Match.league,
                Match.updated_at,
            ).filter(
                Match.is_fixture == False,  # noqa: E712
                Match.home_goals.isnot(None),
                Match.training_exclusion_reason.is_(None),  # s5.3 — learns/measures
            ).order_by(Match.match_date.asc(), Match.id.asc()).all()

    @staticmethod
    def _fetch_delta(db, since):
        """Rows changed at or after ``since`` — completed or not.

        No is_fixture / home_goals filter: membership of the completed set is
        decided per row after the fetch, so that a fixture which just gained a
        result is added and a row whose result was cleared is removed.
        """
        with db.get_session() as session:
            return session.query(
                Match.id, Match.match_date,
                Match.home_team_id, Match.away_team_id,
                Match.home_goals, Match.away_goals,
                Match.home_xg, Match.away_xg, Match.league,
                Match.updated_at, Match.is_fixture,
            ).filter(
                Match.updated_at >= since
            ).order_by(Match.updated_at.asc()).all()

    def sync(self, db):
        """Bring the mirror up to date and return it as a pandas DataFrame.

        Raises MirrorUnavailable if the mirror cannot be used, in which case the
        caller reads from the database as before.
        """
        pd = self._pandas()
        self._require_parquet_engine()
        if not self.supports(db):
            raise MirrorUnavailable("postgres + matches.updated_at required")

        # Read, merge and write under one lock — see _SyncLock for why the two
        # atomic renames are not enough on their own.
        with _SyncLock(self.dir / (_PARQUET_NAME + ".lock")):
            return self._sync_locked(db, pd)

    def _sync_locked(self, db, pd):
        meta = self._read_meta()
        watermark = meta.get("watermark")
        frame = None

        if watermark:
            try:
                frame = pd.read_parquet(self.parquet_path)
            except Exception as e:
                logger.warning(f"history mirror unreadable ({e}) — full resync")
                frame = None
                watermark = None

        if frame is None or watermark is None:
            return self._full_resync(db, pd, reason="no usable local mirror")

        since = pd.Timestamp(watermark).to_pydatetime()
        rows = self._fetch_delta(db, since)

        if rows:
            frame, new_watermark = self._merge(pd, frame, rows)
        else:
            new_watermark = since

        # Deletes leave no trace in updated_at, so reconcile on count. One
        # aggregate per sync; a mismatch is rare enough that a full resync is
        # the right response.
        db_count = self._completed_count(db)
        if len(frame) != db_count:
            logger.info(
                f"history mirror count drift: local={len(frame)} db={db_count} "
                f"— full resync"
            )
            return self._full_resync(db, pd, reason="row-count drift")

        frame = frame.sort_values(["match_date", "id"], kind="mergesort")
        self._write_atomic(frame, new_watermark)
        logger.debug(
            f"history mirror synced: {len(rows)} changed rows, "
            f"{len(frame):,} total, watermark {new_watermark}"
        )
        return frame

    def _merge(self, pd, frame, rows):
        """Apply changed rows: upsert the completed ones, drop the rest."""
        new_watermark = max(r.updated_at for r in rows)

        completed, removed = [], []
        for r in rows:
            if (not r.is_fixture) and r.home_goals is not None:
                completed.append({
                    "id": r.id, "match_date": r.match_date,
                    "home_team_id": r.home_team_id, "away_team_id": r.away_team_id,
                    "home_goals": r.home_goals, "away_goals": r.away_goals,
                    "home_xg": r.home_xg, "away_xg": r.away_xg, "league": r.league,
                })
            else:
                removed.append(r.id)

        touched = {row["id"] for row in completed} | set(removed)
        if touched:
            frame = frame[~frame["id"].isin(touched)]
        if completed:
            frame = pd.concat([frame, pd.DataFrame(completed, columns=list(COLUMNS))],
                              ignore_index=True)
        return frame, new_watermark

    def _full_resync(self, db, pd, reason: str):
        logger.info(f"history mirror: full resync ({reason})")
        rows = self._fetch_full(db)
        frame = pd.DataFrame(
            [
                (r.id, r.match_date, r.home_team_id, r.away_team_id,
                 r.home_goals, r.away_goals, r.home_xg, r.away_xg, r.league)
                for r in rows
            ],
            columns=list(COLUMNS),
        )
        watermark = max((r.updated_at for r in rows), default=None)
        self._write_atomic(frame, watermark)
        logger.info(f"history mirror rebuilt: {len(frame):,} completed matches")
        return frame
