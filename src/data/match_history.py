"""Process-wide cache of completed-match history — the project's egress choke point.

Why this module exists
----------------------
``Elo.fit()`` and ``PoissonModel.fit()`` both read the same slice of ``matches``
(completed games with a result). Before this module they each issued
``session.query(Match)`` — SQLAlchemy renders that as ``SELECT matches.*``, i.e.
all 45 columns at ~329 bytes/row on the wire, for 38k+ rows = ~12 MB per call.
Elo had no LIMIT at all.

Two things made that catastrophic for a Supabase free-plan egress quota:

1. The models consume 5 (Elo) and 8 (Poisson) of those 45 columns.
2. ``predictor.fit()`` runs ~10x per CI day — the daily workflow is 7 separate
   Python processes, and ``learn_from_settled`` alone fits three times (once
   fresh, once leak-resistant at ``as_of_date``, once to restore the full fit).

So: fetch the 9-column superset **once per process**, then serve every fit from
memory, filtering by league / as_of_date / limit in Python. Identical rows in,
identical models out — ~12 MB/call becomes ~3.6 MB once.

Freshness
---------
The cache must not go stale *within* a process: ``daily_update`` deliberately
re-fits after a backfill has inserted new matches, and settlement fills in
``home_goals`` on rows that were previously fixtures. Both change the
(count, max_id, max_date) of the completed-match set, so a tiny aggregate probe
(~50 bytes of egress) is enough to detect them and refetch. Writers that only
touch already-completed rows (e.g. xG backfill) run in their own process, which
starts cold anyway.
"""

from __future__ import annotations

import threading
from typing import Optional

from src.data.models import Match
from src.utils.logger import get_logger

logger = get_logger()

# The superset of columns Elo + Poisson actually read. Keep this list minimal:
# every column added here is paid for on every process start, forever.
#   Elo:     home_team_id, away_team_id, home_goals, away_goals, match_date
#   Poisson: the above + league, home_xg, away_xg
#   id:      not used by the models — carried only to make ordering deterministic
_CORE_COLUMNS = (
    Match.id,
    Match.match_date,
    Match.home_team_id,
    Match.away_team_id,
    Match.home_goals,
    Match.away_goals,
    Match.home_xg,
    Match.away_xg,
    Match.league,
)
_CORE_FIELDS = (
    "id", "match_date", "home_team_id", "away_team_id",
    "home_goals", "away_goals", "home_xg", "away_xg", "league",
)


class MatchRow:
    """A completed match, 9 columns, attribute access.

    ``__slots__`` keeps 38k of these cheap, and attribute access means the model
    code that used to hold ORM ``Match`` instances (``m.home_goals``) needs no
    changes at all.
    """

    __slots__ = _CORE_FIELDS

    def __init__(self, row):
        (self.id, self.match_date, self.home_team_id, self.away_team_id,
         self.home_goals, self.away_goals, self.home_xg, self.away_xg,
         self.league) = row


class _HistoryCache:
    def __init__(self):
        self._rows: Optional[list] = None
        self._signature = None
        self._lock = threading.Lock()

    @staticmethod
    def _base_filter():
        return (Match.is_fixture == False, Match.home_goals.isnot(None))  # noqa: E712

    def _fetch_signature(self, db):
        """Cheap change-detector: (row count, max id, max match_date).

        One aggregate row (~50 bytes of egress) versus ~3.6 MB for a refetch.
        Catches new inserts *and* fixtures that just got a result.
        """
        from sqlalchemy import func
        with db.get_session() as session:
            return session.query(
                func.count(Match.id), func.max(Match.id), func.max(Match.match_date)
            ).filter(*self._base_filter()).one()

    def get(self, db) -> list:
        """All completed matches, oldest first, as lightweight ``MatchRow``s.

        Three tiers, cheapest first:
          1. this process's in-memory list, if the freshness probe still agrees;
          2. the on-disk Parquet mirror, brought up to date with an incremental
             sync (only rows whose ``updated_at`` moved);
          3. a full projected read from the database.

        Tier 2 needs PostgreSQL, ``matches.updated_at`` (migration 001) and
        pyarrow. Missing any of them simply skips to tier 3 — that costs egress,
        never correctness.
        """
        with self._lock:
            try:
                signature = self._fetch_signature(db)
            except Exception as e:
                # A probe failure must never break fitting: fall through to a
                # refetch (correct, just not free).
                logger.debug(f"match-history signature probe failed: {e}")
                signature = None

            if self._rows is not None and signature is not None and signature == self._signature:
                return self._rows

            rows = self._load_from_mirror(db)
            if rows is None:
                rows = self._load_from_db(db)

            self._rows = rows
            self._signature = signature
            return self._rows

    def _load_from_mirror(self, db) -> Optional[list]:
        if _mirror_disabled():
            return None
        try:
            from src.data.history_mirror import HistoryMirror, MirrorUnavailable
        except Exception as e:  # pragma: no cover - import shape
            logger.debug(f"history mirror import failed: {e}")
            return None
        try:
            frame = HistoryMirror().sync(db)
        except MirrorUnavailable as e:
            logger.debug(f"history mirror unavailable ({e}) — reading from database")
            return None
        except Exception as e:
            # Never let a mirror problem take the models down.
            logger.warning(f"history mirror sync failed ({e}) — reading from database")
            return None

        rows = [
            MatchRow((
                _int_or_none(r[0]), _to_datetime(r[1]),
                _int_or_none(r[2]), _int_or_none(r[3]),
                _int_or_none(r[4]), _int_or_none(r[5]),
                _float_or_none(r[6]), _float_or_none(r[7]),
                r[8] if r[8] is not None and r[8] == r[8] else None,
            ))
            for r in frame[list(_CORE_FIELDS)].itertuples(index=False, name=None)
        ]
        logger.debug(f"match-history loaded from mirror: {len(rows):,} matches")
        return rows

    def _load_from_db(self, db) -> list:
        with db.get_session() as session:
            rows = (
                session.query(*_CORE_COLUMNS)
                .filter(*self._base_filter())
                .order_by(Match.match_date.asc(), Match.id.asc())
                .all()
            )
        logger.debug(
            f"match-history loaded from database: {len(rows):,} completed matches "
            f"(9 of 46 columns)"
        )
        return [MatchRow(r) for r in rows]

    def invalidate(self):
        with self._lock:
            self._rows = None
            self._signature = None


def _mirror_disabled() -> bool:
    import os
    return os.environ.get("HISTORY_MIRROR_DISABLED", "").lower() in ("1", "true", "yes")


def _to_datetime(value):
    """pandas hands back Timestamp/NaT; model code expects datetime/None."""
    if value is None or value != value:  # NaT/NaN are not equal to themselves
        return None
    to_pydatetime = getattr(value, "to_pydatetime", None)
    return to_pydatetime() if to_pydatetime else value


def _int_or_none(value):
    """Undo pandas' float-widening of integer columns that contain NULLs."""
    if value is None or value != value:
        return None
    return int(value)


def _float_or_none(value):
    if value is None or value != value:
        return None
    return float(value)


_cache = _HistoryCache()


def get_completed_matches(db, league: str = None, as_of_date=None,
                          limit: int = None, newest_first: bool = False) -> list:
    """Completed matches, served from the per-process cache.

    Mirrors what ``session.query(Match).filter(is_fixture=False,
    home_goals.isnot(None))`` used to return, with the optional ``league`` /
    ``match_date < as_of_date`` filters and ``limit`` applied — but in Python,
    against rows already in memory.

    Args:
        league: Restrict to one league (``Match.league == league``).
        as_of_date: Keep only matches strictly before this date/datetime.
        limit: Keep at most this many rows *after* ordering.
        newest_first: Order newest-first (Poisson's ``order_by(desc).limit(n)``
            semantics). Default is oldest-first (Elo's chronological pass).

    Returns:
        A new list of ``MatchRow`` — callers may slice or reorder it freely
        without disturbing the cache.
    """
    cached = _cache.get(db)  # oldest first

    if newest_first:
        rows = cached[::-1]
    else:
        rows = list(cached)  # never hand out the cache's own list

    if league is not None:
        rows = [m for m in rows if m.league == league]

    if as_of_date is not None:
        cutoff = _coerce_cutoff(as_of_date)
        rows = [m for m in rows if m.match_date is not None and m.match_date < cutoff]

    if limit is not None and len(rows) > limit:
        rows = rows[:limit]

    return rows


def _coerce_cutoff(as_of_date):
    """Make ``as_of_date`` comparable with ``match_date``.

    ``match_date`` is a DateTime column, but callers pass ``date`` objects
    (``tune_ensemble_weights`` uses ``SavedPick.pick_date``). Postgres coerced
    that server-side; comparing in Python would raise TypeError, so do the same
    coercion here.
    """
    from datetime import date as _date, datetime as _datetime
    if isinstance(as_of_date, _datetime):
        return as_of_date
    if isinstance(as_of_date, _date):
        return _datetime(as_of_date.year, as_of_date.month, as_of_date.day)
    return as_of_date


def invalidate():
    """Drop the cached history (call after bulk-writing matches)."""
    _cache.invalidate()
