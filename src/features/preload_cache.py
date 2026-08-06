"""Completeness-checked accessors for the feature preload cache.

Why this exists
---------------
``FeatureEngineer.preload_batch`` bulk-loads match history so per-fixture
feature building doesn't issue O(N) queries. Every consumer used to reach into
``cache["team_history"][team_id]`` directly and treat "not there" or "fewer rows
than I asked for" as **"this team has no history"** — returning zeroed features
instead of asking the database.

That silently corrupted two feature families:

* **League standings** — ``_get_league_standings`` ranks *every* team in the
  division, but the cache only ever held the two teams playing the fixture. The
  other ~18 clubs came back with 0 points, so the fixture's teams always
  finished 1st and 2nd and ``league_position`` was meaningless in the picks path
  and in ML training.
* **Referee stats** — the cached branch scanned only those two teams' matches
  for the official (and skipped the 365-day window the live query applies), so
  the averages were computed from a handful of matches instead of the
  referee's last 30.

The fix is to make "can the cache answer this exactly?" a question with a
provable answer, asked in one place instead of seven.

The completeness rule
---------------------
Cached row lists are **prefixes of the true history in descending date order**
(that is how ``preload_batch`` builds them: order by ``match_date DESC``, then
truncate by count and/or date). Filtering a descending prefix by any row
predicate yields a descending prefix of the filtered history. Therefore a
cached answer is *exactly* what ``... ORDER BY match_date DESC LIMIT n`` would
return whenever either:

1. at least ``n`` cached rows match the predicate — every row the database
   would return is at least as recent as our n-th match, so it is in our
   prefix; or
2. the cached list is not truncated at all — it *is* the full history.

Anything else is a genuine cache miss, and the accessor returns ``None`` so the
caller runs its normal live query. That makes preload/live equivalence
structural rather than something to remember.
"""

from __future__ import annotations

from typing import Callable, Optional

# Cache dict keys. Kept as plain strings so the cache stays a JSON-ish dict and
# existing `cache.get("team_history", {})` reads keep working.
TEAM_HISTORY = "team_history"
TEAM_COMPLETE = "team_complete"
LEAGUE_HISTORY = "league_history"
LEAGUE_COMPLETE = "league_complete"
REFEREE_HISTORY = "referee_history"
REFEREE_COMPLETE = "referee_complete"
H2H_HISTORY = "h2h_history"
H2H_COMPLETE = "h2h_complete"


def _resolve(rows: Optional[list], is_complete: bool,
             limit: Optional[int],
             predicate: Optional[Callable] = None) -> Optional[list]:
    """Apply the completeness rule. ``None`` means "ask the database"."""
    if rows is None:
        return None
    matched = [r for r in rows if predicate(r)] if predicate else list(rows)
    if limit is None:
        return matched if is_complete else None
    if len(matched) >= limit:
        return matched[:limit]
    return matched if is_complete else None


def team_rows(cache: Optional[dict], team_id: int, limit: Optional[int] = None,
              predicate: Optional[Callable] = None) -> Optional[list]:
    """Rows for one team from the per-team scope, or ``None`` on a miss."""
    if not cache:
        return None
    rows = cache.get(TEAM_HISTORY, {}).get(team_id)
    complete = team_id in cache.get(TEAM_COMPLETE, set())
    return _resolve(rows, complete, limit, predicate)


def league_team_rows(cache: Optional[dict], league: str, team_id: int,
                     limit: Optional[int] = None,
                     predicate: Optional[Callable] = None) -> Optional[list]:
    """Rows for one team *within one league*, from the league-wide scope.

    Backs ``_get_league_standings``, which needs every club in the division —
    not just the two in the fixture.
    """
    if not cache:
        return None
    by_team = cache.get(LEAGUE_HISTORY, {}).get(league)
    if by_team is None:
        return None
    complete = league in cache.get(LEAGUE_COMPLETE, set())
    # A team with no rows in a league we loaded completely genuinely has none.
    rows = by_team.get(team_id, [] if complete else None)
    return _resolve(rows, complete, limit, predicate)


def league_team_ids(cache: Optional[dict], league: str) -> Optional[list]:
    """Team ids known to play in ``league``, or ``None`` if not preloaded."""
    if not cache:
        return None
    if league not in cache.get(LEAGUE_COMPLETE, set()):
        return None
    by_team = cache.get(LEAGUE_HISTORY, {}).get(league)
    return None if by_team is None else list(by_team.keys())


def referee_rows(cache: Optional[dict], referee: str, limit: Optional[int] = None,
                 predicate: Optional[Callable] = None) -> Optional[list]:
    """Rows officiated by ``referee``, or ``None`` on a miss."""
    if not cache:
        return None
    rows = cache.get(REFEREE_HISTORY, {}).get(referee)
    complete = referee in cache.get(REFEREE_COMPLETE, set())
    return _resolve(rows, complete, limit, predicate)


def h2h_key(team_a: int, team_b: int) -> tuple:
    """Order-independent key for a head-to-head pairing."""
    return (team_a, team_b) if team_a <= team_b else (team_b, team_a)


def h2h_rows(cache: Optional[dict], home_team_id: int, away_team_id: int,
             limit: Optional[int] = None,
             predicate: Optional[Callable] = None) -> Optional[list]:
    """Past meetings between two teams, or ``None`` on a miss."""
    if not cache:
        return None
    key = h2h_key(home_team_id, away_team_id)
    rows = cache.get(H2H_HISTORY, {}).get(key)
    complete = key in cache.get(H2H_COMPLETE, set())
    return _resolve(rows, complete, limit, predicate)
