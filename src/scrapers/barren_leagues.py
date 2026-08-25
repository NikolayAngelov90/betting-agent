"""Stage 15, L2 — stop paying for leagues the provider does not price.

THE WASTE. `refresh_imminent` selects a league because it has an imminent
fixture with a pending pick, spends 2 credits, and receives an empty event
list. The provider simply does not price that competition. Nothing is written,
no closing line becomes available, and the same league is selected again on the
next run, and the next. Measured across the cached CI logs: 17 such requests,
34 credits, **zero** observations — and 17 is a LOWER bound, for the reason in
`_last_league_outcomes` below.

WHY THIS IS THE ONLY LEVER IMPLEMENTED IN STAGE 15. It is the sole priced lever
whose credit cost is strictly negative and whose coverage cost is exactly zero:
a request that returns no rows cannot produce an observation, so declining to
make it cannot lose one. Every other lever measured buys coverage WITH credits
and is therefore an operator's decision about money, not a defect to fix.

WHY IT IS NOT A BLOCKLIST. The obvious implementation — a hard-coded set of
UEFA competition keys — is wrong twice over. It would have excluded
france/ligue-1, netherlands/eredivisie, spain/laliga and portugal/primeira-liga,
each of which returned no_rows exactly once in the same window and is plainly
priced. And a competition's coverage CHANGES: the provider does not price
Conference League qualifiers in July and does price the group stage in
September. A static list encodes a fact that expires and never learns that it
has.

So: a league must earn its exclusion (three consecutive empty fetches), the
exclusion EXPIRES (a probe is forced after the TTL), and one successful fetch
clears the record entirely. The failure mode this is designed against is not
"we skip a league we should have fetched" — it is "we skip it forever and never
find out we were wrong".
"""

from __future__ import annotations

import json
import logging
import pathlib
from datetime import timedelta
from typing import Dict

from src.utils.logger import utcnow

logger = logging.getLogger(__name__)

DEFAULT_PATH = pathlib.Path("data/odds_barren_leagues.json")

# Three, not one. france/ligue-1, netherlands/eredivisie, spain/laliga and
# portugal/primeira-liga each returned no_rows exactly once in the measured
# window; a threshold of 1 would have silently stopped pricing four of the
# largest leagues in the experiment.
BARREN_CONSECUTIVE_THRESHOLD = 3

# After this long an excluded league is probed again, at a cost of one request.
# The point is not to be cheap, it is to be self-correcting: 10 days bounds how
# long a wrong exclusion can persist, and costs at most 2 credits per league per
# 10 days to find out.
BARREN_TTL_DAYS = 10


class BarrenLeagueCache:
    """Remembers which leagues returned no odds, and forgets on a schedule."""

    def __init__(self, path: pathlib.Path | str = DEFAULT_PATH):
        self.path = pathlib.Path(path)
        self._state: Dict[str, dict] = self._load()

    def _load(self) -> Dict[str, dict]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            return {}
        return raw if isinstance(raw, dict) else {}

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._state, indent=2, sort_keys=True),
                                 encoding="utf-8")
        except OSError as exc:
            # A cache that cannot persist must not take the refresh down with
            # it: the consequence is spending credits we could have saved,
            # which is the status quo this replaces.
            logger.warning(f"barren-league cache: could not save: {exc}")

    def should_skip(self, league: str, *, now=None) -> bool:
        """True only for a league that has earned exclusion and not aged out."""
        rec = self._state.get(league)
        if not rec or rec.get("consecutive", 0) < BARREN_CONSECUTIVE_THRESHOLD:
            return False
        now = now or utcnow()
        try:
            from datetime import datetime
            since = datetime.fromisoformat(rec["excluded_at"])
        except (KeyError, ValueError):
            return False
        if since.tzinfo is None:
            since = since.replace(tzinfo=now.tzinfo)
        if now - since >= timedelta(days=BARREN_TTL_DAYS):
            # Aged out. Reset so the league must re-earn its exclusion from
            # scratch rather than being re-excluded by a single empty probe.
            self._state.pop(league, None)
            self.save()
            logger.info(
                f"ODDS_REFRESH barren cache EXPIRED for league={league} after "
                f"{BARREN_TTL_DAYS}d — probing again")
            return False
        return True

    def record(self, league: str, *, empty: bool, now=None) -> None:
        """One fetch outcome. A success clears the record outright."""
        now = now or utcnow()
        if not empty:
            if self._state.pop(league, None) is not None:
                logger.info(f"ODDS_REFRESH barren cache CLEARED for league={league}")
            return
        rec = self._state.setdefault(league, {"consecutive": 0})
        rec["consecutive"] = rec.get("consecutive", 0) + 1
        rec["last_empty_at"] = now.isoformat()
        if rec["consecutive"] >= BARREN_CONSECUTIVE_THRESHOLD and "excluded_at" not in rec:
            rec["excluded_at"] = now.isoformat()
            logger.info(
                f"ODDS_REFRESH barren cache EXCLUDING league={league} after "
                f"{rec['consecutive']} consecutive empty fetches")

    def describe(self) -> str:
        excluded = [l for l, r in self._state.items()
                    if r.get("consecutive", 0) >= BARREN_CONSECUTIVE_THRESHOLD]
        return (f"barren cache: {len(excluded)} league(s) excluded, "
                f"{len(self._state)} tracked")
