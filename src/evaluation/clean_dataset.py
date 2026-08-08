"""The clean evaluation dataset — the only data future experiments may use.

Stage 4, Phase 3. Every model result produced before 2026-08-07 is potentially
contaminated: API-Football's two-way ``"Home/Away"`` bet was written into the
``1X2`` market type, overwriting genuine three-way prices on 13,274 rows across
2,548 matches and seven bookmakers (Bet365 92% of matches, William Hill 94%,
Unibet 96%, Betfair 86%, 10Bet 98%, Betano 96%, 888Sport 97%, Pinnacle 26%).

Rather than exclude whole bookmakers — Pinnacle is corrupt on only a quarter of
its matches, so dropping it would throw away three quarters of good data — the
filter works at the level of a single (match, bookmaker, market) book, and a
match qualifies on how many *surviving* books it has.

A record qualifies for the clean dataset when ALL of the following hold:

1. **Correct market mapping.** The book supplies every declared leg of the
   market (``market_spec.extract_legs``), so no leg is missing or borrowed.
2. **Valid odds.** Every leg is a decimal price > 1.0.
3. **Plausible overround.** ``sum(1/odds)`` lies inside the market's declared
   band. This is what excludes the corrupted books: the two-way-contaminated 1X2
   markets sit at a median of 1.3524, far outside [1.005, 1.25].
4. **Sufficient bookmaker coverage.** At least ``min_books`` surviving books
   price the market. One book is an opinion; the consensus needs a quorum.
5. **Valid outcome.** The match is complete with a recorded score.
6. **Known prediction timestamp.** The match has a kickoff time, so a forecast
   can be dated relative to it.
7. **No post-match information.** Odds rows are only admitted when their
   timestamp precedes kickoff. Rows with no timestamp are admitted only under
   ``allow_untimestamped`` (default True) because the odds table pre-dates
   reliable timestamping — this is recorded in the manifest rather than hidden,
   since it is the weakest link in the chain.

The result carries a **manifest** documenting exactly how many records were
rejected at each gate, so "clean" is an auditable claim rather than an adjective.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Sequence

from src.data.market_spec import (
    MARKET_SPECS,
    check_overround,
    devig,
    extract_legs,
)
from src.utils.logger import get_logger

logger = get_logger()

#: Bookmaker whose prices are display-only composites, never bettable quotes.
EXCLUDED_BOOKMAKERS = frozenset({"Flashscore"})

#: Default quorum. Chosen empirically in Phase 5, not by intuition — see
#: docs/stage4-clean-baseline-2026-08-07.md.
DEFAULT_MIN_BOOKS = 2


@dataclass
class CleanMatch:
    """One match that qualified, with its surviving books."""

    id: int
    match_date: date
    kickoff: Optional[datetime]
    home_team_id: int
    away_team_id: int
    home_goals: int
    away_goals: int
    league: str
    #: market_type -> list of per-book de-vigged probability vectors
    devigged: Dict[str, List[List[float]]] = field(default_factory=dict)
    #: market_type -> list of per-book raw price vectors (kept so any metric can
    #: be recomputed later without re-querying)
    raw_prices: Dict[str, List[List[float]]] = field(default_factory=dict)
    #: market_type -> number of surviving books
    n_books: Dict[str, int] = field(default_factory=dict)

    @property
    def outcome_1x2(self) -> int:
        if self.home_goals > self.away_goals:
            return 0
        return 1 if self.home_goals == self.away_goals else 2

    @property
    def total_goals(self) -> int:
        return self.home_goals + self.away_goals

    def consensus(self, market_type: str) -> Optional[List[float]]:
        """Per-outcome median of the surviving books, renormalised."""
        books = self.devigged.get(market_type)
        if not books:
            return None
        med = []
        for col in zip(*books):
            s = sorted(col)
            n = len(s)
            med.append(s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)
        total = sum(med)
        return [v / total for v in med] if total > 0 else None

    def best_price(self, market_type: str, leg_index: int) -> Optional[float]:
        """Median offered price for one leg — the price EV should be computed
        against, kept consistent with the probability source."""
        books = self.raw_prices.get(market_type)
        if not books:
            return None
        col = sorted(b[leg_index] for b in books)
        n = len(col)
        return col[n // 2] if n % 2 else (col[n // 2 - 1] + col[n // 2]) / 2


@dataclass
class Manifest:
    """Auditable record of what was rejected and why."""

    total_matches: int = 0
    qualified_matches: int = 0
    rejected: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    books_seen: int = 0
    books_rejected: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    per_market_matches: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    min_books: int = DEFAULT_MIN_BOOKS
    markets: Sequence[str] = ()
    allow_untimestamped: bool = True

    def render(self) -> str:
        lines = [
            "clean_evaluation_dataset manifest",
            f"  markets                : {', '.join(self.markets)}",
            f"  min surviving books    : {self.min_books}",
            f"  untimestamped odds     : "
            f"{'admitted' if self.allow_untimestamped else 'rejected'}",
            f"  matches considered     : {self.total_matches}",
            f"  matches qualified      : {self.qualified_matches}",
            "  match rejections:",
        ]
        for reason, n in sorted(self.rejected.items(), key=lambda kv: -kv[1]):
            lines.append(f"      {reason:<44} {n}")
        lines.append(f"  bookmaker-market books seen : {self.books_seen}")
        lines.append("  book rejections:")
        for reason, n in sorted(self.books_rejected.items(), key=lambda kv: -kv[1]):
            lines.append(f"      {reason:<44} {n}")
        lines.append("  qualified matches per market:")
        for mkt, n in sorted(self.per_market_matches.items()):
            lines.append(f"      {mkt:<44} {n}")
        return "\n".join(lines)


def _market_variants(market_type: str):
    """(label, line, side) tuples a market is evaluated at."""
    if market_type == "over_under":
        return [("over_under_2.5", "2.5", None), ("over_under_1.5", "1.5", None),
                ("over_under_3.5", "3.5", None)]
    if market_type == "team_goals":
        return [("team_goals_home_1.5", "1.5", "Home"),
                ("team_goals_away_1.5", "1.5", "Away")]
    return [(market_type, None, None)]


def build(matches: Sequence, odds_rows: Sequence,
          markets: Sequence[str] = ("1X2", "over_under", "btts"),
          min_books: int = DEFAULT_MIN_BOOKS,
          allow_untimestamped: bool = True) -> tuple:
    """Assemble the clean dataset.

    Args:
        matches: objects with id, match_date, home/away_team_id, home/away_goals,
            league. Completed matches only.
        odds_rows: objects with match_id, bookmaker, market_type, selection,
            odds_value and (optionally) timestamp.
        markets: which declared market types to evaluate.
        min_books: quorum of surviving books required per market.
        allow_untimestamped: admit odds rows with no timestamp.

    Returns:
        (list[CleanMatch], Manifest)
    """
    manifest = Manifest(min_books=min_books, markets=tuple(markets),
                        allow_untimestamped=allow_untimestamped)

    by_match = defaultdict(lambda: defaultdict(dict))   # match -> book -> {sel: odds}
    kickoff_by_match = {m.id: getattr(m, "match_date", None) for m in matches}
    for o in odds_rows:
        if o.bookmaker in EXCLUDED_BOOKMAKERS:
            manifest.books_rejected["excluded bookmaker (display-only prices)"] += 1
            continue
        ts = getattr(o, "timestamp", None)
        if ts is None:
            if not allow_untimestamped:
                manifest.books_rejected["odds row has no timestamp"] += 1
                continue
        else:
            ko = kickoff_by_match.get(o.match_id)
            if ko is not None and ts > ko:
                # Gate 7: an odds row written after kickoff cannot inform a
                # pre-match forecast.
                manifest.books_rejected["odds timestamped after kickoff"] += 1
                continue
        key = (o.bookmaker, o.market_type)
        by_match[o.match_id][key][o.selection] = o.odds_value

    out: List[CleanMatch] = []
    for m in matches:
        manifest.total_matches += 1
        if m.home_goals is None or m.away_goals is None:
            manifest.rejected["no valid outcome (incomplete match)"] += 1
            continue
        if getattr(m, "match_date", None) is None:
            manifest.rejected["no kickoff timestamp"] += 1
            continue

        raw_date = m.match_date
        md = raw_date.date() if hasattr(raw_date, "date") else raw_date

        cm = CleanMatch(
            id=m.id, match_date=md, kickoff=raw_date,
            home_team_id=m.home_team_id, away_team_id=m.away_team_id,
            home_goals=m.home_goals, away_goals=m.away_goals,
            league=getattr(m, "league", None) or "unknown",
        )

        books = by_match.get(m.id, {})
        for market_type in markets:
            if market_type not in MARKET_SPECS:
                continue
            for label, line, side in _market_variants(market_type):
                survivors_p, survivors_raw = [], []
                for (bookmaker, mt), sels in books.items():
                    if mt != market_type:
                        continue
                    manifest.books_seen += 1
                    prices = extract_legs(market_type, sels, line=line, side=side)
                    if prices is None:
                        manifest.books_rejected[f"{label}: incomplete legs"] += 1
                        continue
                    ok, why = check_overround(market_type, prices)
                    if not ok:
                        manifest.books_rejected[
                            f"{label}: implausible overround (corruption gate)"] += 1
                        continue
                    probs = devig(market_type, prices)
                    if probs is None:
                        manifest.books_rejected[f"{label}: de-vig refused"] += 1
                        continue
                    survivors_p.append(probs)
                    survivors_raw.append(prices)

                if len(survivors_p) >= min_books:
                    cm.devigged[label] = survivors_p
                    cm.raw_prices[label] = survivors_raw
                    cm.n_books[label] = len(survivors_p)
                    manifest.per_market_matches[label] += 1

        if not cm.devigged:
            manifest.rejected[
                f"no market reached the {min_books}-book quorum"] += 1
            continue
        manifest.qualified_matches += 1
        out.append(cm)

    out.sort(key=lambda c: c.match_date)
    logger.info(
        f"clean_evaluation_dataset: {manifest.qualified_matches}/"
        f"{manifest.total_matches} matches qualified (min_books={min_books})")
    return out, manifest


def load_from_db(since: date = date(2022, 1, 1), **kwargs) -> tuple:
    """Build the dataset straight from the database (read-only)."""
    from src.data.database import get_db
    from src.data.models import Match, Odds

    db = get_db()
    with db.get_session() as session:
        matches = session.query(
            Match.id, Match.match_date, Match.home_team_id, Match.away_team_id,
            Match.home_goals, Match.away_goals, Match.league,
        ).filter(
            Match.is_fixture == False,  # noqa: E712
            Match.home_goals.isnot(None),
            Match.away_goals.isnot(None),
            Match.match_date >= since,
        ).all()
        match_ids = {m.id for m in matches}
        odds = session.query(
            Odds.match_id, Odds.bookmaker, Odds.market_type,
            Odds.selection, Odds.odds_value, Odds.timestamp,
        ).filter(Odds.match_id.in_(match_ids)).all() if match_ids else []
    return build(matches, odds, **kwargs)
