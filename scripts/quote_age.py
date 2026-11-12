"""How old is each stored quote? The census of `odds_snapshots` by freshness.

Stage 31 Part A. `observed_at` is OUR clock; `book_last_update` is the
bookmaker's. Their difference is the quote's AGE, and a generous price is only
worth something if it is not simply an old one.

    FRESH        the book changed the quote <= FRESH_MAX_AGE_S before we saw it
    STALE        older than that, or stamped after our own observation
    UNEVALUATED  no stamp came with the quote. NOT fresh. NOT stale.

Three populations are UNEVALUATED by construction, and the report names each
instead of folding them into a rate:

    * every row written before the column existed -- never backfilled;
    * every API-Football row -- that feed stamps the FIXTURE, not the book;
    * every row, on a database the column has not been migrated to yet. The
      column is nullable and `_migrate_missing_columns` adds it the next time
      an agent is constructed, so between the commit and the first run after
      it the honest reading is "the instrument is not installed", and this
      script says that rather than failing or reporting zero stale quotes.

Reads prices and timestamps only. No score, no pick, no result. Credentials
come from `.env` / `.env.local`, read inside `_load_env()`.

    python -m scripts.quote_age            # the census
    python -m scripts.quote_age --json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional

from src.data.price_history import (
    FRESH_MAX_AGE_S,
    QUOTE_FRESH,
    QUOTE_STALE,
    QUOTE_UNEVALUATED,
    quote_age_seconds,
    quote_freshness,
)

STATES = (QUOTE_FRESH, QUOTE_STALE, QUOTE_UNEVALUATED)
COLUMN = "book_last_update"


def _load_env() -> None:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in (".env", ".env.local"):
        try:
            with open(os.path.join(here, name), encoding="utf-8") as fh:
                for line in fh:
                    m = re.match(r"\s*([A-Z0-9_]+)\s*=\s*(.*)", line)
                    if m and not os.environ.get(m.group(1)):
                        os.environ[m.group(1)] = \
                            m.group(2).strip().strip('"').strip("'")
        except FileNotFoundError:
            continue


def feed_of(bookmaker: str) -> str:
    return "theodds" if (bookmaker or "").startswith("TheOddsAPI-") \
        else "apifootball"


def tally(rows) -> dict:
    """Pure. `rows` is `(bookmaker, observed_at, book_last_update)`.

    Returns counts by state overall, per feed and per book, plus the ages of
    the evaluated quotes. A rate is over EVALUATED quotes only and is None
    when there are none -- an empty denominator is not a zero.
    """
    overall = {s: 0 for s in STATES}
    by_feed: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {s: 0 for s in STATES})
    by_book: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {s: 0 for s in STATES})
    ages: Dict[str, List[float]] = defaultdict(list)
    n = 0
    for bookmaker, observed_at, last_update in rows:
        n += 1
        state = quote_freshness(observed_at, last_update)
        overall[state] += 1
        by_feed[feed_of(bookmaker)][state] += 1
        by_book[bookmaker][state] += 1
        age = quote_age_seconds(observed_at, last_update)
        if age is not None:
            ages[bookmaker].append(age)

    def _rate(c: Dict[str, int]) -> Optional[float]:
        evaluated = c[QUOTE_FRESH] + c[QUOTE_STALE]
        return None if evaluated == 0 else c[QUOTE_FRESH] / evaluated

    def _median(xs: List[float]) -> Optional[float]:
        if not xs:
            return None
        s = sorted(xs)
        m = len(s) // 2
        return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2

    return {
        "rows": n,
        "fresh_max_age_s": FRESH_MAX_AGE_S,
        "overall": overall,
        "fresh_rate_of_evaluated": _rate(overall),
        "by_feed": {f: dict(c, fresh_rate_of_evaluated=_rate(c))
                    for f, c in sorted(by_feed.items())},
        "by_book": {b: dict(c, fresh_rate_of_evaluated=_rate(c),
                            median_age_s=_median(ages.get(b, [])))
                    for b, c in sorted(by_book.items())},
    }


def census() -> dict:
    _load_env()
    from sqlalchemy import inspect, text

    from src.data.database import get_db
    db = get_db()
    assert db.is_postgres, "refusing to measure against the SQLite fallback"
    cols = {c["name"] for c in inspect(db.engine).get_columns("odds_snapshots")}
    with db.get_session() as s:
        if COLUMN not in cols:
            # THE INSTRUMENT IS NOT INSTALLED. Every row is unevaluated, and
            # that is reported as its own state, with the count.
            n, lo, hi = s.execute(text(
                "SELECT COUNT(*), MIN(observed_at), MAX(observed_at) "
                "FROM odds_snapshots")).one()
            return {
                "column_present": False, "rows": int(n),
                "observed_from": str(lo), "observed_until": str(hi),
                "overall": {QUOTE_FRESH: 0, QUOTE_STALE: 0,
                            QUOTE_UNEVALUATED: int(n)},
                "fresh_rate_of_evaluated": None,
                "why": (f"`odds_snapshots.{COLUMN}` does not exist in this "
                        f"database yet. It is added by the nullable-column "
                        f"migration the next time an agent is constructed."),
            }
        rows = s.execute(text(
            f"SELECT bookmaker, observed_at, {COLUMN} "
            f"FROM odds_snapshots")).fetchall()
    out = tally(rows)
    out["column_present"] = True
    return out


def _print(r: dict) -> None:
    print("=" * 78)
    print("QUOTE AGE  -  odds_snapshots, grain = one stored quote")
    print("=" * 78)
    o = r["overall"]
    print(f"  rows {r['rows']:,}   FRESH {o[QUOTE_FRESH]:,}   "
          f"STALE {o[QUOTE_STALE]:,}   UNEVALUATED {o[QUOTE_UNEVALUATED]:,}")
    rate = r["fresh_rate_of_evaluated"]
    print("  fresh share of EVALUATED quotes: "
          + ("UNEVALUATED (no quote carries a stamp)" if rate is None
             else f"{rate:.1%}"))
    if not r["column_present"]:
        print(f"  THE INSTRUMENT IS NOT INSTALLED: {r['why']}")
        print(f"  rows observed {r['observed_from']} .. {r['observed_until']}")
        return
    print(f"  fresh = age <= {r['fresh_max_age_s']:.0f}s")
    for feed, c in r["by_feed"].items():
        fr = c["fresh_rate_of_evaluated"]
        print(f"  {feed:12s} FRESH {c[QUOTE_FRESH]:>8,}  STALE "
              f"{c[QUOTE_STALE]:>8,}  UNEVALUATED {c[QUOTE_UNEVALUATED]:>8,}  "
              f"fresh/evaluated "
              + ("UNEVALUATED" if fr is None else f"{fr:.1%}"))
    for book, c in r["by_book"].items():
        if c[QUOTE_FRESH] + c[QUOTE_STALE] == 0:
            continue
        print(f"    {book:34s} fresh {c['fresh_rate_of_evaluated']:.1%}  "
              f"median age {c['median_age_s']:.0f}s  "
              f"n {c[QUOTE_FRESH] + c[QUOTE_STALE]:,}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = census()
    if a.json:
        print(json.dumps(r, indent=1, default=str))
    else:
        _print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
