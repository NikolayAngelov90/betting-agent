"""Mark the matches and picks hit by the 1X2 outcome-assignment defect.

    python -m scripts.mark_1x2_assignment_corruption            # dry run
    python -m scripts.mark_1x2_assignment_corruption --apply

THE DEFECT (fixed 2026-10-10, `abb053e`). `_save_game_odds` classified each
Odds-API `h2h` outcome with a chain testing the HOME name first, using the
fuzzy `_team_names_similar`. `_tok_match` accepts substring containment, so
`"ce"` matches `"ceuta"` and a one-token name scores 1/1 — the AWAY outcome
satisfied the home branch. `odds` is UNIQUE on
(match_id, bookmaker, market_type, selection), so the second write OVERWROTE
the first and the stored Home price became the away team's price, with no Away
row written at all.

MARK, NEVER DELETE — fourth instance of that rule in this project. Nothing here
removes a row, a price or a pick. Two orthogonal fields are stamped and the
`odds` table is left exactly as it is, because it is the evidence.

WHY THE SET IS DERIVED, NOT LISTED. A hardcoded id list would be a snapshot of
one query at one moment and would silently diverge the moment anything changed.
The population is defined by the defect's own SIGNATURE and re-derived on every
run, so the script is idempotent and auditable:

    within TheOddsAPI bookmakers, a 1X2 (match, bookmaker) key with
    Home present and Away ABSENT

THE DIRECTIONALITY IS THE PROOF, and it is why this signature is not merely
"incomplete markets". Measured 2026-10-10 in production:

    TheOddsAPI    486 keys missing AWAY,  0 missing HOME   (28 matches)
    API-Football   49 keys missing AWAY, 49 missing HOME   (49 matches)

An asymmetric absence rules out a symmetric cause. A feed gap, a naming
mismatch and a coverage hole are all symmetric — the API-Football population
above is the control, drawn from the same table by the same query with a
different writer, and it is symmetric to the row. Only the ordered test chain
produces 486/0. **The API-Football rows are NOT marked by this script**: they
are a different population with a different (and still open) cause.

CONFIRMED vs PRESUMED, because `odds_snapshots` only began on 2026-08-25.
Where snapshot coverage exists the defect leaves a second signature — two
`Home` rows for one (match, bookmaker) inside a single second, because
`_upsert_odds` calls `record_price` on every invocation. Outside coverage that
check returns zero, and **a zero from absent instrumentation is not a clean
bill**. Both groups are marked; the split is reported.

CACHE INVALIDATION, stated because it is NOT automatic. `filter_generation()`
digests `_base_filter`'s SOURCE TEXT, so marking DATA does not move it. The
Parquet mirror still self-corrects: it reconciles row COUNTS against the
database and does a full resync on mismatch, and excluding rows changes that
count. The ML pickles do NOT self-invalidate on a data change and rely on
`_ml_models_stale()`'s 3-day rule, so they carry the excluded matches until the
next retrain. Run `--train` to close that window immediately.
"""

from __future__ import annotations

import argparse
import sys

#: `training_exclusion_reason`. Distinct from `corrupt_team_identity`: that one
#: says a participant row belongs to a different club, this one says the market
#: rows describe the right clubs with two of the prices on the wrong
#: selections. 48-char column, and this is 24.
EXCLUSION_REASON = "corrupt_1x2_assignment"

#: `evidence_status`. The EXISTING value, chosen rather than invented, and its
#: documented meaning is this case exactly: "real fixture, genuine wager, but
#: the model's inputs described a different club". The wager was genuine — every
#: one of the 13 picks took a real price from a market this defect never
#: touched — and the model's 1X2 features were inverted.
EVIDENCE_STATUS = "void_corrupt_features"

#: The defect was fixed on 2026-10-10. Scoping to kickoffs before 10-11 stops a
#: future genuine single-book gap from being swept in by the same signature.
FIXED_BEFORE = "2026-10-11"

_AFFECTED_MATCHES = """
    SELECT DISTINCT z.match_id
    FROM (
        SELECT o.match_id, o.bookmaker,
               count(*) FILTER (WHERE o.selection = 'Home') AS h,
               count(*) FILTER (WHERE o.selection = 'Away') AS a
        FROM odds o
        JOIN matches m ON m.id = o.match_id
        WHERE o.market_type = '1X2'
          AND o.bookmaker LIKE 'TheOddsAPI-%%'
          AND m.match_date < :fixed_before
        GROUP BY o.match_id, o.bookmaker
    ) z
    WHERE z.a = 0 AND z.h >= 1
"""

#: The second signature, where snapshot coverage exists.
_DUP_CONFIRMED = """
    SELECT DISTINCT y.match_id
    FROM (
        SELECT match_id, bookmaker, date_trunc('second', observed_at) AS sec,
               count(*) AS n
        FROM odds_snapshots
        WHERE market_type = '1X2' AND selection = 'Home'
        GROUP BY 1, 2, 3
        HAVING count(*) > 1
    ) y
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true",
                    help="write the markings (default: report only)")
    args = ap.parse_args(argv)

    # `.env` is read HERE, never at import: an import-time load re-introduces
    # DATABASE_URL after conftest strips it, which once let a unit test write
    # to production.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    from sqlalchemy import text

    from src.data.database import get_db

    db = get_db()
    if not db.is_postgres:
        print("REFUSING: not connected to the production database. Marking a "
              "local SQLite file would record a judgement nothing reads.",
              file=sys.stderr)
        return 2

    with db.get_session() as s:
        matches = [r[0] for r in s.execute(
            text(_AFFECTED_MATCHES), {"fixed_before": FIXED_BEFORE}).fetchall()]
        confirmed = {r[0] for r in s.execute(text(_DUP_CONFIRMED)).fetchall()}
        picks = s.execute(text(
            "SELECT id, match_id, market, selection, odds, is_paper, result, "
            "       evidence_status "
            "FROM saved_picks WHERE match_id = ANY(:ids) ORDER BY id"),
            {"ids": matches}).fetchall()
        settled = s.execute(text(
            "SELECT count(*) FROM matches "
            "WHERE id = ANY(:ids) AND home_goals IS NOT NULL"),
            {"ids": matches}).scalar()
        fitting = s.execute(text(
            "SELECT count(*) FROM matches WHERE home_goals IS NOT NULL "
            "AND training_exclusion_reason IS NULL")).scalar()
        already = s.execute(text(
            "SELECT count(*) FROM matches WHERE id = ANY(:ids) "
            "AND training_exclusion_reason IS NOT NULL"),
            {"ids": matches}).scalar()

    conf = [m for m in matches if m in confirmed]
    pres = [m for m in matches if m not in confirmed]

    print("=" * 78)
    print("1X2 ASSIGNMENT CORRUPTION — MARKING" +
          ("" if args.apply else "  [DRY RUN — nothing written]"))
    print("=" * 78)
    print(f"  affected matches (TheOddsAPI, kickoff < {FIXED_BEFORE}) : {len(matches)}")
    print(f"      CONFIRMED by duplicate-Home signature              : {len(conf)}")
    print(f"      PRESUMED — outside odds_snapshots coverage          : {len(pres)}")
    print(f"      (a zero from absent instrumentation is not a clean bill)")
    print(f"  already carrying a training_exclusion_reason            : {already}")
    print(f"  of them SETTLED, i.e. in the fitting set                : {settled}")
    print(f"  fitting set total (settled, not excluded)               : {fitting}")
    if fitting:
        print(f"  share of the fitting set removed                        : "
              f"{100.0 * settled / fitting:.3f}%")
    print(f"  picks on affected matches                               : {len(picks)}")
    for pid, mid, market, sel, odds, paper, result, ev in picks:
        tag = "paper" if paper else "LIVE "
        print(f"      pick {pid:<6} match {mid:<6} {tag} "
              f"{(result or 'pending'):<8} {market:<14} {sel:<22} @ {odds}"
              + (f"   [already {ev}]" if ev else ""))

    if not args.apply:
        print("\n  Dry run. Re-run with --apply to write the two markings.")
        return 0

    with db.get_session() as s:
        s.execute(text(
            "UPDATE matches SET training_exclusion_reason = :r "
            "WHERE id = ANY(:ids) AND training_exclusion_reason IS NULL"),
            {"r": EXCLUSION_REASON, "ids": matches})
        # Write-once by validator, and re-writing the SAME value is allowed, so
        # the filter keeps this idempotent without relying on that.
        s.execute(text(
            "UPDATE saved_picks SET evidence_status = :e "
            "WHERE match_id = ANY(:ids) AND evidence_status IS NULL"),
            {"e": EVIDENCE_STATUS, "ids": matches})
        s.commit()

    with db.get_session() as s:
        m_done = s.execute(text(
            "SELECT count(*) FROM matches WHERE id = ANY(:ids) "
            "AND training_exclusion_reason = :r"),
            {"ids": matches, "r": EXCLUSION_REASON}).scalar()
        p_done = s.execute(text(
            "SELECT count(*) FROM saved_picks WHERE match_id = ANY(:ids) "
            "AND evidence_status = :e"),
            {"ids": matches, "e": EVIDENCE_STATUS}).scalar()
    print(f"\n  MARKED: {m_done} match(es) {EXCLUSION_REASON!r}, "
          f"{p_done} pick(s) {EVIDENCE_STATUS!r}")
    print("  `odds` is UNCHANGED — it is the evidence.")
    print("  The Parquet mirror self-corrects on its row-count reconcile; the "
          "ML pickles do NOT, so run --train to close that window now.")
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())
