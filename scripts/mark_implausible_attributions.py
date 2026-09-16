"""Mark the fixtures a club cannot have played. ONE-OFF, DRY RUN BY DEFAULT.

    `--apply` writes.  Reversible: `training_exclusion_reason` is deliberately
    NOT write-once, because a match's exclusion is a statement about CURRENT
    data quality rather than a judgement about a past event.

WHY THIS RUNS IN THE SAME CHANGE THAT GATES THE PATH. The identity-writing path
now honours `training_exclusion_reason`, so the mark finally does something —
but a gate with nothing marked is a gate with nothing to act on for the
population that motivated it. Detection and enforcement ship together or the
detector is wired to nothing.

THE SET IS NOT "THE 17", AND THAT IS LIMIT 1 OF THE INVARIANT BITING EXACTLY AS
WRITTEN DOWN. `find_implausible_attributions` reports the MINORITY country as a
convenience and explicitly refuses to call it the wrong one. For two of the five
unmarked rows the MAJORITY is the contaminated side:

    1608 `York City`    usa=37  england=4   -> the 37 are NEW YORK CITY FC's
                        entire 2024 MLS season (Cincinnati, Red Bulls, Inter
                        Miami, CF Montreal...). The 4 English League Two
                        fixtures are the real club.
    1353 `NK Varazdin`  azerbaijan=4 croatia=4  -> a tie, broken arbitrarily
                        toward azerbaijan. The club is Croatian; the 4 Neftchi
                        Baku fixtures are an Azerbaijani club's.

So the minority heuristic would have marked 6 LEGITIMATE fixtures and missed 41
contaminated ones. Every row below is decided on its own evidence.

NOT TOUCHED: the `other/world` fixtures on these rows. They are neutral ground
by construction, the invariant exempts them, and deciding them needs a
competition-by-competition judgement this script does not make.
"""
from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from sqlalchemy import text

from src.data.database import DatabaseManager

load_dotenv(".env")

REASON = "corrupt_team_identity"

#: team_id -> (contaminated league country, why). Decided per row from its own
#: fixtures, never from which side happened to be smaller.
DECISIONS = {
    374:  ("cyprus",     "Greek club; 4 other/cyprus v Omonia Nicosia are Aris Limassol's"),
    1528: ("israel",     "Dutch club; 4 other/israel v Hapoel Beer Sheva are Maccabi Tel Aviv's"),
    187:  ("cyprus",     "Turkish club; 3 other/cyprus v Omonia Nicosia are a Cypriot club's"),
    1353: ("azerbaijan", "Croatian club; 4 other/azerbaijan v Neftchi Baku are an Azerbaijani club's"),
    1608: ("usa",        "English League Two club; 37 other/usa are New York City FC's 2024 season"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    db = DatabaseManager()
    with db.get_session() as s:
        total = 0
        for tid, (country, why) in sorted(DECISIONS.items()):
            name = s.execute(text("SELECT name FROM teams WHERE id=:i"),
                             {"i": tid}).scalar()
            rows = s.execute(text("""
                SELECT m.id, m.league, m.match_date, m.training_exclusion_reason x
                FROM matches m
                WHERE (m.home_team_id=:i OR m.away_team_id=:i)
                  AND (m.league LIKE :a OR m.league = :b)
                ORDER BY m.match_date
            """), {"i": tid, "a": f"{country}/%", "b": f"other/{country}"}).fetchall()
            todo = [r for r in rows if not r.x]
            print(f"\n  team {tid} {name!r} — {why}")
            print(f"     {country} fixtures: {len(rows)}   already marked: "
                  f"{len(rows)-len(todo)}   to mark: {len(todo)}")
            for r in todo[:4]:
                print(f"        match {r.id} [{r.league}] {str(r.match_date)[:10]}")
            if len(todo) > 4:
                print(f"        ... and {len(todo)-4} more")
            for r in todo:
                s.execute(text("UPDATE matches SET training_exclusion_reason=:r "
                               "WHERE id=:i"), {"r": REASON, "i": r.id})
            total += len(todo)

        after = s.execute(text(
            "SELECT count(*) FROM matches WHERE training_exclusion_reason=:r"),
            {"r": REASON}).scalar()
        print(f"\n  fixtures marked by this run : {total}")
        print(f"  `{REASON}` total after       : {after}")

        # The invariant must STILL flag these rows — marking records that a
        # fixture is not evidence; it does not make the row plausible. A check
        # that fell silent here would be reporting the mark, not the data.
        from src.data.fixture_plausibility import find_implausible_attributions
        still = find_implausible_attributions(s)
        print(f"  rows the invariant still flags: {len(still)} "
              f"(expected: unchanged — marking is not repair)")

        if not a.apply:
            s.rollback()
            print("\n  DRY RUN — rolled back. Re-run with --apply.")
            return 0
        s.commit()
        print("\n  APPLIED. The identity-writing path will no longer read these "
              "fixtures as evidence.")

    # THE MARK DOES NOT REACH THE LEARNING PATHS ON ITS OWN, AND THAT IS THE
    # SAME RULE ONE LEVEL DOWN.
    #
    # `history_mirror.filter_generation()` digests the PREDICATE'S SOURCE, so it
    # invalidates when the filter changes and NOT when the data does — its
    # docstring says so explicitly. The mirror is also watermark-incremental, so
    # marking a 2024 fixture moves no watermark and the cached row is served as
    # clean forever. The ML pickles share the same generation.
    #
    # So marking without invalidating is detection wired to nothing, one layer
    # below the path this change was written to gate. Both happen here or
    # neither is real.
    try:
        from src.data.history_mirror import HistoryMirror
        HistoryMirror().invalidate()
        print("  History mirror INVALIDATED — the next sync rebuilds it from the "
              "database, which is the only way these 52 marks reach the paths "
              "that learn. The ML pickles share the generation and retrain on "
              "their own staleness clock.")
    except Exception as e:                           # pragma: no cover
        print(f"  WARNING: could not invalidate the history mirror ({e}). "
              f"The marks are written but the cache may still serve the rows — "
              f"run `HistoryMirror().invalidate()` by hand.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
