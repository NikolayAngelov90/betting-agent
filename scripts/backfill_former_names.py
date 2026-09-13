"""Backfill `team_former_names` from s5.10's merge log. ONE-OFF.

    DRY RUN BY DEFAULT.  `--apply` writes.

s5.10 merged 129 team rows on 2026-09-10 and ran `DELETE FROM teams` without
recording the names it removed. Within a day 26 were re-created; by 09-13 it was
44, and **100% of them are EXACT matches to a merged-away name** — no diacritic,
punctuation or token differences at all.

    AN OPERATION THAT REMOVES AN IDENTIFIER MUST RECORD IT, OR IT LEAVES BEHIND
    THE REASON THE IDENTIFIER EXISTED.

This recovers what s5.10 should have written, from the only durable record of
it: the apply log. Run it once; every future merge writes its own entries at
merge time and will never need this.

WHY EXACT, AND WHY NOT AN ALIAS TABLE. The lookup that consumes these is an
exact string match. It is deliberately not `team_names_similar`: unioning that
comparator was measured on 2026-09-13 and produced 38 new matches of which
roughly half are absurd (`ac milan` == `manchester utd`, `cremonese` == `usa`),
because it is a RATIO and the raw-versus-aliased cross product invents
comparisons neither pure form performs. An exact match has no ratio, no cross
product and no deleted-token hazard, so none of the 75 alias rulings are needed
for this purpose.

REFUSES ON AMBIGUITY, exactly as s5.10 refused the two `Sporting Clube` rows
rather than guessing between `Sporting CP` and `Braga`. A name that cannot be
resolved to exactly one surviving club is reported and skipped.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

from dotenv import load_dotenv
from sqlalchemy import text

from src.data.database import DatabaseManager

load_dotenv(".env")

REVISION = "s5.10"

#: s5.10's merge commit. A row carrying a merged-away name and created AFTER
#: this is a resurrection; one created before is a different club.
MERGE_AT = "2026-09-10 12:15"


def merge_pairs(log_text: str):
    """(removed_name, surviving_name) from the apply log. Both forms."""
    pairs = []
    # OP1: "af=74   keep 564  <- 564 Sheff Wed || 1575 Sheffield Wednesday || ..."
    for m in re.finditer(r"af=\S+\s+keep\s+(\d+)\s+<-\s+(.+)", log_text):
        winner_id = int(m.group(1))
        members = []
        for part in m.group(2).split("||"):
            mm = re.match(r"\s*(\d+)\s+(.+?)\s*$", part)
            if mm:
                members.append((int(mm.group(1)), mm.group(2).strip()))
        wname = next((n for i, n in members if i == winner_id), None)
        if wname is None:
            continue
        for i, n in members:
            if i != winner_id:
                pairs.append((n, wname))
    # OP2: "   18 Man City   ->    34 Manchester City   af=50   [E1... ]"
    for m in re.finditer(
            r"^\s+(\d+)\s+(.+?)\s+->\s+(\d+)\s+(.+?)\s+af=\S+\s+\[",
            log_text, re.M):
        pairs.append((m.group(2).strip(), m.group(4).strip()))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, help="s5.10 apply log")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    text_ = pathlib.Path(a.log).read_text(encoding="utf-8", errors="replace")
    pairs = merge_pairs(text_)
    print(f"merge pairs recovered: {len(pairs)}")

    db = DatabaseManager()
    with db.get_session() as s:
        rows = s.execute(text(
            "SELECT id, name, created_at, apifootball_team_id FROM teams")).fetchall()
        by_name = {}
        live_by_name = {}
        for r in rows:
            by_name.setdefault(r.name, []).append(r.id)
            live_by_name.setdefault(r.name, []).append(r)

        writes, ambiguous, missing, self_ref = [], [], [], []
        for removed, surviving in pairs:
            if removed == surviving:
                self_ref.append(removed)
                continue
            ids = by_name.get(surviving, [])
            if len(ids) == 0:
                missing.append((removed, surviving))
                continue
            if len(ids) > 1:
                ambiguous.append((removed, surviving, ids))
                continue
            # A name that still exists as a LIVE team needs care, and the
            # first version of this guard refused all 69 such cases — which
            # were precisely the ones the stage exists to close.
            #
            # The distinction that matters is WHEN the live row appeared. A row
            # created AFTER the merge carrying this name is a RESURRECTION: the
            # merge removed the name, the scraper re-created it, and recording
            # the former name is exactly how the next scrape stops doing that.
            # A row that PREDATES the merge is a different club that legitimately
            # holds the name, and claiming it would override a real club.
            live_rows = live_by_name.get(removed, [])
            pre_existing = [r for r in live_rows
                            if str(r.created_at) < MERGE_AT]
            if pre_existing:
                ambiguous.append((removed, surviving,
                                  f"live row {pre_existing[0].id} PREDATES the merge"))
                continue
            writes.append((removed, ids[0]))

        seen = {}
        conflicts = []
        for n, tid in writes:
            if n in seen and seen[n] != tid:
                conflicts.append((n, seen[n], tid))
            seen[n] = tid

        print(f"  resolvable to exactly one surviving club : {len(seen)}")
        print(f"  REFUSED, ambiguous                       : {len(ambiguous)}")
        for r in ambiguous[:10]:
            print(f"      {r[0]!r} -> {r[1]!r}  ({r[2]})")
        print(f"  REFUSED, surviving name not found        : {len(missing)}")
        for r in missing[:5]:
            print(f"      {r[0]!r} -> {r[1]!r}")
        print(f"  skipped, name unchanged by the merge     : {len(self_ref)}")
        print(f"  CONFLICTS (one name, two clubs)          : {len(conflicts)}")
        for c in conflicts:
            print(f"      {c[0]!r} claimed by {c[1]} and {c[2]}")
        if conflicts:
            print("  ABORT: a name resolving to two clubs is not an identity.")
            return 1

        if not a.apply:
            print("\n  DRY RUN — nothing written. Re-run with --apply.")
            for n, tid in list(seen.items())[:12]:
                print(f"      {n!r} -> team {tid}")
            return 0

        for n, tid in seen.items():
            s.execute(text("""
                INSERT INTO team_former_names (name, team_id, source, revision)
                VALUES (:n, :t, 'merge', :r)
                ON CONFLICT (name) DO NOTHING
            """), {"n": n, "t": tid, "r": REVISION})
        s.commit()
        total = s.execute(text("SELECT count(*) FROM team_former_names")).scalar()
        print(f"\n  WRITTEN. team_former_names now holds {total} rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
