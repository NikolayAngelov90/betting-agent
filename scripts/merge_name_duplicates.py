"""Merge the 16 duplicate rows step 2 has been routing around. ONE-OFF.

    DRY RUN BY DEFAULT.  `--apply` writes.

A MERGE LIST DERIVED FROM PRODUCTION RATHER THAN FROM A SCAN. It did not exist
before 2026-09-17: `resolve_team` step 2 now logs at INFO, and its own output
names these clubs by name every day. 189 hits across 38 names, of which 16 have
a LIVE duplicate row — the 09-11/09-13 resurrection wave Stage 24 could not
merge because those rows carry no provider id, so `apifootball_team_id` grouping
could not see them.

WHY IT MATTERS THAT THEY ARE BEING ROUTED AROUND RATHER THAN REMOVED.
Correctness is currently coming from a lookup rather than from the data being
clean. If step 2 ever fails — a bad alias, a scope change, a sink level — those
fixtures snap back onto the duplicate rows without a word. The 88-day Flashscore
death is what a silent revert looks like.

THE REGISTERED PREDICTION IS THAT STEP-2 HITS DO NOT FALL.

    All 16 names are ALREADY in `team_former_names` — that is WHY step 2 fires
    on them. Step 2 keys on the NAME being a former name, not on whether a
    duplicate row exists. Remove the row and the name is still a former name, so
    the lookup still intercepts and still returns the survivor.

    predicted: rows removed 16 · references re-pointed 29 · former names
    WRITTEN 0 (all 16 already present) · step-2 hits UNCHANGED (~76 from these
    names on a comparable card).

    **If the hits fall, step 2 is keyed on something other than the former-name
    table and the model of it is wrong.** That falsifier is worth more than the
    pass, which is the reason to register it this way round.

record_former_name() IS CALLED EVEN THOUGH IT WILL WRITE NOTHING:

    AN OPERATION THAT REMOVES AN IDENTIFIER RECORDS IT. The property is the
    invariant, not the usual case. A call that writes zero rows today is what
    makes the next removal safe.

That is the rule s5.10 lacked, stated as a property rather than as a remedy.
"""
from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from sqlalchemy import text

from src.data.database import DatabaseManager
from src.data.team_resolution import record_former_name

load_dotenv(".env")

REVISION = "s5.13"

TEAM_REF_COLUMNS = (("matches", "home_team_id"), ("matches", "away_team_id"),
                    ("players", "team_id"), ("injuries", "team_id"))


def _existing_refs(session):
    out = []
    for tbl, col in TEAM_REF_COLUMNS:
        try:
            session.execute(text(f"SELECT {col} FROM {tbl} LIMIT 1"))
            out.append((tbl, col))
        except Exception:
            session.rollback()
    return out


def pairs(session):
    """(duplicate row, survivor) for every live row whose NAME is a former name.

    Derived from the data, not from a hand list: a row whose own name resolves —
    via `team_former_names` — to a DIFFERENT row is by construction the thing
    step 2 is routing around.
    """
    return session.execute(text("""
        SELECT t.id AS dup_id, t.name AS dup_name, t.apifootball_team_id AS dup_af,
               s.id AS surv_id, s.name AS surv_name, s.apifootball_team_id AS surv_af
        FROM teams t
        JOIN team_former_names f ON f.name = t.name
        JOIN teams s ON s.id = f.team_id
        WHERE s.id <> t.id
        ORDER BY t.id
    """)).fetchall()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    db = DatabaseManager()
    with db.get_session() as s:
        refs = _existing_refs(s)
        rows = pairs(s)
        fn_before = s.execute(
            text("SELECT count(*) FROM team_former_names")).scalar()

        print(f"duplicate rows step 2 is routing around: {len(rows)}\n")
        removed = repointed = calls = 0
        refused = []
        for r in rows:
            # A provider-id mismatch is the provider asserting two clubs.
            if r.dup_af and r.surv_af and r.dup_af != r.surv_af:
                refused.append((r.dup_id, f"af {r.dup_af} != survivor af {r.surv_af}"))
                continue
            # RECORDED EVEN WHEN IT WRITES NOTHING. The property is the
            # invariant; `ON CONFLICT DO NOTHING` makes the no-op cheap.
            record_former_name(s, r.dup_name, r.surv_id, REVISION)
            calls += 1
            n = 0
            for tbl, col in refs:
                n += s.execute(
                    text(f"UPDATE {tbl} SET {col}=:sv WHERE {col}=:dd"),
                    {"sv": r.surv_id, "dd": r.dup_id}).rowcount or 0
            if r.dup_af and not r.surv_af:
                s.execute(text("UPDATE teams SET apifootball_team_id=:a WHERE id=:i"),
                          {"a": r.dup_af, "i": r.surv_id})
            s.execute(text("DELETE FROM teams WHERE id=:i"), {"i": r.dup_id})
            removed += 1
            repointed += n
            print(f"  drop {r.dup_id:<5} {r.dup_name[:26]:<26} -> keep {r.surv_id:<5} "
                  f"{r.surv_name[:26]:<26} refs={n}")

        left = len(pairs(s))
        fn_after = s.execute(
            text("SELECT count(*) FROM team_former_names")).scalar()
        orphans = 0
        for tbl, col in refs:
            orphans += s.execute(text(
                f"SELECT count(*) FROM {tbl} x WHERE x.{col} IS NOT NULL AND NOT "
                f"EXISTS (SELECT 1 FROM teams t WHERE t.id=x.{col})")).scalar() or 0

        print(f"\n  rows removed            : {removed}")
        print(f"  references re-pointed   : {repointed}")
        print(f"  record_former_name CALLS: {calls}")
        print(f"  former names WRITTEN    : {fn_after - fn_before}   "
              f"<- registered as 0; a call is not a write")
        print(f"  pairs REMAINING         : {left}")
        print(f"  orphaned references     : {orphans}")
        for i, why in refused:
            print(f"  REFUSED {i}: {why}")

        if left or orphans:
            print("\n  ABORT: invariants not met — rolling back.")
            s.rollback()
            return 1
        if not a.apply:
            s.rollback()
            print("\n  DRY RUN — rolled back. Re-run with --apply.")
            return 0
        s.commit()
        print("\n  APPLIED. Step-2 hits are registered to stay UNCHANGED; a fall "
              "falsifies the model of step 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
