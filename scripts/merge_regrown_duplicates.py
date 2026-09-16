"""Merge the shared-provider-id duplicate components s5.10 left at ZERO. ONE-OFF.

    DRY RUN BY DEFAULT.  `--apply` writes.

Stage 24. Registered in `docs/stage24-merge-32-and-step2-first-exercise.md`
BEFORE this was written — read that first; it fixes what this should do so the
result cannot be read selectively.

WHY THIS IS SAFE TO RUN NOW AND WAS NOT SAFE ON 09-13. The gate was: merge only
if the resurrection rate has fallen, because merging into a still-leaking system
is the exact mistake that produced this backlog. Measured 09-16: new team rows
0/day for two days, and the duplicate count flat at 32 for three (+0/+0/+0).
**Bounded backlog, not a rate.**

    AN OPERATION THAT REMOVES AN IDENTIFIER MUST RECORD IT.

s5.10 removed 129 names and recorded none, and 44 came back in three days. This
one records as it goes — that is the single property it exists to have.

BUT A FORMER NAME MUST ACTUALLY BE FORMER. Measured before writing this: 29 of
the 30 removed names are ALREADY in `team_former_names` pointing at the correct
survivor, and 2 of the 32 components are exact-name pairs where the name is not
removed at all because the survivor still carries it. **Recording those would
put a CURRENT name in a table of removed ones**, letting step 2 short-circuit
step 3 for no reason. So a name is recorded only when it genuinely leaves
circulation. The predicted write count is 1, not 32.

WHAT IT REPAIRS, and the first item is not the obvious one:

  1. A LIVE NON-DETERMINISM. `resolve_team` step 1 is
     `.filter(apifootball_team_id == provider_id).first()` with NO ordering, and
     32 provider ids currently match two rows. Which row an API-Football fixture
     attaches to is whatever the database returns first.
  2. SPLIT HISTORY. 19 of the 32 duplicates hold real references —
     `1561/1775 Erzurumspor` alone has 46 matches, 1 pick and 91 odds rows on
     the wrong row.
  3. The one genuinely missing former name.

SURVIVOR = LOWEST ID, matching s5.10's OP1. It is the older row, the one whose
history is longest, and the one other tables already reference most.
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

#: Tables carrying a team_id that must follow the survivor. Derived from the
#: schema rather than remembered: a table missed here orphans rows silently,
#: which is the failure mode of every hand-maintained list in this project.
TEAM_REF_COLUMNS = (
    ("matches", "home_team_id"),
    ("matches", "away_team_id"),
    ("players", "team_id"),
    ("injuries", "team_id"),
)


def _existing_refs(session):
    """Only the (table, column) pairs that actually exist in THIS database."""
    out = []
    for tbl, col in TEAM_REF_COLUMNS:
        try:
            session.execute(text(f"SELECT {col} FROM {tbl} LIMIT 1"))
            out.append((tbl, col))
        except Exception:
            session.rollback()
    return out


def components(session):
    """Provider ids held by more than one team row, with their members."""
    rows = session.execute(text("""
        SELECT apifootball_team_id AS af, id, name, league, created_at
        FROM teams
        WHERE apifootball_team_id IS NOT NULL
          AND apifootball_team_id IN (
              SELECT apifootball_team_id FROM teams
              WHERE apifootball_team_id IS NOT NULL
              GROUP BY 1 HAVING count(*) > 1)
        ORDER BY apifootball_team_id, id
    """)).fetchall()
    by_af: dict = {}
    for r in rows:
        by_af.setdefault(r.af, []).append(r)
    return by_af


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    db = DatabaseManager()
    with db.get_session() as s:
        refs = _existing_refs(s)
        print(f"reference columns in this database: "
              f"{', '.join(f'{t}.{c}' for t, c in refs)}\n")

        comps = components(s)
        print(f"shared-provider-id components: {len(comps)}")

        merged = removed = recorded = repointed = 0
        refused = []
        # A CALL IS NOT A WRITE. `record_former_name` is ON CONFLICT DO NOTHING,
        # so 30 calls against 29 existing rows write ONE row. The registration's
        # falsifier is "more than 1 former name WRITTEN", and a counter that
        # reported 30 would read as a failed prediction while the truth was 1.
        # Same class as a definition read as an occurrence: count the effect,
        # not the attempt.
        fn_before = s.execute(
            text("SELECT count(*) FROM team_former_names")).scalar()
        # ALWAYS WRITE, THEN ROLL BACK UNLESS --apply. A dry run that skips the
        # writes is not a preview of the apply: later steps then read state the
        # earlier ones did not change. Stage 22 shipped that bug and its dry run
        # disagreed with its apply.
        for af, members in sorted(comps.items()):
            surv, dead = members[0], members[1:]
            # THE ONE DISQUALIFIER THAT CAN ACTUALLY FIRE. A provider-id check
            # here would be vacuous — the query groups BY provider id, so every
            # member carries it by construction, and the first version of this
            # guard crashed on a column it had not selected. A guard that cannot
            # fail is not a guard; this one can.
            #
            # National teams and clubs are disjoint identity spaces (the USA/
            # Lausanne match is why `_partition_filter` exists). A provider id
            # spanning both is a provider error, not a duplicate, and merging it
            # would fuse a country into a club.
            from src.models.poisson_model import NATIONAL_TEAM_LEAGUES
            parts = {(m.league in NATIONAL_TEAM_LEAGUES) for m in members}
            if len(parts) > 1:
                refused.append(
                    (af, "members span the national/club identity partition"))
                continue
            for d in dead:
                # A name is recorded ONLY if it leaves circulation. When the
                # survivor carries the identical name, nothing was removed.
                if d.name != surv.name:
                    record_former_name(s, d.name, surv.id, REVISION)
                    recorded += 1
                for tbl, col in refs:
                    n = s.execute(
                        text(f"UPDATE {tbl} SET {col} = :sv WHERE {col} = :dd"),
                        {"sv": surv.id, "dd": d.id}).rowcount or 0
                    repointed += n
                s.execute(text("DELETE FROM teams WHERE id = :i"), {"i": d.id})
                removed += 1
                print(f"  af={af:<7} keep {surv.id} {surv.name!r} "
                      f"<- drop {d.id} {d.name!r}"
                      + ("" if d.name != surv.name else "   [same name: nothing recorded]"))
            merged += 1

        # ---- verification, as INVARIANTS rather than as a static snapshot ----
        # Stage 22's check demanded an unchanging database and aborted on a live
        # one. These hold whatever else is writing concurrently.
        left = len(components(s))
        total_fn = s.execute(
            text("SELECT count(*) FROM team_former_names")).scalar()
        orphans = 0
        for tbl, col in refs:
            orphans += s.execute(text(
                f"SELECT count(*) FROM {tbl} x WHERE x.{col} IS NOT NULL AND "
                f"NOT EXISTS (SELECT 1 FROM teams t WHERE t.id = x.{col})"
            )).scalar() or 0

        print(f"\n  components merged      : {merged}")
        print(f"  team rows removed      : {removed}")
        print(f"  references re-pointed  : {repointed}")
        print(f"  record_former_name CALLS: {recorded}")
        print(f"  former names WRITTEN    : {total_fn - fn_before}   "
              f"<- the registered figure ({recorded - (total_fn - fn_before)} "
              f"already present)")
        print(f"  team_former_names total: {total_fn}")
        print(f"  components REMAINING   : {left}")
        print(f"  orphaned references    : {orphans}")
        for af, why in refused:
            print(f"  REFUSED af={af}: {why}")

        ok = (left == 0 and orphans == 0 and not refused)
        if not ok:
            print("\n  ABORT: invariants not met — rolling back.")
            s.rollback()
            return 1

        if not a.apply:
            s.rollback()
            print("\n  DRY RUN — rolled back, nothing written. Re-run with --apply.")
            return 0

        s.commit()
        print("\n  APPLIED.")
        print("  Step 2's first production hit is NOT caused by this merge — it "
              "comes from the next scrape of a recorded name by a path that "
              "passes no provider id. Watch for:")
        print("    resolve_team: '<name>' is a FORMER NAME of team <id> ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
