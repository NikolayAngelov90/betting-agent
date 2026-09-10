"""Stage 22 — team identity repair. ONE STAGE, FOUR SYMPTOMS, ONE SUBJECT.

    DRY RUN BY DEFAULT.  `--apply` writes.

`teams.apifootball_team_id` is incompletely and sometimes wrongly populated, and
every mechanism keyed on it inherits the gap. Specified in
`docs/stage22-team-identity-repair.md`; this is the executable form.

OPERATION ORDER — CORRECTED FROM THE SPEC, AND THE REASON MATTERS
-----------------------------------------------------------------
The spec ordered the operations 1, 2, 3. That order is WRONG, and running it
would have corrupted operation 2.

Operation 2 finds twins by E1 (shared fixture), which ANCHORS ON PROVIDER IDS:
two match rows are the same fixture when one slot's teams share an
`apifootball_team_id`. Operation 3 exists precisely because two rows hold the
WRONG provider id. Running 2 before 3 lets a known-wrong id anchor a merge —
and it did: with row 411 (`Rakow`) still holding af=350, E1 proposed
`Cracovia -> Rakow`, which is two distinct Polish clubs.

    A repair that consumes the field a later step is about to fix must run
    after that step, not before it.

So: OP3 (clear the wrong ids) -> OP1 (provable merges) -> OP2 (evidenced
merges, recomputed on the repaired field).

HOW OPERATION 2 DECIDES, AND THE TWO VETOES THAT WERE TRIED AND REJECTED
------------------------------------------------------------------------
The spec forbids a bare name match. Evidence, strongest first:

  E1  SHARED FIXTURE — s5.9 branch 1 in reverse. Unresolved U and resolved R
      occupy the same slot of two match rows that are the same fixture by
      independent evidence (same league, same kickoff, the OTHER slot sharing a
      provider club id). No name is consulted.
  E2  same_team_strict — the conservative write-path comparator, used only when
      E1 offers nothing.

E1 IS NOT PROOF, because `matches` itself carries mis-resolved rows. Measured:
14 `france/ligue-2` rows place `St. Pauli` (row 66) in fixtures belonging to
`Pau FC` (row 341). They arrived from API-Football (`apifootball_id` set) while
the genuine rows came from Flashscore (`apifootball_id` NULL), and the cause is
`_tok_match`'s prefix rule — "pauli".startswith("pau") is True. E1 faithfully
reported the consequence and proposed fusing two clubs.

    A comparison is only as good as the resolution state of its inputs.

Two vetoes were measured and REJECTED before the one used here:

  * PRIMARY DOMESTIC LEAGUE — rejects `Wrexham AFC`/`Wrexham` and `Celtic FC`/
    `Celtic`, whose unresolved rows appear only in European ties, and caught
    only 1 of 3 known-bad merges.
  * team_names_similar — MISSES the worst case (it returns True for
    "Pau FC"/"St. Pauli", by the very prefix rule that created the corruption)
    while rejecting nine correct merges.

Both reason about names. The one adopted does not:

    ## A CLUB CANNOT PLAY TWO DIFFERENT FIXTURES AT THE SAME TIME,
    ## AND CANNOT PLAY ITSELF.

If U and R are the same club, no row holding U may collide in time with a row
holding R unless the two rows are the same fixture, and no row may hold both.
A collision is PROOF of distinctness that no lexical test can fake. Measured:
it disqualifies 3 of 3 known-bad merges and 1 further case, and rejects nothing
else. It can only ever REFUSE a merge, never create one — refusing is free, a
wrong merge fuses two clubs' histories and is not.

WHAT THIS DOES NOT DO
---------------------
Operation 4 — CREATE Maccabi Tel Aviv — is NOT performed here. Clearing row
124's wrong id unblocks creation; it does not perform it. Nothing will create
the row until Maccabi next appears in a fetched fixture, which depends on its
European participation and is not in this system's control. Stated separately
so a reader does not assume OP3 completed OP4.

It also does not dedupe `matches`. Rows that become identical after a merge are
exactly the rows s5.9 can now group at pick time, which is the payoff.

COHORT: SELECTION-AFFECTING. Merging changes which fixtures resolve, therefore
which are priced, therefore which are picked — and Elo/Poisson key on
`team_id`, so a club's split history becomes one rating. ONE BUMP: s5.10.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import timedelta

from dotenv import load_dotenv
from sqlalchemy import text

from src.data.database import DatabaseManager
from src.utils.team_names import same_team_strict

load_dotenv(".env")

#: The two wrong provider ids, with the id each row is ASSERTED to hold. The
#: assertion is verified before the write and the run aborts on a mismatch —
#: a repair that clears a field it did not confirm is a guess.
WRONG_IDS = {124: 604, 411: 350}

#: Two fixtures colliding within this window are treated as simultaneous.
COLLISION_WINDOW = timedelta(hours=2)

#: Rows referencing teams.id that a merge must repoint. `injury_observations`
#: carries no FK and is the one a schema-driven sweep would miss.
REFERENCES = (
    ("matches", "home_team_id"),
    ("matches", "away_team_id"),
    ("players", "team_id"),
    ("injuries", "team_id"),
    ("injury_observations", "team_id"),
)


def snapshot(s):
    """Counts that a repair must preserve, or explain."""
    out = {}
    out["teams"] = s.execute(text("SELECT count(*) FROM teams")).scalar()
    out["resolved"] = s.execute(text(
        "SELECT count(*) FROM teams WHERE apifootball_team_id IS NOT NULL")).scalar()
    out["matches"] = s.execute(text("SELECT count(*) FROM matches")).scalar()
    out["picks"] = s.execute(text("SELECT count(*) FROM saved_picks")).scalar()
    for tbl, col in REFERENCES:
        out[f"{tbl}.{col}"] = s.execute(text(
            f"SELECT count(*) FROM {tbl} WHERE {col} IS NOT NULL")).scalar()
    return out


def discovered_population(s):
    """Fixtures whose BOTH participants resolve — the API-Football route.

    This is the measurement the spec requires in the history entry, taken on
    the same windows before and after so a later reader can attribute a cohort
    difference to the merge rather than guess at it.
    """
    out = {}
    for label, since in (("since 2026-08-01", "2026-08-01"),
                         ("last 365 days", "2025-09-10")):
        both = s.execute(text("""
            SELECT count(*) FROM matches m
            JOIN teams h ON h.id = m.home_team_id
            JOIN teams a ON a.id = m.away_team_id
            WHERE m.match_date >= :d
              AND h.apifootball_team_id IS NOT NULL
              AND a.apifootball_team_id IS NOT NULL
        """), {"d": since}).scalar()
        tot = s.execute(text("SELECT count(*) FROM matches WHERE match_date >= :d"),
                        {"d": since}).scalar()
        out[label] = (both, tot)
    return out


def load(s):
    teams = s.execute(text(
        "SELECT id, name, apifootball_team_id FROM teams")).fetchall()
    matches = s.execute(text("""
        SELECT id, league, match_date, home_team_id, away_team_id
        FROM matches WHERE EXTRACT(microsecond FROM match_date) = 0
    """)).fetchall()
    return teams, matches


def merge(s, loser: int, winner: int) -> dict:
    """Repoint every reference from `loser` to `winner`, then delete `loser`.

    Nothing is deleted until the repoint is verified to have left zero
    references behind, which is the precondition the spec states.
    """
    moved = {}
    for tbl, col in REFERENCES:
        n = s.execute(text(f"SELECT count(*) FROM {tbl} WHERE {col} = :l"),
                      {"l": loser}).scalar()
        moved[f"{tbl}.{col}"] = n
        if n:
            s.execute(text(f"UPDATE {tbl} SET {col} = :w WHERE {col} = :l"),
                      {"w": winner, "l": loser})
    left = sum(s.execute(text(f"SELECT count(*) FROM {tbl} WHERE {col} = :l"),
                         {"l": loser}).scalar()
               for tbl, col in REFERENCES)
    if left:
        raise RuntimeError(
            f"ABORT: {left} references to team {loser} survived the repoint")
    s.execute(text("DELETE FROM teams WHERE id = :l"), {"l": loser})
    return moved


def op3_clear_wrong_ids(s, log):
    log("=" * 78)
    log("OPERATION 3 — clear the two WRONG provider ids  (runs FIRST)")
    log("=" * 78)
    done = []
    for rid, expected in sorted(WRONG_IDS.items()):
        row = s.execute(text(
            "SELECT id, name, apifootball_team_id FROM teams WHERE id = :i"),
            {"i": rid}).fetchone()
        if row is None:
            raise RuntimeError(f"ABORT: team row {rid} does not exist")
        if row.apifootball_team_id != expected:
            raise RuntimeError(
                f"ABORT: row {rid} ({row.name}) holds "
                f"{row.apifootball_team_id}, spec asserts {expected}. "
                "The repair does not clear a field it did not confirm.")
        log(f"    row {rid} {row.name!r} holds af={expected} — CONFIRMED, "
            f"clearing to NULL")
        s.execute(text(
            "UPDATE teams SET apifootball_team_id = NULL WHERE id = :i"),
            {"i": rid})
        done.append(rid)
    log("    NULL-out, not re-assignment: the normal resolution path claims the")
    log("    correct id. Guessing the true id is what put the wrong ones here.")
    return done


def op1_merge_shared_ids(s, log):
    log("")
    log("=" * 78)
    log("OPERATION 1 — merge rows sharing a provider id  (PROVABLE)")
    log("=" * 78)
    groups = defaultdict(list)
    for r in s.execute(text("""
            SELECT id, name, apifootball_team_id FROM teams
            WHERE apifootball_team_id IS NOT NULL ORDER BY id""")).fetchall():
        groups[r.apifootball_team_id].append(r)
    comps = {af: rows for af, rows in groups.items() if len(rows) > 1}
    absorbed = sum(len(v) - 1 for v in comps.values())
    log(f"    components: {len(comps)}   rows absorbed: {absorbed}")
    log("    (the spec's '44' is the PAIR count; two components are 3-way)")
    merged = []
    for af, rows in sorted(comps.items()):
        winner = min(r.id for r in rows)          # keep the lowest id
        names = " || ".join(f"{r.id} {r.name}" for r in rows)
        log(f"      af={af:<7} keep {winner}  <- {names}")
        for r in rows:
            if r.id != winner:
                merge(s, r.id, winner)
                merged.append((r.id, winner, "shared_provider_id", 0))
    return merged


def op2_merge_twins(s, log):
    log("")
    log("=" * 78)
    log("OPERATION 2 — merge unresolved rows into evidenced twins")
    log("=" * 78)
    teams, matches = load(s)
    name = {t.id: t.name for t in teams}
    af = {t.id: t.apifootball_team_id for t in teams}
    unresolved = [t.id for t in teams if t.apifootball_team_id is None]
    resolved = [t.id for t in teams if t.apifootball_team_id is not None]

    plays = defaultdict(list)
    bucket = defaultdict(list)
    for m in matches:
        plays[m.home_team_id].append(m)
        plays[m.away_team_id].append(m)
        bucket[(m.league, m.match_date)].append(m)

    # ---- E1: shared fixture, anchored on a shared provider id -----------
    e1 = defaultdict(lambda: defaultdict(int))
    for grp in bucket.values():
        if len(grp) < 2:
            continue
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                a, b = grp[i], grp[j]
                ha = (af.get(a.home_team_id) is not None
                      and af.get(a.home_team_id) == af.get(b.home_team_id))
                aw = (af.get(a.away_team_id) is not None
                      and af.get(a.away_team_id) == af.get(b.away_team_id))
                if ha and not aw:
                    x, y = a.away_team_id, b.away_team_id
                elif aw and not ha:
                    x, y = a.home_team_id, b.home_team_id
                else:
                    continue
                if x == y:
                    continue
                for u, r in ((x, y), (y, x)):
                    if af.get(u) is None and af.get(r) is not None:
                        e1[u][r] += 1

    # ---- E2: the conservative comparator, only where E1 is silent -------
    e2 = defaultdict(list)
    for u in unresolved:
        if e1.get(u):
            continue
        for r in resolved:
            if same_team_strict(name[u], name[r]):
                e2[u].append(r)

    proposed, refused_multi = {}, []
    for u in unresolved:
        c1, c2 = e1.get(u, {}), e2.get(u, [])
        if len(c1) == 1:
            proposed[u] = (next(iter(c1)), "E1_shared_fixture", next(iter(c1.values())))
        elif len(c1) > 1:
            refused_multi.append((u, "E1 multiple candidates", dict(c1)))
        elif len(c2) == 1:
            proposed[u] = (c2[0], "E2_same_team_strict", 0)
        elif len(c2) > 1:
            refused_multi.append((u, "E2 multiple candidates", c2))

    # ---- the disqualifier: one club, one schedule -----------------------
    def disqualify(u, r):
        for m in plays[u]:
            if {m.home_team_id, m.away_team_id} == {u, r}:
                return "head-to-head: the club would play itself"
        for mu in plays[u]:
            for mr in plays[r]:
                if mu.id == mr.id:
                    continue
                if abs(mu.match_date - mr.match_date) > COLLISION_WINDOW:
                    continue
                if mu.league == mr.league and mu.match_date == mr.match_date:
                    continue                    # duplicate rows of one fixture
                return (f"schedule collision: m{mu.id} {mu.league} vs "
                        f"m{mr.id} {mr.league} at {mu.match_date}")
        return None

    clean, blocked = [], []
    for u, (r, ev, votes) in sorted(proposed.items()):
        why = disqualify(u, r)
        (blocked if why else clean).append((u, r, ev, votes, why))

    log(f"    unresolved rows      : {len(unresolved)}")
    log(f"    proposed by evidence : {len(proposed)}")
    log(f"    DISQUALIFIED         : {len(blocked)}  (schedule conflict)")
    log(f"    refused, ambiguous   : {len(refused_multi)}")
    log(f"    MERGING              : {len(clean)}")
    log(f"    left unresolved      : {len(unresolved) - len(clean)}"
        "   (refusing is free)")
    log("")
    log("    disqualified — PROOF of distinctness, not a heuristic:")
    for u, r, ev, votes, why in blocked:
        log(f"      {u:5d} {name[u][:26]:26s} -> {r:5d} {name[r][:22]:22s} "
            f"[{ev} v={votes}]")
        log(f"            {why}")
    if refused_multi:
        log("")
        log("    refused — more than one candidate:")
        for u, why, cands in refused_multi:
            ids = cands if isinstance(cands, list) else list(cands)
            log(f"      {u:5d} {name[u][:26]:26s} {why}: "
                + ", ".join(f"{k}:{name[k]}" for k in ids))

    log("")
    log("    merging:")
    merged = []
    for u, r, ev, votes, _why in clean:
        log(f"      {u:5d} {name[u][:30]:30s} -> {r:5d} {name[r][:24]:24s} "
            f"af={af[r]:<7} [{ev} v={votes}]")
        merge(s, u, r)
        merged.append((u, r, ev, votes))
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="write. Without it, nothing is changed.")
    args = ap.parse_args()

    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    db = DatabaseManager()
    with db.get_session() as s:
        before = snapshot(s)
        before_team_ids = {r[0] for r in s.execute(
            text("SELECT id FROM teams")).fetchall()}
        pop_before = discovered_population(s)

        log("=" * 78)
        log("STAGE 22 — TEAM IDENTITY REPAIR   "
            + ("*** APPLYING ***" if args.apply else "DRY RUN (no writes)"))
        log("=" * 78)
        log("    before:")
        for k, v in before.items():
            log(f"      {k:28s} {v}")
        log("    discovered-fixture population BEFORE "
            "(both participants resolve):")
        for k, (b, t) in pop_before.items():
            log(f"      {k:20s} {b:6d} / {t:6d} = {100.0*b/t:.2f}%")

        # Every operation writes INSIDE the transaction, in both modes, so a
        # dry run previews exactly what --apply commits. The first version
        # gated the writes on --apply, which meant OP2 read the field OP3 had
        # not yet repaired — reproducing, in the preview, the very ordering
        # defect this script exists to avoid.
        cleared = op3_clear_wrong_ids(s, log)
        m1 = op1_merge_shared_ids(s, log)
        m2 = op2_merge_twins(s, log)

        s.flush()
        after = snapshot(s)
        after_team_ids = {r[0] for r in s.execute(
            text("SELECT id FROM teams")).fetchall()}
        pop_after = discovered_population(s)

        log("")
        log("=" * 78)
        log("VERIFICATION — abort on any mismatch")
        log("=" * 78)
        # The FIRST version of this block demanded that every count be
        # UNCHANGED. It aborted on a live database — players +31, injuries +50,
        # injury_observations +100 — because a scraper was writing CONCURRENTLY
        # between the two snapshots. The abort was correct and the check was
        # not: it asserted the whole database was static, which is not a
        # property of this repair and not a property of production.
        #
        # An invariant must be about what the operation does. A repoint moves
        # references and deletes exactly the rows it merged; it can never
        # decrease a count or remove an unrelated row. So:
        ok = True
        losers = [u for u, _w, *_ in m1] + [u for u, _w, *_ in m2]

        # 1. every merged row is GONE and carries ZERO surviving references.
        still_present = s.execute(
            text("SELECT count(*) FROM teams WHERE id = ANY(:ids)"),
            {"ids": losers}).scalar() if losers else 0
        dangling = 0
        for tbl, col in REFERENCES:
            dangling += s.execute(
                text(f"SELECT count(*) FROM {tbl} WHERE {col} = ANY(:ids)"),
                {"ids": losers}).scalar() if losers else 0
        for label, got in (("merged rows still present", still_present),
                           ("references to merged rows", dangling)):
            good = got == 0
            ok = ok and good
            log(f"    {label:34s} {got:8d}  expected        0  "
                + ("OK" if good else "*** MISMATCH ***"))

        # 2. no OTHER team row vanished. Concurrent creation may ADD ids; it
        #    may not remove one, and neither may this repair.
        vanished = sorted((before_team_ids - set(losers)) - after_team_ids)
        good = not vanished
        ok = ok and good
        log(f"    {'unrelated team rows vanished':34s} {len(vanished):8d}"
            f"  expected        0  " + ("OK" if good else "*** MISMATCH ***"))
        if vanished:
            log(f"        {vanished[:20]}")

        # 3. nothing was DESTROYED. Counts may rise (a concurrent writer); a
        #    fall means this repair lost rows it was only supposed to repoint.
        for label, got, floor in (
                ("matches", after["matches"], before["matches"]),
                ("saved_picks", after["picks"], before["picks"]),
                *[(f"{tbl}.{col}", after[f"{tbl}.{col}"], before[f"{tbl}.{col}"])
                  for tbl, col in REFERENCES]):
            good = got >= floor
            ok = ok and good
            delta = got - floor
            log(f"    {label:34s} {got:8d}  was {floor:8d} "
                f"({delta:+d})  " + ("OK" if good else "*** LOST ROWS ***"))

        if not ok:
            raise RuntimeError("ABORT: verification failed; nothing committed")

        log("")
        log(f"    rows cleared (OP3): {len(cleared)}   "
            f"merged (OP1): {len(m1)}   merged (OP2): {len(m2)}")
        log("")
        log("    discovered-fixture population AFTER:")
        for k in pop_after:
            b0, t0 = pop_before[k]
            b1, t1 = pop_after[k]
            log(f"      {k:20s} {b1:6d} / {t1:6d} = {100.0*b1/t1:.2f}%   "
                f"(was {100.0*b0/t0:.2f}%, {b1-b0:+d} fixtures)")

        log("")
        log("    OPERATION 4 — CREATE Maccabi Tel Aviv: NOT PERFORMED.")
        log("    Clearing row 124 unblocks creation; it does not perform it.")
        log("    Nothing creates the row until Maccabi next appears in a")
        log("    fetched fixture, which depends on European participation.")

        if args.apply:
            s.commit()
            log("")
            log("    COMMITTED.")
        else:
            s.rollback()
            log("")
            log("    DRY RUN — rolled back. Re-run with --apply to write.")


if __name__ == "__main__":
    main()
