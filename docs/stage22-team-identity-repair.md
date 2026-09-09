# STAGE 22 — TEAM IDENTITY REPAIR

**Specified 2026-09-09. Not built. Runs AFTER H5.**

**One subject: `teams.apifootball_team_id` is incompletely and sometimes wrongly
populated, and every mechanism keyed on it inherits the gap.** Four symptoms
that have been carried as four separate deferrals for three weeks.

**This is local work.** No API calls for the provable part, no new column, no
30,128-row backfill. **The precedent is documented: 50 clubs merged 2026-07-22
under the same description.**

---

## THE PAYOFF, measured rather than asserted

| | |
| --- | --- |
| s5.9 residual pairs it removes the condition for | **187 of 202 (92.6%)** |
| fixtures since 2026-08-01 touching an unresolved team | **155** |
| unresolved teams appearing in a fixture since 2026-08-01 | **66 of 176** |
| fixtures touching one in the last 365 days | **1,435** |

> **It removes the condition under which s5.9 fails, rather than adding a branch
> or loosening one.** Branch 1 becomes reachable for fixtures that currently
> fall through to branch 2's name comparison and then past it.

**Two of the three known guarantee violations came from this class.**

---

## THE FOUR OPERATIONS, in dependency order

### 1. MERGE the 44 shared-provider-id pairs — PROVABLE, no judgment required

**Two rows carrying the same `apifootball_team_id` are the same club by the
provider's own assertion.** No name comparison is involved and none should be.

```
 564 Sheff Wed           ||  1575 Sheffield Wednesday  ||  1606 Sheffield Wed   af=74
 488 Lille OSC           ||  1623 Lille                                          af=79
 569 Paris SG            ||  1628 Paris Saint Germain                            af=85
 581 Clermont Foot       ||  1551 Clermont                                       af=99
 487 FC Metz             ||  1535 Metz                                           af=112
 576 Racing Club de Lens ||  1624 Lens                                           af=116
```

**`Sheffield Wednesday` is a THREE-way split**, so the merge is over connected
components, not pairs. **Keep the lowest id; repoint `matches.home_team_id` /
`away_team_id`; delete nothing until the repoint is verified.**

### 2. MERGE the 176 unresolved rows into their resolved twins

**These are not obscure clubs — they are duplicates of clubs already in the
table with a resolved id:**

| unresolved | resolved twin |
| --- | --- |
| 764 `Sporting Clube de Portugal` | 228 `Sporting CP` |
| 494 `Celta` | 53 `Celta Vigo` |
| 571 `SBV Excelsior` | 129 `Excelsior` |
| 492 `Sporting Clube de Braga` | 142 `Braga` |
| 490 `Porto`, 563 `Wrexham AFC`, 565 `Charlton Athletic FC`, 445 `Sheffield United` | … |

**PRIORITISE THE 66 that appear in a fixture since 2026-08-01.** The other 110
are historical-only and carry no live cost — **they can be left, and leaving
them is cheaper than a wrong merge.**

**Twin-finding is the stage's own work and it must NOT be a bare name match.**
The available evidence, strongest first: a shared fixture (same league, same
kickoff minute, the other slot resolving to the same club — s5.9's own branch 1
run in reverse), then `same_team_strict`, then a provider lookup. **A row with
no defensible twin stays unresolved. Refusing is free; a wrong merge is not.**

### 3. CORRECT rows 124 and 411 — wrong ids, and they differ in severity

| row | holds | belongs to | what it actually blocks |
| --- | --- | --- | --- |
| **124** `Telstar` | **604** | Maccabi Tel Aviv | **Maccabi has NO row at all** — the gate refuses at step 0 every time, so it can never be created |
| **411** `Rakow` | **350** | Cracovia | Cracovia **row 420 EXISTS** but unresolved: **3 fixtures since 08-01, 1 priced** |

> **They are not equivalent. 124 makes a club invisible; 411 makes one
> unresolvable.** Row 420 exists, so Cracovia is discoverable by other paths and
> merely loses the API-Football route — which is a smaller harm than Maccabi's,
> and the record should not flatten the two.

**Both are `NULL`-out operations**, not re-assignments: clear the wrong id and
let the normal resolution path claim the correct one. **Do not guess Raków's or
Telstar's true id from memory** — that is what put the wrong ones there.

### 4. CREATE Maccabi Tel Aviv — a SECOND operation, not a consequence

**Clearing 124 unblocks creation; it does not perform it.** Nothing will create
the row until Maccabi next appears in a fetched fixture, and **whether that
happens depends on its European participation**, which is not in this system's
control.

**Stated separately so a reader does not assume step 3 completed step 4.**

---

## COHORT

> **SELECTION-AFFECTING. Merging changes which fixtures resolve, therefore which
> are priced, therefore which are picked. ONE BUMP: `s5.10`.**

**The history entry must carry the MEASURED effect on the discovered-fixture
population** — before/after counts of fixtures resolving through the
API-Football path, on the same window — so a later reader can attribute a cohort
difference to the merge rather than guess at it. **That measurement is part of
the stage, not a follow-up.**

---

## WHAT THIS DOES NOT CLOSE

**15 of the 202 residual pairs are a PURE NAMING residual** — both rows already
carry provider ids, and no merge reaches them. Their remedy is a curated alias,
of which the open case is:

> `Sporting Clube de Portugal` / `Sporting CP` — shares the token `sporting`,
> rejected by the **0.7 overlap ratio**. **A threshold is tolerance; an alias is
> knowledge.** Loosening 0.7 is the move refused five times here.

**That decision stays SEPARATE and UNTAKEN.** It is not part of this stage and
must not be folded into it, because a merge and a threshold are different kinds
of change and bundling them would make the cohort break unattributable.

**Also not closed:** branch 3's residual does not go to zero. The stage removes
the *condition* for 187 pairs; it does not prove the class empty.

---

## ORDER

**H5 runs first.** It is free, its sample is present (n=127 against a registered
n≥50), it runs once, and **Q2's outcome bears on the lead-time claim still
marked UNVERIFIED in `stage21-schedule-prediction.md`.**

**This stage carries a cohort break and will still be there tomorrow.**

*Specified 2026-09-09.*
