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

---

# APPLIED — 2026-09-10, as `s5.10`

**Executable form: `scripts/stage22_team_identity_repair.py`. Dry run by
default; `--apply` commits. Applied once, verified, committed.**

## The measured payoff, which is the part the spec required in the history entry

**Fixtures whose BOTH participants resolve to a provider id — the API-Football
route, and s5.9 branch 1's precondition:**

| window | before | after | change |
| --- | --- | --- | --- |
| since 2026-08-01 | 1,236 / 1,544 = **80.05%** | 1,378 / 1,544 = **89.25%** | **+142 fixtures** |
| last 365 days | 8,108 / 9,717 = **83.44%** | 9,160 / 9,717 = **94.27%** | **+1,052 fixtures** |

| | before | after |
| --- | --- | --- |
| team rows | 1,577 | **1,448** |
| unresolved | 178 | **94** |
| provider ids held by more than one row | 40 | **0** |
| dangling foreign keys | 0 | **0** |

**2 wrong ids cleared, 45 rows absorbed into provable components, 84 unresolved
rows merged into evidenced twins.** No match, pick or reference row was
destroyed.

## THREE CORRECTIONS TO THIS SPEC, found by building it

### 1. The operation ORDER was wrong, and running it as written would have fused two clubs

The spec ordered the merges (1, 2) before the wrong-id clearance (3). **But the
twin-finder anchors on provider ids, and the clearance exists precisely because
two of those ids are wrong.**

**It is not hypothetical. With row 411 (`Rakow`) still holding af=350, the
evidence proposed `Cracovia -> Rakow` — two distinct Polish clubs.** Reordered
to clearance-first, that proposal does not arise at all: `Rakow` becomes
unresolved, is proposed against `Raków Częstochowa` instead, and is then refused
on head-to-head.

> **A repair that consumes the field a later step is about to fix must run
> AFTER that step, not before it.**

### 2. "44 provider ids held by more than one row" was the PAIR count

**40 components, 44 pairs, 42 rows absorbed** at specification time (45 by the
time it ran, the table having grown). Two components are three-way, so pairs and
components differ. The spec's own text called them "provider ids", which is the
component count — the number quoted was the pair count.

### 3. The dry run was not previewing what `--apply` would do

The first implementation gated each write on `--apply`. **So in dry run, OP2 read
the field OP3 had not yet repaired — reproducing, inside the preview, the exact
ordering defect the reordering existed to prevent.** Fixed by always writing
inside the transaction and rolling back when not applying. **A preview that does
not execute the earlier steps is not a preview of the later ones.**

## HOW THE MERGES WERE DECIDED — two vetoes measured and rejected first

**Evidence is a SHARED FIXTURE (E1) — s5.9 branch 1 run in reverse, no name
consulted — falling back to `same_team_strict` (E2) only where E1 is silent.**

**E1 IS NOT PROOF.** `matches` itself carries mis-resolved rows: **14
`france/ligue-2` rows place `St. Pauli` (row 66) in fixtures belonging to
`Pau FC` (row 341).** They came from API-Football (`apifootball_id` set) while
the genuine rows came from Flashscore (`apifootball_id` NULL), and the cause is
`_tok_match`'s prefix rule — `"pauli".startswith("pau")` is True. **E1
faithfully reported the consequence and proposed fusing two clubs.**

> **A comparison is only as good as the resolution state of its inputs** — the
> guard-design rule, arriving inside the repair that was written to apply it.

**Two vetoes were measured against the three known-bad merges and REJECTED:**

| veto | why rejected |
| --- | --- |
| **primary domestic league** | rejects `Wrexham AFC`/`Wrexham` and `Celtic FC`/`Celtic`, whose unresolved rows appear only in European ties. **Caught 1 of 3.** |
| **`team_names_similar`** | **MISSES the worst case** — returns True for `Pau FC`/`St. Pauli`, by the very prefix rule that created the corruption — while rejecting nine correct merges |

**Both reason about names. The rule adopted does not:**

> ## A CLUB CANNOT PLAY TWO DIFFERENT FIXTURES AT THE SAME TIME, AND CANNOT PLAY ITSELF.

**It disqualified 3 of 3 known-bad merges and one further case, and rejected
nothing else:**

| refused | proof |
| --- | --- |
| `Pau FC` → `St. Pauli` | schedule collision: m51201 `france/ligue-2` vs m51195 `germany/2-bundesliga`, both 2026-09-04 18:00 |
| `Rakow` → `Raków Częstochowa` | head-to-head — the club would play itself |
| `BW Linz` → `LASK` | head-to-head |
| `Sport Lisboa e Benfica` → `Benfica` | schedule collision within `portugal/primeira-liga` |

**It can only ever REFUSE a merge, never create one.** Refusing is free; a wrong
merge fuses two clubs' histories.

**Two more were refused as AMBIGUOUS** — `Sporting Clube de Braga` and
`Sporting Clube de Portugal` each drew two candidates (`Sporting CP` **and**
`Braga`). **That is the Sporting CP residual this spec ruled must stay separate
and untaken, and it stayed untaken.** A threshold is tolerance; an alias is
knowledge; and bundling either into a merge would make the cohort break
unattributable.

## A FIFTH SYMPTOM, FOUND AND NOT FIXED

**Match rows assigned to the WRONG TEAM ROW by name-first matching at
ingestion** — the `Pau FC` / `St. Pauli` class, 14 rows on one club. The merge
is guarded against it and does not propagate it, **but the rows remain
mis-assigned and the `_tok_match` prefix rule that creates them is unchanged.**

**Recorded as its own open item rather than folded into this stage**, for the
reason this stage exists: four symptoms of one subject were carried as four
deferrals, and the fix was to name the subject — not to keep widening a stage
until it swallows every neighbour.

**Not closed either:** branch 3's residual does not go to zero. This removes the
*condition* for most of it; it does not prove the class empty. And **Maccabi Tel
Aviv was NOT created** — clearing row 124 unblocks creation, it does not perform
it, and nothing creates the row until Maccabi next appears in a fetched fixture.

*Applied 2026-09-10. `s5.10`, one bump, cohort empty at the time of writing.*

## The s5.9 residual, re-measured after the repair

**Duplicate-fixture pairs in the last 365 days, by the branch that reaches them:**

| branch | pairs |
| --- | --- |
| **1 — shared provider id (PROVABLE)** | **129** |
| 2 — both stored names similar | 2 |
| **RESIDUAL — neither** | **28** |

**The spec measured the residual at 202 before the repair and predicted the
condition would be removed for 187 of them.** After: **28**.

> **Stated with its caveat rather than as a clean 202 → 28.** The before-figure
> was measured on the spec's own predicate and the after-figure on the one above;
> they are close but not provably identical instruments, and the repair has
> already destroyed the state needed to re-run the first. **The direction and
> rough magnitude are established; the exact delta is not**, and this project
> does not quote a difference between two instruments as though it came from one.

---

## THE FIFTH SYMPTOM HAS A RATE — 20% of this stage undone in one day

**Measured 2026-09-11, the day after the merge.** Of 27 new unresolved team rows
created since, **26 are EXACT-NAME resurrections of rows this stage merged away**.
129 rows merged; 26 back within ~24 hours; **the whole merge undone in about five
days at that rate**.

**The mechanism is this stage's own survivor-selection rule.** OP1 kept the
LOWEST ID, which is frequently not the name the scraper writes — `1624 Lens` was
merged into `576 Racing Club de Lens`, and Flashscore goes on writing "Lens".
`same_team_strict` correctly refuses to equate them, so a new row is created.
**The merge removed the row and left the reason it existed.**

Six of eight sampled names are explained by that; `PSG`/`Paris SG` and
`Metz`/`FC Metz` return True from `same_team_strict` and were re-created anyway,
so **at least one other creation path exists — flagged, not diagnosed.**

> **A repair that deletes rows without changing what creates them has a
> half-life.** This one's is roughly two and a half days.

**The successor item is not "ingestion sometimes mis-assigns".** It is: attach
the source's names to the merge survivors as curated aliases, so the write path
matches instead of twinning. *A threshold is tolerance; an alias is knowledge.*
Same class as the NEC alias, and it is the thing that would make this stage
durable rather than momentary.

*Recorded 2026-09-11 with the rate attached, per the ruling that "ingestion
sometimes mis-assigns" and "the merge undoes itself at 20%/day" are not the same
item.*
