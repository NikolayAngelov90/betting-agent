# STAGE 24 — MERGING THE 32, AND STEP 2's FIRST PRODUCTION PASS

**Registered 2026-09-16, BEFORE the merge is written or run.**

The gate set on 2026-09-13 has opened: the resurrection rate fell to 0 and the
duplicate count held flat at 32 for three days (+0/+0/+0), so these are a
**bounded backlog** and not a still-leaking system.

---

## WHY THIS IS REGISTERED AND NOT JUST DONE

**The inventory that produced this stage found that the mechanism which had
never run was the one that was wrong.** `resolve_team`'s step 2 — the
former-name lookup, the step the stage is named after — **has never fired in
production**: zero `FORMER NAME` lines on 09-14 and 09-15, with DEBUG
demonstrably reaching CI.

The credit gate's first refusal was *deliberately exercised* before it could
fire by accident. Step 2 and the country check were not. **So this merge is
written as a deliberate exercise of step 2, under observation, rather than as a
cleanup that might happen to touch it.**

---

## CORRECTION FOUND WHILE SIZING IT — the merge does NOT arm step 2

**The obvious plan was: merge, `record_former_name()` writes 32 names, the next
scrape hits the lookup. Measuring it first showed that is wrong in two ways.**

| | |
| --- | --- |
| removed names **already** in `team_former_names`, already pointing at the correct survivor | **29** |
| removed names already mapped to a DIFFERENT team (a conflict) | **0** |
| removed names genuinely **new** to the table | **3** |
| of those 3, pairs where survivor and duplicate share an **identical** name, so no name is removed at all | **2** (`PEC Zwolle`, `Fortuna Sittard`) |
| **rows the merge will actually write** | **1** — `Vitória SC` -> 601 `Guimaraes` |

> **The backfill already recorded 29 of these 30 names on 2026-09-13.** Step 2 is
> ALREADY armed for them. The merge does not create the interception; it was
> there before the merge was considered.

**And `flashscore`, `footballdataorg` and `historical_loader` all call
`resolve_team()` with NO provider id**, so for them step 1 is skipped and **step
2 is the first lookup performed**. Nothing about a duplicate row prevents it
firing. A name in the table resolves to the survivor today.

**A former name must be a name that is no longer current.** For the two
exact-name pairs the name still lives on the survivor, so recording it would put
a CURRENT name in a table of removed ones and let step 2 short-circuit step 3
for no reason. The merge records a former name **only where the merged-away name
actually differs from the survivor's**.

---

## SO WHAT DOES THE MERGE ACTUALLY BUY

**1. It removes a live non-determinism in step 1.** Two rows share a provider
id, and step 1 is `.filter(apifootball_team_id == provider_id).first()` — **with
no ordering**. Which row an API-Football fixture attaches to today is whatever
the database returns first. That is not a latent risk; it is a coin flip on
every AF resolution for 32 clubs.

**2. It re-points live references.** 19 of the 32 duplicates carry real
references — `1561/1775 Erzurumspor` alone holds 46 matches, 1 pick and 91 odds
rows on the wrong side. Split history is what makes a duplicate expensive.

**3. It writes the one genuinely missing former name.**

---

## THE PREDICTION, fixed before the run

### A. What the merge itself should do

| | predicted |
| --- | --- |
| components merged | **32** (all size 2) |
| team rows removed | **32** |
| `team_former_names` rows written | **1** (`Vitória SC`) — NOT 32 |
| `team_former_names` total after | **125** (from 124) |
| duplicates whose references must be re-pointed | **19** |
| shared-provider-id components remaining after | **0** |
| merges refused for ambiguity | **0** — every component is exactly 2 rows with one provider id |

**If more than 1 former name is written, the merge is recording names that are
still current** and the "record what you remove" rule has been applied to
something that was not removed.

### B. What step 2 should do afterwards — THE ACTUAL EXERCISE

**Step 2's first hit does NOT come from the merge.** It comes from the next
scrape of any of the 124 recorded names by a path that passes no provider id.

> **PREDICTION: on the next `daily-picks` run that CREATES 50 or more fixtures,
> step 2 fires at least once, and at the pre-s5.12 resurrection rate
> (~0.16 per created fixture) roughly 8-14 times on a full ~88-fixture card.**

**Why that rate:** every pre-s5.12 resurrection was an exact match to a
merged-away name. The same scrape that used to create a row must now be
intercepted by the lookup. **The old creation rate IS the predicted hit rate** —
if the two do not match, one of the two measurements is wrong.

**The log line to look for**, at DEBUG, confirmed to reach CI logs:

```
TEAM_RESOLVE name='Lens' step=former_name team=576 resolved='Racing Club de Lens' league='france/ligue-1'
```

**AMENDED 2026-09-16, BEFORE the card it measures.** The original registration
quoted step 2's own prose line, which announced only step 2 — so a run could
report zero interceptions and zero creations and leave no way to tell whether
anything was attempted. `resolve_team` now emits ONE structured record at every
one of its five steps, carrying the INCOMING name beside the resolved row, and
`ci_audit` prints the split as `resolve[provider_id=N former_name=N exact_name=N
strict=N create=N]` next to `disc[...]`. The prediction is unchanged; what
changed is that it can now be checked, and the amendment landed while the
2026-09-16 run had still not fired.

### C. What each outcome means, fixed in advance

| outcome | reading |
| --- | --- |
| **hits ≈ the old creation rate** | the model is right end to end: the scrapes that created rows are the scrapes the lookup now absorbs |
| **hits = 0 on a full card** | **step 2 is not reachable in production.** The unit tests pass and the branch is dead — exactly the shape the credit-gate exercise existed to rule out, and the most valuable possible result |
| **hits ≫ the old creation rate** | the lookup is intercepting names that were resolving correctly before, i.e. it is shadowing step 3 and the table contains names that are still current |

### D. THE HONEST LIMIT ON THE 09-14/09-15 RESULT

**Both the creations AND the step-2 hits were zero, and under the model above
they cannot both be zero on the same exposure.** At ~0.16 per created fixture,
the 32 fixtures created over those two days predict ~5 events, whichever side of
the fix they land on.

**So the exposure model is not established.** The likelier explanation is that
those two small cards simply contained none of the clubs with duplicate history
— which the measurement supports: **0 of the 64 team rows those fixtures
referenced carries a name in `team_former_names`.**

**The counterfactual cannot be reconstructed**, because the scraped NAME is
never stored — only the row it resolved to. So "what would the old code have
done with this card" is unanswerable after the fact, and the p = 0.004 on the
creation drop rests on fixtures-created being a fair exposure measure, which
this is the evidence against.

> **That is why prediction B is stated on the NEXT full card and not claimed
> from the two days already observed.**

---

## OUT OF SCOPE, stated so the result is not read as whole

* ~~**No resolution-path instrument.**~~ **CLOSED the same day, before the next
  card.** `resolve_team` now emits `TEAM_RESOLVE name=… step=… team=…` at all
  five steps. The 09-14/09-15 attribution had to be argued from `league IS NULL`
  in the data because only step 2 announced itself; that argument will not be
  needed again.
* **The country check** stays in the same never-fired state. It refuses 0 of 25
  possible joins, so there is nothing to exercise; it is armed and idle by
  design, and that is recorded rather than resolved.
* **The exact-name collapse** — still unsized; its remedy is splitting rows.
* **Cohort** — a merge changes which rows predictions attach to. s5.12 already
  carries 30 picks, so this takes a **bump**, not an amend.

*Registered 2026-09-16, before the code.*
