# Stage 13 — Response to Part A, and the revised order for the rest

Part A is accepted. The ledger is good work and the A1 diagnosis is exactly the kind of finding this stage existed to produce — a self-inflicted defect, reproduced in isolation, with production impact measured rather than estimated.

Three corrections follow, then the order to execute.

---

## 1. A3 is misclassified. Elevate it.

You classified A3 as MEDIUM with the root cause *"provider has no data for UEFA qualifiers (Stage 12.2)"* and proposed only to stop re-paying for the discovery.

The ledger does not support that root cause. It records three consecutive paper-trading reports:

* 2026-08-11 — 700 picks considered, **0 valid CLV pairs**
* 2026-08-12 — 691 picks considered, **0 valid CLV pairs**
* 2026-08-13 — 726 picks considered, **0 valid CLV pairs**

UEFA qualifiers are a small minority of the fixture set. They cannot produce zero across ~700 picks spanning 30 leagues. And the mechanism is worse than a coverage gap: the same-snapshot rule requires a closing observation from odds observed **strictly after** the pick was taken. If the pre-kickoff refresh writes **0 odds rows**, then no closing line can ever resolve **for any pick in any league** — not because coverage is thin, but because nothing is ever written for the resolver to find.

That is not a degraded experiment. It is an experiment that measures nothing, and it has been in that state for the entire life of the deployment.

**A3 is CRITICAL and becomes Part D1 below.**

One detail in your own findings points at the answer and you passed over it: the capture logs read `0 games matched, 0 unmatched`. If matching were failing, `unmatched` would be non-zero. Both being zero means **the provider returned an empty event list** — the request itself found nothing. That relocates the fault from name matching to request construction, and it is testable directly.

## 2. Do not re-run the six discarded reviews

Matches 49423, 49458, 49460, 49468, 49485, 49486 lost their verdict and now read `review_action = none`. Leave them.

`none` is the truthful record of what happened: no verdict bound. Re-running the review today would produce a decision made with information that did not exist at pick time, stamped onto a pick taken at a different price. That is a fabricated observation, and the KEEP/CHANGE breakdown in the paper-trading report would silently absorb it.

Record the six in the Stage 13 report as a known gap with its cause. Do not backfill.

## 3. One cohort break for the whole stage — not one per fix

This is the structural instruction for everything that follows, and it changes how you sequence the work.

Three of the remaining items alter what the system persists or predicts:

| Item | Effect | Cohort-relevant? |
| --- | --- | --- |
| **B** — refuse fixtures whose identity is unresolved | changes which fixtures produce picks | **yes** — selection-affecting |
| **C** — one pick per match | changes which picks are persisted | **yes** — selection-affecting |
| **D2** — restoring injury features (*if* restorable) | changes feature values fed to the model | **yes** — prediction-affecting |
| **A1** — cascade fix | restores a decision path; changes no prediction and no selection | no |
| **D1** — closing capture | measurement side only | no |
| **D3** — alerting | observability only | no |

If these land as separate commits over separate days, you get three cohort resets and no cohort ever accumulates enough observations to be worth anything.

**Therefore: B, C and D2 land inside a single `CODE_REVISION` bump to `s5.3`, documented as one entry in the `model_version.py` history block covering all three.** A1, D1 and D3 land outside it and must not move the fingerprint. If D2 turns out to be unfixable (see below), say so and the break covers B and C only.

If you discover a fourth prediction- or selection-affecting change while working, stop and report it before committing — it either joins the same break or waits for Stage 14.

---

## 4. The cohort reset is approved, and now is the only cheap moment

§C5 required me to state the cost of resetting the 500-closing-line checkpoint. Part A answered its own question: **the counter is already zero.** 124 observations exist and not one has a resolved closing line. There is nothing to preserve.

The reverse now matters more. Once D1 restores capture and closing lines start resolving, every subsequent cohort-breaking change destroys real accumulated data. So the correct sequence is: **break the cohort first, fix the measurement second.** That is why C comes before D1 in the order below, even though D1 is the more severe defect.

Proceed with `s5.3`. Existing picks keep their old version. Nothing is restamped.

---

## 5. Execution order

### Step 1 — A1 + A6 (no cohort effect, commit alone)

Fix the cascade. The `pick = relationship("SavedPick", backref="observations")` added in Stage 10 carries SQLAlchemy's default cascade, so `session.delete(parent)` issues `UPDATE pick_observations SET pick_id = NULL` and the DB's `ON DELETE CASCADE` never fires against a `NOT NULL` column.

Requirements:

* the ORM must delete children, or must defer to the database and let `ON DELETE CASCADE` do it — choose one deliberately and say in the code comment which, and why. Do not configure both halves inconsistently.
* `pick_id` stays `NOT NULL`. The constraint is correct; the cascade was wrong.
* **A6 is part of this fix, not a separate item.** `tests/test_pick_observations.py` exercises the cascade with `session.execute(SavedPick.__table__.delete())`, which bypasses the ORM entirely and therefore tested the database constraint while the ORM path was broken. That is why this shipped. Rewrite it to use `session.delete(pick_object)` — the path production actually uses — and confirm it fails against the pre-fix code.
* audit the rest of the suite for the same shape: any test that reaches the DB through `Table.delete()`, `Table.insert()` or raw SQL where production goes through the ORM. Report how many you found. A test that cannot fail is worse than no test, because it is counted in the 651.
* verify no other `relationship(...)` in `src/data/models.py` has the same defect.

This is a prerequisite for Part C, not a parallel task: §C4.5 asks you to keep the consolidation branch as a defensive assertion, and you cannot reason about the behaviour of a branch that has never once executed successfully.

### Step 2 — Part B (CSKA wrong fixture)

As written in `docs/stage-13-prompt.md` §B, unchanged. Do not commit yet — B and C share the cohort break.

### Step 3 — Part C (one pick per match)

As written in `docs/stage-13-prompt.md` §C, with §C5 resolved: the reset is approved, and the `s5.3` history entry must cover B, C and D2 together rather than C alone.

Commit B and C together with the `CODE_REVISION` bump.

### Step 4 — D1: why the closing capture writes nothing (CRITICAL)

Establish facts before proposing a fix.

* **When did the refresh path last write an odds row?** Query the DB for the most recent row attributable to the pre-kickoff refresh. If the answer is "never since Stage 6", the Stage 6 optimisation broke it and the measured "212 credits at 88% coverage" figure in the README describes a system that no longer exists.
* **Reproduce one request end to end.** Take one specific pick that was awaiting a close on 2026-08-13, and trace: which league it belongs to, which provider `sport_key` that league maps to, what request was constructed, what the provider returned verbatim, and where the result was discarded.

Hypotheses to test, each with evidence and a verdict:

1. **Empty event list from the provider.** `0 matched / 0 unmatched` says the response contained no events. Check the `sport_key` mapping for the leagues involved, and whether it is missing, stale, or defaulted.
2. **Commence-time window.** The provider returns events inside a time window; check whether the requested window and the fixture kickoff actually overlap, including timezone handling.
3. **Request parameters.** `regions`, `markets`, and any date filter — verify against the provider's current contract, not against what the code assumes.
4. **The Stage 6 narrowing.** Credits are spent only on leagues with an imminent fixture **and** a pick awaiting a close. Verify that predicate selects the leagues it is supposed to, and is not excluding exactly the ones that need capture.
5. **Write path.** If events *were* returned, confirm whether the failure is in matching, in de-vigging validation, or in the write itself — and whether anything is swallowing an exception.
6. **Genuine provider coverage gap**, per Stage 12.2. This is the hypothesis you already accepted; it is last on the list because it must be the residual explanation after the others are eliminated, not the first one that fits.

Then fix the cause. If the residual really is a coverage gap for a subset of competitions, implement the no-re-pay cache you proposed — but only for that subset, and it must expire rather than being permanent, because provider coverage changes.

Report, honestly: after the fix, how many of the currently pending observations can resolve, and how many are permanently outside measurement. If the answer is still near zero, say so plainly. An experiment that cannot measure its subject should be declared broken, not reported as awaiting data.

D1 does not touch the cohort.

### Step 5 — D2: injuries (was A2)

`Injury update: saved 0 injuries from 30 fixtures`, every daily run, no error raised.

Determine first whether this is **fixable at all**. The README states API-Football's free tier covers seasons 2022–2024. If the 2026/27 season is simply not served on the current plan, injuries are structurally unavailable and no amount of code will produce them.

Establish:

* the last date any injury row was written
* what the provider actually returns for one of the 30 fixtures — verbatim response, not a summary
* whether the request is rejected, returns empty, or returns data the parser drops
* whether daily API-Football quota (100 req/day) is exhausted before the injury step runs

**Then the integrity question, which matters more than the fix.** When injuries are absent, what do the injury features receive? If they default to `0.0` and are fed to the model as though they were measured, the model has been trained and predicting on fabricated values — silently, and possibly for a long time. Report which feature sections do this and how many. Absent must be representable as absent.

Two outcomes:

* **Restorable** — fix it. This changes feature values and therefore predictions, so it lands in the **same `s5.3` break** as B and C. Reorder your commits accordingly rather than taking a second reset.
* **Not restorable** — do not silently keep sending zeros. Make the absence explicit and alerted, document it in the README as a known structural gap with its cause, and leave the model's inputs unchanged so the cohort is unaffected.

State which outcome applies before you write code.

### Step 6 — D3: alerting and the daily procedure (§A4)

Implement `.claude/commands/daily-ci-audit.md` and the post-run assertions. Part A supplies the exact conditions that must fire, because all of them occurred and none alerted:

* a briefing decision was discarded / an exception was swallowed inside the review loop
* injuries saved 0 from N > 0 fixtures
* the closing refresh claimed credits and wrote 0 odds rows
* a paper-trading report produced 0 valid CLV pairs from > 0 considered picks
* `--picks` saved 0 picks while fixtures were available
* a scraper returned 0 rows for all leagues
* `pick_observations` written ≠ 2 × picks saved
* the test step reported any failure

The headline number from Part A — **27 runs, 0 flagged, every one green** — is the specification for this step. If the assertions you write would not have caught all six anomalies, they are not finished. Demonstrate that by replaying them against the saved logs.

---

## 6. What remains deferred

A4 (Flashscore/Camoufox timeouts, 12 results pages including four top-5 leagues) stays deferred as you proposed — but record it in the ledger as an open item with a date, not as a closed one. If it recurs in the next audit, it is promoted.

---

## 7. Hard rules, unchanged and extended

1. Report at each step boundary. Do not run steps 1–6 as one silent pass.
2. No fix without a reproduction.
3. Fail closed, never guess.
4. No scope drift into the predictive core. Restoring an input that was supposed to exist is not tuning; changing an ensemble weight is.
5. No history rewriting. No re-running the six discarded reviews.
6. **One cohort break for the stage.** If a change would move `model_version` and is not B, C or D2, stop and report before committing.
7. If any invariant in `tests/test_experiment_invariants.py` fails: STOP, do not fix, report.

Final declaration when steps 1–6 are complete and green:

`STAGE 13 — DAILY OPERATIONS AUDIT COMPLETE`

If D1 concludes that the experiment cannot measure its subject:

`STAGE 13 — EXPERIMENT NOT MEASURING` — and stop before D2.
