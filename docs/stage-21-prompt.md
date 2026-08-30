# Stage 21 — Fix the Regression, Then Move the Clock

Two things, both operational, both decided rather than explored. The research stage that follows this one is H1/H4 — the substrate trigger fired at 91 keys with three or more observations, and price momentum is testable for the first time in this project's life. Do not start it here.

---

## PART A — The Standard Liege regression. First, before anything else.

Stage 20 fixed two identity-gate false positives and created a third. `Standard Liege` / `St. Liege` share an anchor and are refused anyway, because canonicalisation rewrites via `TEAM_NAME_ALIASES["Standard Liege"] = "Standard"`, leaving `{stan, standard}` against `{lieg, liege}`.

Your diagnosis is exact and the fix you named is right: **union rather than replace.**

**Record the rule, because it is larger than this bug.** A one-directional map — many provider forms to one canonical form — applied to **both sides of a symmetric comparison** can delete the very token the comparison depends on. Canonicalisation is not symmetric-safe. State it in the guard-design notes, and check whether any other normalisation in this codebase is applied symmetrically: `_norm`, `_tokens`, the market taxonomies, the competition map. Report what you find; fix only this one.

**Verify the fix the way the gate was verified.** Every previously-passing pair must still pass, and every impostor must still be refused — Telstar/Maccabi, Rapid Vienna/Rapid București, Pau/St. Pauli, Cracovia/Rakow. A fix that admits an impostor is worse than the false positive it removes.

While you are there: **row 411 (Rakow) carries Cracovia's AF id 350.** That is the fifth confirmed identity corruption and the gate found it by refusing a real fixture. Record it. Do not repair it — the population is still unestablished and a repair pass is its own decision.

---

## PART B — The schedule. Measure two things, then move it.

The decision is made: **the cron moves.** What is not decided is where to, and that must come from measurement rather than from instinct.

### B1. The weekend asymmetry is the case, and it is already in your data

Your Part B table averaged across days and lost the shape:

| day | delay | in-window | already kicked off | picks |
|---|---|---|---|---|
| 08-27 (Thu) | 10h21m | 36 | 36 of 36 | 0 |
| 08-28 (Fri) | 11h21m | 29 | 29 of 29 | 0 |
| **08-29 (Sat)** | **5h03m** | **103** | **57 of 103** | 35 |

Saturday carries three times the card and it kicks off in the afternoon. A five-hour delay — well inside the historical envelope, not an episode — cost 55% of it. Weekday evening cards tolerate lateness; weekend daytime cards do not.

Quantify it properly across the cached period: fixture count and kickoff-time distribution by day of week, and the fraction of each day's card that would already have kicked off at 1h, 3h, 5h, 8h and 11h of delay. That table is the argument, and it decides how much margin the new cron needs.

### B2. How early can it run? `first_seen_at` now answers this

Moving the cron to the small hours buys margin against the delay and costs odds availability — bookmakers post prices progressively, and a run twelve hours before kickoff may see fixtures with no market.

That was unmeasurable before Stage 18. It is measurable now: `first_seen_at` is populated on 2,628 rows and rising.

Report the distribution of **first-seen time relative to kickoff**, per market and per league tier, and specifically: what fraction of fixtures have a priced `h2h` market at 24h, 18h, 12h, 9h and 6h before kickoff. The earliest cron that keeps odds coverage acceptable is the answer, and "acceptable" must be stated as a number before you look at it.

### B3. Then choose, with the arithmetic shown

Constraints to satisfy simultaneously:

* **settlement safety** — the run settles the previous day's matches first, so it must start after the last of them has finished. Establish the latest kickoff in the active league set and add match duration; a 23:39 UTC cron fails this, an 03:00 UTC one probably does not. Verify rather than assume.
* **weekend coverage** — from B1, the delay margin needed to still reach a Saturday afternoon card.
* **odds availability** — from B2.
* **pick lead time** — you measured that lateness *compresses* lead to 2.1h against 4.4–8.8h on time. An earlier cron extends it, which is the one effect that helps the MODEL series. Report the projected lead distribution at the chosen time.

If the three constraints cannot be satisfied together, say so and report which one binds. That is a finding, not a failure — and it would mean the pipeline needs splitting rather than shifting, with settlement on its own later schedule.

### B4. Cohort

Changing when prices are taken changes every pick's taken price and lead time, and changes which fixtures are inside the window at all. By the `s5.2` precedent — *a changed population of predictions is a different experiment* — this is selection-affecting.

Run `cohort_status.py`. Expect `BUMP`. Record the measured effect on lead time and on the discovered-fixture population in the history entry, as `s5.3` did with the 1.6% cap figure, so nobody later attributes a cohort difference to the clock.

### B5. What this does not fix

State it plainly in the record: **moving the cron buys margin, it does not remove the dependency.** GitHub's scheduler has produced 0.5h to 11h21m of delay and nothing here changes that. A cron at 03:00 UTC delayed 11h lands at 14:00 UTC and still misses a Saturday noon kickoff.

So keep OPS-3 open with its new escalation criterion, and record the residual: the schedule is now robust to the *typical* delay and not to the *observed maximum*.

---

## PART C — Correct one framing in the record

Your Part C called the capture collapse "permanent loss of the experiment's only instrument" — 1 capture from 69 picks against 61 from ~116.

That is severe by its own measure, and Stage 16 established that measure no longer decides anything: 500 was over-specified ~29×, seventeen observations suffice to exclude a decision-relevant effect, and at n=46 the one-sided upper bound is +0.107% against a +1.85% threshold. Lost captures buy precision on an axis that is already resolved — the same finding Stage 15 reached about the seven months to March 2027.

The CLV instrument retains **forward** value: if a model with a plausible edge is ever built, this is how it would be tested. That is a reason to keep it working, not a reason to treat each lost observation as urgent.

Amend the entry so the severity is stated against the right question. It also strengthens your own recommendation rather than weakening it — the small-sample argument was one reason not to panic; that the loss is cheap is a second.

---

## D. Hard rules

1. **Report at each part boundary.** A and B are two reports.
2. **No number without provenance** — `measured`, `simulated`, `assumed` — with a date.
3. **State "acceptable odds coverage" as a number before measuring it**, not after. B2 is otherwise a threshold fitted to its own sample.
4. **A single anomalous result is evidence about the measurement first.**
5. **Fail closed, never guess.** Part A fixes knowledge, not tolerance — no ratios, no widened bands.
6. **One definition, and a guard.** The symmetric-canonicalisation rule joins THE HABIT's family; record it, do not build a guard for it in this stage.
7. **Cohort:** `cohort_status.py` before committing. One bump covering Parts A and B together.
8. If any invariant in `tests/test_experiment_invariants.py` fails: **STOP, do not fix, report.** Invariants 2 and 3 are known defective; the declaration must not cite the invariant count.

Declare `STAGE 21 — CLOCK MOVED` with the chosen cron, the three constraints it satisfies, the measured lead-time change, and the day-of-week table that justified it.

If B2 shows odds are not available early enough to move the cron meaningfully, declare `STAGE 21 — CLOCK CANNOT MOVE` and say what that implies — because then the weekend card is unreachable by scheduling alone, and the answer lies in splitting the pipeline rather than shifting it.
