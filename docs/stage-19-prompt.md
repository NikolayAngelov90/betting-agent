# Stage 19 — Fixture Discovery Has Collapsed

Stage 18 is closed: `STAGE 18 — SUBSTRATE BUILT`, with its trigger dates withdrawn on the same day they were derived.

The 2026-08-26 audit established that this system has stopped seeing football. All three fixture sources returned zero for a day carrying six real fixtures — four Champions League, one Conference League, one LaLiga. Real fixtures discovered have gone **13 → 6 → 1 → 0** over roughly two weeks. Champions League was last discovered on 2026-08-19, Conference League on 08-14.

A second defect disguised the first: `_parse_match_date` returns `datetime.now()` as a silent default when a kickoff time will not parse, so unparseable rows enter `matches` stamped with the current moment. Twenty such rows arrived today, all carrying scores, all created *after* the log line saying no fixtures were found. Those phantoms inflated the denominators in the audit's own entries — `110` and `24` fixtures were mostly not fixtures.

The corrected picture is the opposite of what was recorded: **there was never a pick-selection collapse.** Real fixtures against picks ran 13/11, 6/6, 1/1, 0/0. The pipeline converted nearly everything it saw. It stopped seeing.

**Stage 19 is therefore not the momentum research.** There is no point looking for signal in the prices of matches the system cannot find. The substrate accumulates nothing because there is nothing to accumulate — 85 rows, one match, zero keys with three observations.

---

## 0. Scope

**In scope:**

* the `datetime.now()` default in `_parse_match_date`, and the phantom rows it has created
* why Flashscore returns zero for leagues that demonstrably have fixtures
* whether football-data.org's "0 new fixtures added" means *none available* or *none new*
* the audit's blindness to a zero-fixture day
* restoring discovery

**Out of scope:**

* the momentum research. H1, H3 and H4 stay untestable and their dates stay withdrawn until discovery works and the accumulation query — not a calendar — says otherwise.
* the predictive core. No models, weights, blend, calibration, thresholds.
* the deferred backlog: MASK-1's mechanism tests, invariants 2 and 3, MASK-3, L6a, L6b, the review-path findings, the June injury collapse.
* API-Football's reinstatement. The support request is outstanding and outside this stage. **Design the fix so it does not depend on that account returning** — the suspension is precisely the condition the other two sources exist for, and they did not survive it.
* `paper_trading_mode` stays `true`.

**Cohort discipline.** Restoring a broken scraper returns the system to intended behaviour and is not a policy change. But the phantom rows are **in the `matches` table**, and models fit from that table. Removing or correcting them changes what the models learn, which is prediction-affecting. Settle that with evidence before touching a row — the way MIR-1's cohort question was settled — and stop for a decision if it turns out to be a cohort event.

---

## PART A — The phantom rows

Do this first. It disguised everything else and it is still running.

### A1. The defect

`_parse_match_date` falls back to `datetime.now()` when a kickoff will not parse, and does so silently. A row that should have failed loudly instead enters the database asserting it kicks off at the moment it was scraped.

This is the same family as the swallowed `IntegrityError`, the short-circuit that suppresses its own logging, and the safety net that masked its mechanism: **a default that makes a failure look like a success.** Name it as such in the report.

### A2. Quantify before fixing

* how many rows in `matches` carry a `now()`-stamped date, over what period, and by what test you identify them — a heuristic that catches genuine same-moment fixtures is a false positive generator, so state its precision
* the interaction with Stage 17's finding that **53.5% of match rows were created after their own kickoff, mean +14 days**. That was attributed to backfill stamps. Establish whether some of it is this defect instead — the two produce different signatures and were never separated.
* which consumers read those rows: the fitting set, the feature pipeline, the pick eligibility filter, the audit's denominators
* whether any saved pick or observation sits on a phantom row

### A3. The fix

Fail loudly. An unparseable kickoff is a scraper defect and must surface as one — refused, logged with the raw value, counted in the run summary. This project has settled the fail-closed question three times now; apply the same answer.

Then the historical rows: **mark, do not delete.** Use the existing `training_exclusion_reason` mechanism if it fits, and state whether marking them is prediction-affecting. If they are currently inside the fitting set, it is, and that halts this part for a decision.

---

## PART B — Why Flashscore returns zero

This is the failure that matters. API-Football's suspension was supposed to be survivable because two independent sources remain. One of them was asked for `spain/laliga`, `portugal/primeira-liga` and `europe/champions-league` and returned nothing for all three.

### B1. Establish where it breaks

Trace one league end to end, today, and report each step verbatim:

* did Camoufox launch and load the page at all, or fail silently
* what did the page contain — real fixture rows, an empty shell, a consent wall, a bot challenge, a redirect
* did the parser find rows and reject them, or find none
* does the league URL still resolve to what the code expects

Distinguish clearly between **the page had nothing**, **the page had something the parser did not recognise**, and **the page was never fetched**. Those have completely different fixes and the log currently reports all three identically.

### B2. Establish when it broke

Real discovery went 13 → 6 → 1 → 0. That is a decline, not a switch, which argues against a single site change and toward something progressive — degrading selectors, tightening bot detection, an expiring session, or A4's timeouts widening.

Use the cached CI logs. Report the first day each active league last produced a fixture. A4 recorded 12 Flashscore results-page timeouts on 08-23 including four top-five leagues; establish whether this is the same failure spreading from results pages to fixture pages, or a different one.

### B3. Do not assume it is fixable

If Flashscore has become inaccessible to this scraping approach, that is a finding, and the honest response is to say so rather than to iterate on selectors. Report what it would take, and what the system's discovery capability is without it.

---

## PART C — football-data.org

It reported `0 new fixtures added`. Its documented coverage is nine top leagues, which includes LaLiga, and a LaLiga fixture existed today.

Establish whether `0 new` means the API returned nothing, returned fixtures that were all already known, or returned fixtures that were rejected. As with Part B, the log conflates them.

If football-data.org is working and simply covers less than is needed, say what fraction of the 30 active leagues it can supply alone. That number is the system's discovery floor if both other sources stay unavailable — and it is the number that determines whether this project has a data pipeline at all.

---

## PART D — The audit could not see this

`ci_audit.py` flagged today only for the API-Football suspension. A day with zero fixtures analysed, on a day with six real fixtures, was invisible to the mechanical pass. It surfaced because Niki looked at a football calendar.

The self-calibrating assertions designed in Stage 14 were specified for exactly this shape — *a unit that produced data within the last N days produces none.* Fixture discovery going from 13 to 0 is that pattern precisely.

Establish whether the assertion was never implemented for discovery, or was implemented and did not fire. Then add it, and prove it by replaying the last two weeks: the assertion must fire on 08-23, 08-25 and 08-26 and stay silent on days that were genuinely thin.

**And record the deeper point.** The audit measures what the pipeline reports about itself. It has no external reference for "how many matches were actually played today", so a pipeline that sees nothing and says nothing reads as a quiet day. Note what an external check would cost — even a weekly one — and leave it as a proposal, not an implementation.

---

## E. Hard rules

1. **Report at each part boundary.** A, B, C, D are four reports.
2. **No fix without a reproduction.** "Not reproduced" is a legitimate finding.
3. **Fail closed, never guess.** Every new decision in this stage refuses and logs when uncertain, rather than defaulting to a plausible value. That default is what this stage exists to remove.
4. **No number without provenance** — `measured`, `simulated`, `assumed` — with a date.
5. **A single anomalous result is evidence about the measurement first.**
6. **One definition, and a guard.** Watch for THE HABIT; it has appeared five times in the data layer.
7. **Mark, never delete.** Phantom rows are evidence of the defect.
8. **Cohort-neutral or stop.** If correcting the phantom rows changes what any model reads, halt for a decision.
9. If any invariant in `tests/test_experiment_invariants.py` fails: **STOP, do not fix, report.** Invariants 2 and 3 are known defective; the declaration must not cite the invariant count.

Declare:

`STAGE 19 — DISCOVERY RESTORED`, with the measured fixture count over a full day compared against the real card,

or `STAGE 19 — DISCOVERY NOT RESTORABLE` if Parts B and C establish that this system cannot see the fixture list without an account it does not have — in which case say plainly what the project's options are, because a betting agent that cannot find matches has no downstream question worth asking.
