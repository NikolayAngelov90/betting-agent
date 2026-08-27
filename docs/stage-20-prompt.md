# Stage 20 — The Discovery Loop Wastes What It Has and Refuses What It Shouldn't

Stage 19 is closed: `STAGE 19 — DISCOVERY RESTORED`. Flashscore's own count for `spain/laliga` was 2, the fail-closed parser produced 139 loud refusals where yesterday it produced silent phantoms, and 105 matches were created with zero sub-minute kickoffs.

The same run exposed three defects in the loop that now works. All three cost coverage on the day coverage was restored.

**Cohort note, read before committing anything.** The Stage 19 run created picks at `s5.6` / `60caed`, so the amend-while-empty window has closed. Run `scripts/cohort_status.py` before any prediction- or selection-affecting change — it will say `BUMP`. Anything in Part A that changes which fixtures resolve is selection-affecting and needs `s5.7`. Group them so there is one bump, not three.

---

## PART A — The three identity-gate refusals

Niki spotted these in the log and they are the priority. The gate fired three times, each refusal skipping a fixture entirely.

A fail-closed gate's first live firing is precisely when its refusals must be inspected rather than trusted. The two-way overround band met a 20-of-20 replay standard before shipping; this gate shipped `correct by construction, unverified in production` and has now run for the first time.

### A1. Identify each refusal

The gate logs at two sites — `apifootball_scraper.py:1514` for the country check and `:1524` for the lexical anchor. For each of the three, report verbatim:

* the incoming name, the stored name, the stored row's id and its country
* the competition and date of the refused fixture
* which of the two checks refused it
* **whether the refusal was correct** — the same club under two names, or two genuinely different clubs

### A2. Classify, then fix only the false positives

**If a refusal was correct**, the gate did its job and the fixture was rightly skipped. Record it as a success and leave it.

**If a refusal was a false positive**, a legitimate fixture was lost. Fix it through `NAME_ALIASES` in `src/utils/team_names.py` — the curated-alias supplement §B4 explicitly permits, and explicitly forbids as a *mechanism*. Do not loosen the anchor rule, do not add a similarity ratio, do not widen the country check. The gate's fail-closed shape is correct; what it lacks is knowledge of specific names.

The residual class the anchor test cannot cover is a legitimate pair with **zero shared tokens** — a nickname or a short form that shares no anchor with the full name. That is exactly what a curated alias is for, and it is why the alias table exists rather than a threshold.

### A3. Establish the size of the problem, not just these three

Three refusals on one day is a rate, not an incident. Report:

* how many fixtures the gate has refused since it shipped, per day
* whether the refusals cluster on particular sources, competitions or leagues
* whether any refused pair recurs — a recurring false positive is a permanent blind spot on that fixture, every day

If the recurring set is non-trivial, the alias table needs those entries whether or not they appeared in this run.

### A4. Prove the fix the way the gate was proved

Replay every alias added against the historical name pairs: each must now resolve, and **the Telstar / Maccabi Tel Aviv impostor must still be refused**, along with the Rapid Vienna / Rapid București and Pau FC / St. Pauli cases the country check closed. An alias that admits an impostor is worse than the false positive it fixed.

---

## PART B — The budget is spent on leagues that return nothing

23 of 30 leagues were attempted before `_FIXTURES_BUDGET_S` exhausted at 301 seconds. The never-attempted set includes `europe/europa-league` and `europe/europa-conference-league` — **which carried 34 of the day's 36 fixtures.**

The system discovered 94% of the day's football only because API-Football happened to be restored, from two competitions its own fixture loop never reached.

### B1. Measure the cost before touching the budget

UCL alone consumed **90 seconds of 300 on a double timeout that produced nothing.** Thirty per cent of the budget, for zero rows.

Raising `_FIXTURES_BUDGET_S` is the obvious move and it treats the symptom. Measure first:

* the per-league timeout and retry policy on the fixtures path — why does a league returning nothing cost 90s rather than failing fast
* the distribution of per-league durations across the run: how much total time went to leagues that returned zero
* how many additional leagues a shorter timeout would reach, derived from the measured durations
* only then, whether a budget increase or rotation is needed for the residual

Your registered decision rule offered rotation. The evidence points at cost first: rotation changes *which* leagues starve, it does not recover wasted time.

### B2. The UCL zero belongs to this investigation

The open three-source-zero entry and the 90-second timeout are the same phenomenon. Under camoufox the UCL fixtures page yielded **zero rows of any kind**, against 110–120 for every domestic league — an empty date still renders a table, so zero rows of any kind means the page is not the shape the parser expects.

Establish what the UCL fixtures URL actually returns. A competition that renders differently between phases, or lives at a different path, explains both the timeout and the zero, and it would explain them for UEL and UECL too — which is where 34 of 36 fixtures were.

Do not iterate on selectors before looking at the page. That discipline found the `--static` rename.

### B3. Ordering is not neutral

Whatever the fix, the truncation order determines which leagues are permanently unseen, and it is currently config order. Record the trailing set by name after any change, and state whether it is stable — a league that is never scraped produces no fixtures, which never changes its position, and that circularity has already been fixed once in this pipeline.

---

## PART C — `fixtures_zero_active` fires 21 times on a normal day

All 21 were false positives: the check flags leagues with no fixtures inside `max_days_ahead=1`, and both the check and the scraper's underlying warning ignore that window.

You named the consequence exactly — this is how a check gets ignored. It is also DEL-1's lesson in a third form: **an alert that never arrives and an alert that always fires carry the same amount of information, which is none.** And it is *zero is only an anomaly relative to what was asked for*, in a check written after that rule was recorded.

Fix it so the check asks the question the system actually asked: a league with no fixtures **inside the window it was queried for** is not an anomaly. Then replay it across the cached period and report how often it fires — if the answer is still routinely non-zero, it is not finished.

---

## D. Hard rules

1. **Report at each part boundary.**
2. **No fix without a reproduction.** "Not reproduced" is a legitimate finding, and Part B's page inspection must precede any selector work.
3. **Fail closed, never guess.** Part A fixes knowledge, not tolerance. No thresholds, no ratios, no widened bands.
4. **No number without provenance** — `measured`, `simulated`, `assumed` — with a date.
5. **A single anomalous result is evidence about the measurement first**, including about your own tooling.
6. **One definition, and a guard.** THE HABIT has appeared five times in the data layer; do not add a sixth.
7. **Cohort:** run `cohort_status.py` before committing. `s5.6` now carries picks, so selection-affecting changes bump to `s5.7`. One bump covering all of them, and record the measured effect on the discovered-fixture population in the history entry.
8. If any invariant in `tests/test_experiment_invariants.py` fails: **STOP, do not fix, report.** Invariants 2 and 3 are known defective; the declaration must not cite the invariant count.

Declare `STAGE 20 — LOOP REPAIRED` with, for the following day's run: the fixture count per league, the trailing set by name, the identity gate's refusal count with each classified, and how many times `fixtures_zero_active` fired.

If Part B establishes that UEFA competitions cannot be scraped from Flashscore at all, say so plainly — then discovery for 34 of 36 fixtures depends entirely on an API-Football account the provider's own rules place at risk, and that dependency belongs in front of Niki rather than inside a fix.
