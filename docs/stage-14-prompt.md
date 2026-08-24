# Stage 14 — Make the Experiment Measure Something

Stage 13 and 13.1 are closed. Production runs `s5.3` / `stage5_baseline_20260807.098437`, paper mode on, API-Football suspended since 2026-08-19 (OPS-1 open).

This stage has one purpose, stated plainly: **the experiment has never collected a single valid closing line.** Three consecutive paper-trading reports considered ~700 picks each and returned **0 valid CLV pairs**. The system's entire justification — measure whether the model's prices move favourably before the market closes, reach 500 closing lines, then decide about real money — rests on a measurement that has never once happened.

Everything else in this stage exists to make that result trustworthy when it arrives.

---

## 0. Scope

**In scope:**

* MIR-1 — the incremental sync re-admitting excluded matches
* D1 — the closing capture writing zero odds rows
* DEL-1 — alert delivery with no guarantee
* `.claude/commands/daily-ci-audit.md` — the D3 deliverable that was never built
* egress instrumentation, where D1's fix makes it measurable anyway

**Out of scope, and do not drift into them:**

* the predictive core — no models, no ensemble weights, no blend weight, no calibration, no thresholds
* ST14-1, the matcher consolidation — it is a cohort event and it cannot be validated while no fixtures arrive. Design notes only if they fall out naturally; no code.
* the injected-statistic design question — it needs the evidence bar, not this stage
* `paper_trading_mode` stays `true`. The 8 `edge` gates stay off.
* anything blocked on OPS-1: §2's ranking half, §7's identity gate, D2's residual

**Cohort discipline.** This stage should need no `CODE_REVISION` bump. If any change turns out to be prediction- or selection-affecting, stop and report before committing — do not fold it in quietly. MIR-1 is the one to watch (see A3).

---

## Sequence

Build the instrument before the measurement. That is Stage 13's most-repeated lesson and it applies literally here: D1's fix will have to be observed across several runs, and Part A of Stage 13 established that this pipeline can be entirely dead while every audit reads green.

**A → C → D → B.** MIR-1 first because it is active. The delivery guarantee and the audit command next, because they are what will tell you whether D1's fix worked. D1 last, and its demonstration waits for the credit reset.

---

# PART A — MIR-1: the incremental sync re-admits excluded matches

## A1. What is established

`_fetch_incremental` omits the exclusion filter deliberately and decides membership per row afterwards with `if (not r.is_fixture) and r.home_goals is not None` — a hand-written re-implementation of `_base_filter()` that replicates two of three conditions and never consults `training_exclusion_reason`. The column is not even fetched.

The watermark makes it permanent rather than one-off: it is `max(updated_at)` over kept rows, so excluded rows never enter it and their newer timestamps stay ahead forever. All 29 currently have `updated_at` inside the last 24h.

Correctness is held only by `_completed_count()`, which does carry the filter, disagrees with the drifted local count, and forces a full resync. That reconcile is now pinned with its reason in the assertion message.

## A2. The fix

**One evaluation context, not two dialects.** Push the filter into the incremental query using the shared predicate, fetch the column it needs, and delete the post-fetch Python test.

Do not extend the guard to recognise `not r.is_fixture`. That is the reachability answer — the guard grows one instance behind forever, which is the `team_names.py` trajectory this project has already documented.

The watermark repair falls out of the same change: a row excluded in the query is a row that was never returned, so it can neither be re-admitted nor outrun the watermark. Verify that explicitly rather than assuming it.

## A3. Settle the cohort question before committing

Between full resyncs, excluded rows can currently reach Poisson and Elo. Fixing MIR-1 removes that window.

So determine, with evidence: **has a fit ever actually consumed the 29 since they were marked?** Compare resync timing against fit timing in the logs since `s5.3` landed. Then state whether the fix changes what the models learn:

* if no fit ever read a drifted mirror, the fix changes nothing that reached a model — cohort-neutral, no bump
* if one did, the fix is prediction-affecting and needs a `CODE_REVISION` decision before it lands

Do not guess. This is exactly the class of question this project's versioning discipline exists to force.

## A4. After the fix

The reconcile stays. It is now a genuine cross-check between two independent paths rather than the only thing holding, and its pin explains why. Update the pin's message to reflect the new state rather than leaving it describing a defect that no longer exists.

---

# PART C — DEL-1: a delivery guarantee, not another alert

## C1. Why this comes before D1

Two observed failures, different causes, identical class:

| when | run | failure |
| --- | --- | --- |
| 2026-08-11 | 31482430418 | alert built, undelivered — HTTP 400 |
| 2026-08-23 | 32646469497 | `Timed out` — 5s after a Flashscore tier-1 alert |

The first was "fixed" in `451fe3f` by adding `scripts/ci_alert.py` — a second sender to the same last hop. The class survived the remedy. That is what makes it structural.

Every alert this project has — the D3 assertions, the CI failure-alert step, the API-Football suspension notice, and whatever D1's fix will need — terminates in a Telegram send that can fail silently.

## C2. What to build

* **retry with backoff** on the send
* **a surface independent of Telegram**: fail the workflow step so the run goes red, and/or write to the GitHub job summary. Something whose own failure is visible where the run already is.
* **record the send's outcome** so "alert fired" and "alert arrived" stop being one line in a log
* **one path, not three.** `ci_alert.py` and the agent's `_send_message` are the habit. Consolidate to a single delivery function and guard the consolidation — a test that fails when a second sender appears.

Replay both historical failures against the new path and show that each would now surface.

---

# PART D — The daily audit command

`.claude/commands/daily-ci-audit.md` does not exist. It was D3's deliverable, deferred, and then referenced as though it existed.

This is the thing Niki originally asked for: watch the automation every day and fix what breaks.

Build it to do what the Stage 13 Part A pass did by hand:

* every run of all three workflows not already in `docs/ci-audit-ledger.md`
* the full log of every step, not the tail and not only failures — `continue-on-error` means green proves nothing
* **Telegram output as a first-class evidence source**, per Part A's own coverage gap
* a `CLEAN` / `DEGRADED` / `BROKEN` verdict per run, appended to the ledger
* the self-calibrating assertions already designed: a unit that produced data within the last N days producing none

Then prove it: run it against the 08-11 → 08-13 window and confirm it independently reaches the verdicts the manual pass reached. A procedure that cannot reproduce a known result is not ready to be trusted on an unknown one.

---

# PART B — D1: why the closing capture writes nothing

## B1. What is established, so you do not re-derive it

* the workflow runs `scripts/refresh_and_capture`, which calls `scraper.refresh_imminent(...)` at ~line 78. The orphaned `--refresh-odds` flag is in `capture_closing_lines.py`, which production never invokes — a real documentation defect (DOC-1), irrelevant to this cause.
* **credits are claimed** on every degraded run, 2–4 at a time. A request was constructed and sent.
* the logs read `0 games matched, 0 unmatched`. If matching were failing, `unmatched` would be non-zero. **Both zero means the provider returned an empty event list.**
* the closing resolver joins on `match_id`, never on team name, so a wrong fixture cannot pull another match's close. Name matching is eliminated.
* `europe/europa-league → soccer_uefa_europa_league` is mapped correctly at `theodds_scraper.py:74`.

The fault is in request construction, inside `refresh_imminent`.

## B2. Establish two facts before theorising

1. **When did the refresh path last write an odds row — ever?** Query the database for the most recent row attributable to the pre-kickoff refresh, as distinct from pick-time pricing. If the answer is "never since Stage 6," the Stage 6 optimisation broke it on the day it shipped and it has never worked.
2. **Reproduce one request end to end.** Take one specific pick awaiting a close on a degraded day and trace it: league → `sport_key` → the request actually constructed → the provider's verbatim response → where the result was discarded.

## B3. Hypotheses — evidence and a verdict for each

1. **Commence-time window.** The provider returns events inside a time window. Check whether the requested window and the fixture kickoff overlap at all, including timezone handling. This is the first place to look given an empty event list on a valid sport key.
2. **Request parameters.** `regions`, `markets`, any date filter — verify against the provider's current contract, not against what the code assumes.
3. **The Stage 6 narrowing.** Credits are spent only on leagues with an imminent fixture **and** a pick awaiting a close. Verify that predicate selects the leagues it is meant to, and is not excluding precisely the ones needing capture.
4. **The write path.** If events were returned, is the failure in de-vigging validation, the overround gate, or the write itself — and is anything swallowing an exception?
5. **The timestamp.** A row written whose timestamp does not satisfy "observed strictly after `taken_at`" produces the identical symptom: zero valid pairs. Confirm what is stored and what the resolver compares.
6. **Genuine provider coverage.** Last, as the residual after the others are eliminated — not first because it fits.

## B4. Two claims in the README that must be checked, not quoted

The README states `measured: 212 credits/month, worst observed day 46` and that per-league targeting achieves `88%` coverage against `85%` for refreshing everything.

Those figures describe **something**. Establish what. If they came from `scripts/simulate_odds_quota` — a simulation — then the coverage claim was never measured on production, and it is the same class as the ~97% egress reduction that nothing measures.

Report which. If simulated, correct the README the way the "11 filter sites" number was corrected: do not restate it with a fresher estimate, qualify it or remove it.

## B5. State the prediction before you look

Stage 13's most productive moment was writing four falsifiable predictions before the run and having two disproved.

Do the same here. Before running anything against the fixed path, state in writing:

* how many currently pending observations should resolve once the fix is deployed
* how many are structurally outside measurement and why — the README already concedes ~36% (Team Goals, BTTS, Double Chance) are outside the `h2h` + `totals` refresh
* what the first paper-trading report after the fix should show for MODEL and FINAL

Then compare. A disproof is the more valuable outcome and you have set it up so it can be one.

## B6. The credit constraint, and what "fixed" means

The August ledger holds 349/350 spendable — one credit, which cannot buy a league request at 2 credits apiece. **Do not raise the budget.** September 1 resets it, eight days out, and the diagnosis and the fix need no credits at all: they are code, logs and database.

So D1 lands as a fix with a stated prediction, and its demonstration runs on or after September 1.

**"Fixed" is not "the code changed."** D1 is fixed when a paper-trading report shows a non-zero count of valid closing observations with the same-snapshot rule satisfied, MODEL and FINAL resolved independently, and coverage reported honestly as `missing` / `late` / `invalid` / `unavailable` rather than assumed. Until then it is `FIX DEPLOYED — UNDEMONSTRATED`, which is a first-class verdict here exactly as `UNTESTABLE` was in 13.1.

---

## E. Egress instrumentation

D1's work touches the read paths anyway, and MIR-1's fix changes what the incremental sync fetches. Add the measurement while you are there.

The README claims a ~97% egress reduction. Nothing measures it. Either instrument it so the figure is a result, or qualify it as an estimate with its basis stated. Do not restate it with a fresher guess — that was the error the "11 filter sites" correction was written to prevent.

---

## F. Hard rules

1. **Report at each part boundary.** Do not run A → C → D → B as one silent pass.
2. **No fix without a reproduction.** "Not reproduced" is a legitimate finding; a plausible-looking line edited on suspicion is not.
3. **Fail closed, never guess.** Any new decision about identity, validity or coverage refuses and logs when uncertain.
4. **No scope drift into the model.** If a change starts to look like it improves predictions, that is the signal it is out of scope.
5. **No history rewriting.** Nothing restamps `model_version`, `is_paper`, `disposition` or `evidence_status`; nothing backfills observations; nothing edits settled results.
6. **One definition, and a guard.** Every fix in this stage removes a duplicate rather than adding a parallel path. If you find yourself writing a second implementation of something that exists, stop — that is the habit, and it is now instance 8.
7. If any invariant in `tests/test_experiment_invariants.py` fails: **STOP, do not fix, report.**

Declare when A, C and D are complete and B is deployed with its prediction recorded:

`STAGE 14 — INSTRUMENT REBUILT, D1 FIX DEPLOYED — UNDEMONSTRATED`

And after the September demonstration, separately:

`STAGE 14.1 — THE EXPERIMENT MEASURES` or `STAGE 14.1 — STILL NOT MEASURING`, with the prediction compared against the result either way.
