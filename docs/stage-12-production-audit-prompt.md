# Stage 12 — Production Smoke Test & Evidence Gate

The Stage 11 deployment is complete and production is now running commit `33539e2`.

Do NOT modify code, configuration, database schema, migrations, model parameters, workflows, or production data.

Do NOT commit, push, deploy, or fix anything automatically.

This stage is **read-only verification only**.

## Frozen production identity

Verify that production is running:

* commit: `33539e2`
* branch: `main`
* model_version: `stage5_baseline_20260807.485823`
* CODE_REVISION: `s5.2`
* paper trading: ON
* real-money betting: OFF

## 1. Verify the first scheduled production run

Inspect the first scheduled run after deployment and determine:

* did the daily picks workflow complete successfully?
* did settlement complete?
* did pick generation complete?
* did the paper trading report complete?
* did the closing-lines workflow run successfully?
* were there any exceptions, retries, timeouts or silent failures?

Do not infer success from workflow existence alone. Inspect actual execution evidence.

## 2. Verify new picks

Identify ONLY picks created after the Stage 11 deployment.

Report:

* number of new picks
* number with `is_paper = true`
* number with `is_paper = false`
* number with `model_version = stage5_baseline_20260807.485823`
* number with NULL `model_version`

The expected result is that all new picks created by the deployed pipeline are paper picks and carry the frozen model version.

If anything differs, STOP and report it.

## 3. Verify PickObservation creation

For every newly created pick, verify:

* exactly one `model` observation
* exactly one `final` observation
* no duplicate `(pick_id, attribution)`
* `taken_odds` is populated
* `taken_at == saved_picks.created_at`
* model market/selection match the frozen model snapshot
* final market/selection match the persisted final selection

Produce a compact reconciliation table:

| Metric                 | Expected | Actual | Status |
| ---------------------- | -------: | -----: | ------ |
| New picks              |       >0 |        |        |
| MODEL observations     |   1/pick |        |        |
| FINAL observations     |   1/pick |        |        |
| Duplicate observations |        0 |        |        |
| NULL model taken_odds  |        0 |        |        |
| NULL final taken_odds  |        0 |        |        |
| taken_at mismatch      |        0 |        |        |

## 4. Verify KEEP and CHANGE behaviour

Determine whether the first production run contains:

* KEEP cases
* CHANGE cases

If KEEP cases exist, verify:

* model and final observations describe the same market/selection
* they share the same taken price
* only one underlying closing observation is required

If CHANGE cases exist, verify:

* model observation retains the original model market
* model observation retains the original model selection
* model observation retains the original model taken_odds
* final observation contains the Claude-selected market/selection
* final taken_odds is independent
* model and final can resolve to different closing markets

If no CHANGE occurred naturally, do NOT manufacture one in production. Use the existing deterministic test suite only to confirm the CHANGE path.

## 5. Verify closing capture

Inspect the first real closing-line capture.

For each newly eligible observation determine:

* considered
* resolved
* missing
* late
* invalid

Verify that MODEL and FINAL use the same closing validity rules.

Verify that a model close is never substituted from the final selection.

Verify that the model observation's market is included in the odds query when MODEL and FINAL differ.

Verify `taken_at` remains the causal boundary and was not moved to review time.

## 6. Verify CLV report

Inspect the first generated `paper_trading_report`.

Verify that:

* MODEL and FINAL are reported separately
* no unavailable MODEL observation is counted as zero CLV
* no FINAL close is substituted for MODEL
* paired CLV uses `final - model`
* fixture clustering remains separate
* checkpoint counters remain separate
* review-action breakdown remains descriptive
* no historical CLV is fabricated

If there are not yet enough valid closing observations for meaningful statistics, report that clearly rather than extrapolating.

## 7. Verify quota safety

Inspect the Odds API ledger after the production run.

Report:

* credits before
* credits consumed
* credits after
* number of Odds API calls
* monthly claimed credits
* remaining budget
* whether the hard 350-credit monthly cap remains respected

Also verify that no unexpected Odds API call occurred outside the intended workflow.

## 8. Historical-data safety

Verify that deployment did NOT:

* backfill `pick_observations`
* alter historical `saved_picks`
* change historical `model_version`
* change historical `is_paper`
* fabricate historical taken prices
* fabricate historical closing lines

The known one-time Stage 7/10 behaviour of marking old closing rows as `late` may occur during the first capture. Measure it and report it separately; do not treat it as a model-data backfill.

## 9. Paper/live isolation

Verify on the actual production execution that:

* new picks are paper-only
* no live-betting execution path was triggered
* paper picks are excluded from ROI/training/calibration paths where required
* `is_paper` remains write-once
* `model_version` remains write-once

## 10. Final evidence table

Produce one final table:

| Area                  | Result | Evidence | Status        |
| --------------------- | ------ | -------- | ------------- |
| Deployment            |        |          | PASS/FAIL     |
| Frozen model identity |        |          | PASS/FAIL     |
| Paper mode            |        |          | PASS/FAIL     |
| Pick observations     |        |          | PASS/FAIL     |
| KEEP attribution      |        |          | PASS/FAIL/N/A |
| CHANGE attribution    |        |          | PASS/FAIL/N/A |
| Closing capture       |        |          | PASS/FAIL     |
| MODEL/FINAL CLV       |        |          | PASS/FAIL/N/A |
| Odds API quota        |        |          | PASS/FAIL     |
| Historical safety     |        |          | PASS/FAIL     |
| Paper/live isolation  |        |          | PASS/FAIL     |

## 11. Hard rules

If any invariant fails:

**STOP. Do not fix it.**

Report:

1. exact failure
2. affected rows/runs
3. likely root cause
4. severity
5. minimal proposed fix
6. whether a migration is required
7. whether the frozen model identity would change
8. whether production should remain running or be paused

If everything passes, declare:

`STAGE 12 — PRODUCTION SMOKE TEST PASSED`

If something fails, declare:

`STAGE 12 — NOT READY`

## Final requirement

This is an evidence-gathering stage, not an implementation stage.

No code changes.
No configuration changes.
No database writes.
No migration.
No commit.
No push.
No deployment.

The purpose is to prove that the deployed Stage 5–11 system behaves correctly under its first real production execution.