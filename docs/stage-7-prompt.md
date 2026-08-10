# Stage 7 — Activate the Prospective Paper-Trading & CLV Experiment

## Role

You are the senior engineer responsible for the next controlled stage of the `betting-agent` project.

You have access to the full repository and the previous Stage 1–6 audit reports.

**Do not start by modifying code.**

First inspect the current repository state, Git branch, recent commits, configuration, CI workflows, database schema, odds ingestion, paper-trading flow, closing-line capture, model-versioning, and all tests.

The objective of this stage is **not to improve the prediction model**.

The objective is to make the existing Stage-5 frozen system operationally capable of collecting **clean, prospective, reproducible evidence** about:

1. bookmaker-market pricing,
2. model-vs-market performance,
3. closing-line value (CLV),
4. prediction calibration,
5. paper-trading outcomes,

while staying within the free Odds API quota and avoiding unnecessary Supabase egress.

---

# 1. Current known state

The previous stages established:

* The system should remain **PAPER TRADING ONLY**.
* The current model is frozen as:

`stage5_baseline_20260807.ac04cc`

* `bookmaker_blend_weight` remains at 0.80.
* The six previously identified betting gates remain disabled.
* `min_expected_value` and `min_confidence` remain unchanged.
* No new model/features/threshold optimisation is authorised in this stage.
* Genuine CLV coverage is currently **0%** because the prospective experiment has not yet collected valid closing lines.
* Stage 5 migrations are already applied to production.
* Production currently has:

  * 1,026 total saved picks
  * 1,018 settled
  * 0 valid closing lines
* Stage 5 created the CLV and paper-trading infrastructure.
* Stage 6 created and tested the Odds API quota/freshness optimisation.
* Stage 6 corrected the previous incorrect estimate of API usage:

  * current historical replay ≈ 434 credits/month
  * proposed selected refresh strategy ≈ 258 credits/month
  * 90% historical coverage under the chosen parameters
* Stage 6 empirically confirmed:

  * `eu + h2h,totals` = 2 credits/request
  * `eu + h2h` = 1 credit/request
* Both h2h and totals are required by the existing settled data.
* Stage 6 did **not** modify the model.
* Stage 6 currently has 511 tests passing.
* Stage 6 is uncommitted.
* Stage 6 identified an important operational issue:

**The database currently contains zero future fixtures, while the live API does contain today's fixtures.**

Therefore the new odds-refresh pipeline currently selects zero leagues until fixture ingestion catches up.

---

# 2. Primary objective

Make the system capable of running the following controlled loop:

```text
Fixture ingestion
       ↓
Upcoming fixtures available in DB
       ↓
Generate paper predictions
       ↓
Save immutable model snapshot
       ↓
Select only relevant/imminent fixtures
       ↓
Refresh odds within quota
       ↓
Capture closing prices
       ↓
Validate market/bookmaker/selection
       ↓
Calculate CLV
       ↓
Settle paper picks
       ↓
Generate evaluation report
       ↓
Health checks
       ↓
100 / 200 / 500-pick checkpoints
```

The system must fail safely at every stage.

---

# 3. IMPORTANT: Do not change the model

This is the most important constraint.

Do NOT:

* retune Elo,
* retune Poisson,
* change half-life,
* change rho,
* change Bayesian weighting,
* change bookmaker blend,
* add new ML models,
* add new features,
* change EV thresholds,
* re-enable disabled gates,
* optimise selection thresholds,
* introduce new betting markets,
* modify the prediction formula,
* modify model-version tracking,
* alter the existing frozen model to improve ROI.

If you discover a potential model improvement, **document it separately** under:

`docs/stage7-model-observations-YYYY-MM-DD.md`

Do not implement it.

The purpose of Stage 7 is to measure the frozen model prospectively, not optimise it.

---

# 4. First: audit Stage 6 before changing anything

Read:

* Stage 5 report
* Stage 6 report
* Stage 6 prompt
* current `git diff`
* current branch
* recent commits
* all changed Stage 6 files
* relevant tests

Pay particular attention to:

```text
src/scrapers/theodds_scraper.py
src/data/odds_quota.py
scripts/refresh_and_capture.py
scripts/capture_closing_lines.py
scripts/paper_trading_report.py
.github/workflows/
config/config.example.yaml
src/agent/betting_agent.py
src/data/models.py
src/evaluation/clv.py
src/models/model_version.py
```

Determine exactly what Stage 6 has already implemented.

Do not duplicate functionality.

Do not rewrite working code merely for style.

---

# 5. Fixture-ingestion reliability

Stage 6 discovered:

> DB contains zero future fixtures while the live API contains today's fixtures.

Investigate why.

Trace the complete fixture lifecycle:

```text
API-Football
    ↓
fixture scraper
    ↓
database
    ↓
prediction workflow
    ↓
Odds API selection
```

Determine:

1. Why today's fixtures were absent from the database.
2. Whether this is expected because the normal ingestion job simply has not run yet.
3. Whether the CI workflow actually runs fixture ingestion before prediction.
4. Whether the current schedule is appropriate for the new season.
5. Whether the system can recover automatically after an ingestion failure.
6. Whether stale fixture data can cause incorrect odds requests.
7. Whether duplicate fixture ingestion is prevented.

### Fix only genuine operational defects.

Do not redesign the fixture ingestion system unnecessarily.

Add tests for any defect you fix.

---

# 6. Make the odds refresh strategy operational

Stage 6 selected this strategy:

* approximately 120-minute window
* approximately 2-hour refresh cadence
* approximately 180-minute minimum refresh interval
* historical estimate ≈ 258 credits/month
* approximately 90% historical coverage

Before implementing anything:

## Recalculate the quota model from the actual current code.

Do not trust the previous report blindly.

Verify:

```text
credits = regions × markets × requests
```

Verify the exact current:

* regions
* markets
* leagues
* request grouping
* refresh frequency
* minimum interval
* kickoff window

Then simulate at least:

* last 30 days
* last 90 days
* representative European matchday
* worst historical matchday available

Report:

```text
estimated credits/month
maximum daily credits
average daily credits
remaining free-tier headroom
```

The design should target **meaningful headroom**, not merely staying below 500.

If the current strategy cannot maintain a safe margin, improve the scheduling logic without touching prediction logic.

---

# 7. Critical requirement: quota safety

The free Odds API quota is a hard constraint.

Implement safeguards so that the application cannot accidentally exhaust it.

There must be:

### A. Monthly budget guard

Before making an API request:

```text
current_month_usage + estimated_request_cost <= configured_monthly_budget
```

If not:

* do not make the request,
* log the reason,
* mark the refresh as skipped,
* continue safely.

### B. Per-run budget

Prevent a single workflow execution from consuming an unexpectedly large number of credits.

### C. Per-league minimum refresh interval

Do not refresh the same league repeatedly inside the configured cooldown.

### D. Idempotency

Repeated CI execution must not generate unnecessary API calls.

### E. Explicit instrumentation

Every refresh should be attributable to:

* date
* league
* request
* estimated credits
* actual credits if available
* reason
* result
* skipped reason

Do not create a new database migration unless absolutely necessary.

Prefer the existing `api_budget` infrastructure.

---

# 8. Make fixture selection demand-driven

The key principle should be:

> Do not refresh odds simply because a league has fixtures.

Refresh odds because there is a **paper prediction / relevant fixture that requires fresh odds**.

Inspect the current architecture carefully.

If it is possible without changing the model:

```text
future fixtures
      ↓
paper predictions / pending picks
      ↓
relevant league + match IDs
      ↓
odds refresh
```

Prefer demand-driven refresh over refreshing every configured league.

However:

**Do not assume demand-driven is automatically cheaper.**

Measure it.

Compare:

### Strategy A

Current daily league refresh.

### Strategy B

Stage-6 scheduled imminent refresh.

### Strategy C

Pending-pick-driven refresh.

For each calculate:

* credits/month
* expected CLV coverage
* fixture coverage
* worst-case daily credits
* API calls
* Supabase queries/egress

Select the strategy based on measured evidence.

Document the decision.

---

# 9. Closing-line capture

Verify that:

```text
prediction
    ↓
opening/taken odds
    ↓
late pre-kickoff refresh
    ↓
closing capture
    ↓
CLV
```

works end-to-end.

A valid closing line must satisfy all existing Stage-5 rules:

* correct match
* correct market
* correct selection
* valid decimal odds
* valid timestamp
* before kickoff
* inside the configured closing window
* valid bookmaker
* valid market structure
* plausible overround
* no corrupt source-market collision

Do NOT weaken any of these checks to increase CLV coverage.

If a closing line cannot be validated:

```text
status = missing / late / invalid
```

Never invent or substitute a price.

---

# 10. Test against real current-season data

Use read-only production inspection where appropriate.

Determine whether there are now:

* current-season fixtures
* current-season odds
* pending paper picks
* bookmakers
* supported markets

Do not create fake production bets.

Do not insert synthetic rows into production.

Use local SQLite/test fixtures for synthetic scenarios.

If live API testing is necessary, use the designated non-production/test API credentials where possible and explicitly account for the credit cost.

---

# 11. Enable paper trading safely

Do NOT blindly enable it immediately.

First verify:

* migrations present
* model version present
* paper flag works
* live ROI excludes paper picks
* settlement works
* CLV fields are immutable
* Claude review cannot overwrite the pre-review experiment record
* paper picks cannot accidentally become real-money picks
* health checks work
* rollback is simple

Then enable:

```yaml
betting:
  paper_trading_mode: true
```

Only if all safety checks pass.

If configuration is environment-specific, make the safest production mechanism available and document exactly how it is activated.

---

# 12. CI scheduling

Review all existing GitHub Actions.

Design a minimal schedule that supports:

### Fixture ingestion

Runs sufficiently early to populate upcoming fixtures.

### Prediction generation

Runs after fixture ingestion.

### Odds refresh

Runs only when needed and within quota.

### Closing capture

Runs shortly before kickoff.

### Settlement

Runs after matches finish.

### Reporting

Runs daily or after settlement.

Avoid creating multiple overlapping workflows if one workflow can orchestrate the sequence.

Avoid high-frequency schedules that consume unnecessary API credits.

All times must be UTC.

Document Bulgarian local-time equivalents only for human readability.

---

# 13. Concurrency and duplicate execution

GitHub Actions can overlap.

Verify that:

* two refresh jobs cannot simultaneously spend the same budget,
* two capture jobs cannot process the same pick incorrectly,
* duplicate predictions are prevented,
* duplicate odds requests are prevented where possible,
* concurrent settlement cannot corrupt state.

If an existing database/API-budget mechanism already handles this, test it rather than replacing it.

If a small lock/concurrency guard is required, implement the smallest safe solution.

---

# 14. Supabase egress constraint

The original reason for this optimisation project was Supabase egress.

Therefore audit every new Stage-6/7 query.

For each query introduced or modified:

* no `SELECT *`
* explicit columns
* no N+1 queries
* no unnecessary full-history scans
* proper indexes
* bounded result sets where appropriate

Measure representative egress for:

* paper prediction workflow
* odds refresh bookkeeping
* closing capture
* settlement
* paper report

Do not solve an Odds API quota problem by creating a Supabase egress problem.

---

# 15. Paper-trading isolation

This is critical.

Prove with tests that:

```text
is_paper = true
```

means:

* excluded from live ROI
* excluded from real-money bankroll calculations
* excluded from production betting actions
* included in experiment reports
* included in CLV statistics
* included in model-vs-market evaluation

Also verify that historical non-paper picks remain untouched.

---

# 16. Build an operational experiment dashboard/report

Improve `paper_trading_report.py` only where necessary.

It should clearly show:

### Operational

```text
fixtures ingested
paper predictions
odds refreshes
API credits used
API credits remaining
closing captures
missing captures
late captures
capture coverage
```

### Market

```text
market probability
model probability
model edge
taken odds
closing odds
CLV
```

### Performance

```text
paper ROI
Brier score
log-loss
calibration
beat-close %
mean CLV
median CLV
```

### Segments

At minimum:

* 1X2
* O/U 2.5
* BTTS
* odds buckets

Do not cherry-pick profitable leagues.

Do not introduce significance claims at low sample sizes.

---

# 17. Explicit experiment rules

Preserve the previously agreed checkpoints.

## First 100 valid closing lines

Purpose:

**DATA QUALITY ONLY**

Require:

* ≥80% valid closing capture coverage
* no captures after kickoff
* no market mismatch
* no selection mismatch
* no invented odds
* model version present
* reasonable bookmaker coverage

Do NOT conclude the model is profitable.

---

## 200 valid closing lines

Measure:

* mean CLV
* median CLV
* bootstrap CI
* % beating close
* model log-loss
* market log-loss
* model Brier
* market Brier
* calibration
* predicted edge vs realised outcome

Break down by market and odds bucket.

Do not change production configuration based solely on 200 picks.

---

## 500 valid closing lines

Perform the predefined decisive comparison:

```text
market-only
model-only
production blend
actual selected outcome
```

Use identical picks for comparisons.

Calculate:

* paired log-loss
* paired Brier
* CLV
* bootstrap confidence intervals
* permutation tests where appropriate
* market-specific results
* odds-bucket results

Do not optimise parameters after seeing the 500-pick result.

---

# 18. Prevent experiment contamination

This stage must preserve the integrity of the experiment.

Do NOT:

* change model parameters because a result looks bad,
* remove losing picks,
* change market filters after observing results,
* add a profitable bookmaker retrospectively,
* cherry-pick leagues,
* change the CLV definition after seeing outcomes,
* change the experiment window retrospectively,
* use settlement results to alter historical predictions.

Any discovered defect should be:

1. identified,
2. fixed prospectively,
3. documented,
4. assigned a new model/experiment version if it changes predictions.

---

# 19. Fix or document any discovered bugs

If you discover bugs during implementation:

Classify them:

### Critical

Could corrupt predictions, odds, CLV, experiment integrity, money, or production data.

→ Fix now.

### Operational

Prevents the experiment from running correctly.

→ Fix now.

### Non-critical

Does not affect the experiment.

→ Document for later.

### Model improvement

Could potentially improve predictive performance.

→ Do not implement. Document separately.

---

# 20. Required validation

Run the complete test suite.

Target:

```text
0 regressions
100% existing tests passing
all new tests passing
```

Add tests for:

* quota guard
* refresh cooldown
* duplicate workflow execution
* fixture-ingestion recovery
* paper/live isolation
* closing capture
* CLV
* concurrency
* budget exhaustion
* current-season fixture selection

If possible, run the full pipeline locally using mocks/fixtures.

---

# 21. Production safety

Before any production write:

Perform a preflight:

```text
DATABASE_URL points to intended project
API key is not a test key
migration state verified
paper mode verified
quota budget verified
```

Never allow tests to load production credentials through import side effects.

Maintain the Stage-5 safety invariant:

> importing a module must never silently restore production credentials.

Do not create test data in production.

Do not alter historical picks.

Do not modify settled records.

---

# 22. Git requirements

Do NOT commit automatically.

At the end provide:

### Working tree

List every changed/untracked file.

### Diff summary

Explain every meaningful change.

### Tests

```text
X passed
Y failed
Z skipped
```

### Production impact

State exactly what was read and what was written.

### API impact

State:

```text
requests made
credits consumed
remaining credits
estimated monthly usage
```

### Experiment state

State:

```text
paper mode
future fixtures
pending paper picks
valid CLV pairs
CLV coverage
model version
```

### Decision

Choose exactly one:

* `READY FOR PAPER TRADING`
* `BLOCKED — FIX REQUIRED`
* `READY FOR 100-PICK DATA COLLECTION`
* `INSUFFICIENT EVIDENCE`

Do not call the system ready for real-money betting.

---

# 23. Final report

Create:

`docs/stage7-operational-paper-trading-YYYY-MM-DD.md`

Include:

1. Executive summary
2. Stage-6 verification
3. Fixture-ingestion findings
4. Odds API quota model
5. Selected refresh strategy
6. Cost simulation
7. Closing-line capture validation
8. Paper-trading safety verification
9. CI scheduling
10. Supabase egress audit
11. Test results
12. Production verification
13. Experiment contamination safeguards
14. 100/200/500-pick measurement plan
15. Remaining blockers
16. Exact next steps

Also create, if model issues are discovered:

`docs/stage7-model-observations-YYYY-MM-DD.md`

Do not implement those model changes.

---

# 24. Most important principle

Do not confuse:

> "the system can now collect data"

with:

> "the model works."

The purpose of this stage is to make the experiment trustworthy.

The model remains frozen.

The bookmaker market remains the benchmark.

CLV is the primary prospective signal.

Paper trading remains mandatory.

No real-money betting should be recommended regardless of how good an individual short-term ROI looks.

Start with the audit. Make only evidence-backed operational fixes. Then run the full validation and stop for review before committing.
