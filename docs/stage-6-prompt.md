# Stage 6 — Free Odds API Optimization & Prospective CLV Collection

We have completed and locally committed the Stage 3–5 baseline.

Current commit:

`c0d3b7f` — `feat: establish stage 5 paper trading and CLV baseline`

Branch:

`stage5-paper-trading-clv-baseline`

Current verified state:

* 483 tests passing
* working tree clean
* nothing pushed
* nothing deployed
* production migrations 003/004/005 applied
* production rows unchanged: 1,026 total / 1,018 settled
* closing odds captured: 0
* CLV coverage: 0%
* model frozen at `stage5_baseline_20260807.ac04cc`
* paper trading currently disabled
* no real-money betting
* Stage 5 infrastructure is ready

Do NOT modify the model in this stage.

---

# PRIMARY OBJECTIVE

Build the smallest, safest and most quota-efficient system that can start collecting **real prospective closing-line data** using the existing free Odds API allowance.

The goal is NOT to improve prediction accuracy yet.

The goal is:

> Make the betting-agent capable of collecting trustworthy closing odds for paper-traded picks while staying safely within the free Odds API quota.

The system must eventually provide enough clean prospective data for the 100 / 200 / 500 pick CLV experiment defined in Stage 5.

---

# IMPORTANT EVIDENCE FROM STAGE 5

The current Odds API usage is not sustainable.

The API charges:

`1 credit × region × market`

The current request uses:

* region: `eu`
* markets: `h2h,totals`

Therefore:

`2 credits/request`

The current architecture can effectively request approximately:

`27 leagues × 2 credits = 54 credits/run`

At one run/day:

`~1,620 credits/month`

The free allowance is:

`500 credits/month`

Therefore the current architecture cannot be used unchanged.

There is also a second problem:

The current odds refresh runs around 09:37 UTC while many European matches kick off in the evening.

The resulting odds are 8–10 hours old and are correctly rejected by the CLV validator as not being a genuine closing snapshot.

Therefore:

> Under the current schedule, the experiment cannot produce valid CLV.

---

# STRICT SCOPE

You MAY modify:

* Odds API request scheduling
* Odds API fixture filtering
* Odds API quota management
* kickoff-window selection
* odds refresh orchestration
* closing-line capture scheduling
* related configuration
* related tests
* related documentation
* logging/metrics needed to monitor quota usage

You MAY NOT modify:

* Elo
* Poisson
* ML algorithms
* feature engineering
* model weights
* bookmaker blend weight
* expected-value calculation
* confidence calculation
* betting gates
* market_spec validation rules
* CLV formulas
* model_version logic
* paper-trading evaluation methodology
* pick selection logic

Do not "improve" the model as part of this stage.

Do not tune thresholds based on existing results.

Do not introduce a new data provider unless absolutely necessary and explicitly justified.

---

# PHASE 1 — AUDIT CURRENT ODDS INGESTION

Before changing anything, inspect the complete odds ingestion pipeline.

Trace:

1. where fixtures are discovered
2. where leagues are selected
3. how `_get_today_fixtures` works
4. how TheOddsScraper builds API requests
5. how regions are selected
6. how markets are selected
7. how often the workflow runs
8. how odds are persisted
9. how existing database odds are reused
10. how `capture_closing_lines.py` consumes them
11. how many requests are made per execution
12. how many Odds API credits each execution consumes

Do not assume the Stage 5 report is perfectly correct.

Verify the current implementation directly from code.

Produce a short baseline:

* requests/run
* credits/run
* expected requests/day
* expected credits/day
* expected credits/month
* current GitHub Actions schedules
* current number of configured leagues
* current fixture selection behaviour

---

# PHASE 2 — DESIGN THE QUOTA MODEL BEFORE CODING

Create a deterministic quota model.

The system must account for:

* 500 credits/month free allowance
* current remaining allowance
* number of markets requested
* number of regions requested
* number of leagues requested
* number of fixtures
* repeated runs
* weekends
* midweek
* European competitions
* days with unusually high fixture volume

Do NOT simply target exactly 500.

Establish a safety margin.

Target:

**≤ 400 credits/month under normal expected operation**

and ideally remain below 500 even under reasonable fixture-volume spikes.

If this cannot be guaranteed, explain why before implementing.

---

# PHASE 3 — IMPLEMENT IMMINENT-FIXTURE REFRESH

The key change should be:

> Do not refresh all leagues once per day.

Instead:

> Refresh only leagues containing fixtures whose kickoff is approaching.

The preferred window should be configurable.

Example:

`odds_refresh_window_minutes: 90`

But do not blindly use this value.

Analyse whether the existing Odds API endpoint returns sufficient upcoming fixtures and whether a slightly larger window is required to ensure coverage.

The important property is:

* odds must be fresh enough to qualify as a closing snapshot
* API calls must remain within quota

The odds refresh should identify only relevant leagues.

For example:

If only 3 leagues have fixtures in the next 90 minutes:

3 league requests × 2 credits = 6 credits

rather than:

27 × 2 = 54 credits.

---

# PHASE 4 — AVOID DUPLICATE API CALLS

Implement protection against unnecessary repeated calls.

If the same league has already been refreshed recently and no new fixture requires another refresh, do not call the API again.

Add configurable freshness:

`odds_refresh_min_interval_minutes`

For example:

60 minutes.

But choose the value based on the actual workflow and CLV requirements.

The logic should be based on:

* league
* fixture kickoff
* last successful refresh
* whether there are pending paper picks requiring closing capture

Do not blindly refresh every league every hour.

---

# PHASE 5 — PRIORITISE PENDING PAPER PICKS

The system should give priority to fixtures that already have a saved paper pick.

The ideal hierarchy is:

1. pending paper picks approaching kickoff
2. fixtures needed for closing-line capture
3. other imminent fixtures that are part of the configured betting universe

This prevents spending free API credits collecting odds that will never be used.

However:

Do not change which picks the model generates.

Only change which odds are refreshed.

---

# PHASE 6 — QUOTA BUDGET GUARD

Implement a hard quota guard.

The application must know approximately how many credits it is allowed to consume.

Before making an API request:

* estimate its credit cost
* compare it with the configured monthly budget
* refuse the request if it would exceed the safety budget
* log why it was skipped

Example configuration:

```yaml
odds_api:
  monthly_credit_budget: 400
  safety_margin_credits: 100
```

Do not hard-code these values if the project already has a better configuration system.

The important requirement is that exceeding the budget must fail safely.

The system must NEVER silently exceed the configured budget.

---

# PHASE 7 — REQUEST MINIMAL MARKETS

Audit whether every currently requested market is actually required for:

* prediction
* paper trading
* CLV
* settlement
* reporting

If a market is not required, do not request it.

However:

Do NOT remove `h2h` or `totals` merely because they appear expensive.

Prove from code that a market is unnecessary before removing it.

For each market document:

* why it is requested
* where it is consumed
* whether CLV requires it
* whether prediction requires it
* whether removing it changes behaviour

Do not change market semantics.

---

# PHASE 8 — CLOSING-LINE WORKFLOW

Make the workflow capable of:

1. identify imminent fixtures
2. refresh their odds
3. persist the fresh odds
4. immediately run closing-line capture
5. validate bookmaker/market/selection
6. store valid closing odds
7. calculate CLV later after settlement

The closing capture must remain the authority for whether an odds snapshot is valid.

Do NOT weaken:

* timestamp validation
* market validation
* bookmaker validation
* overround validation
* selection mapping
* CLV validity rules

Never make an old price "valid" just to increase CLV coverage.

---

# PHASE 9 — SCHEDULING

Inspect the current GitHub Actions workflows.

Design the smallest schedule capable of collecting genuine closing lines.

Do not blindly create hourly jobs for every day.

The preferred design is a lightweight scheduled workflow that:

1. discovers imminent fixtures
2. determines whether a refresh is needed
3. spends credits only when necessary
4. refreshes the relevant leagues
5. runs closing capture

All timestamps must remain UTC.

Do not introduce local-time assumptions.

Document the Bulgaria UTC+2/UTC+3 implication only as operational documentation.

---

# PHASE 10 — IDEMPOTENCY

The new workflow must be safe if:

* GitHub Actions runs twice
* the same league is discovered twice
* a request times out
* the process crashes after the API call
* capture runs twice
* a fixture moves
* a match is postponed

Do not create duplicate odds records unnecessarily.

Do not overwrite historical odds that are required for the experiment.

Do not change the meaning of existing odds records.

---

# PHASE 11 — SUPABASE EGRESS

This is critical.

We previously had a major Supabase egress problem.

Therefore inspect every new query introduced by Stage 6.

Requirements:

* no `SELECT *`
* no unnecessary columns
* no N+1 queries
* no full-history scans when an imminent-fixture query is sufficient
* use existing indexes where possible
* project only required columns

Measure approximate result size where practical.

Do not solve Odds API quota by creating a Supabase egress problem.

---

# PHASE 12 — TESTING

Add comprehensive tests.

At minimum test:

### Quota

* budget below limit → request allowed
* request would exceed budget → request blocked
* safety margin respected
* multiple requests accumulate correctly
* month boundary handled correctly

### Fixture selection

* fixture inside closing window → included
* fixture outside window → excluded
* postponed fixture → handled safely
* completed fixture → excluded
* multiple leagues → only relevant leagues requested

### Refresh deduplication

* recently refreshed league → skipped
* stale league → refreshed
* pending paper pick → prioritised
* duplicate workflow execution → no duplicate refresh

### Closing capture

* fresh odds → valid
* stale odds → invalid
* post-kickoff odds → invalid
* corrupt bookmaker → excluded
* valid bookmaker → retained
* multiple bookmakers → clean consensus
* market mismatch → rejected

### Safety

* no API call when budget exhausted
* no production DB writes from module imports
* no credentials in logs
* no secrets committed

---

# PHASE 13 — BACKTEST / HISTORICAL SIMULATION OF QUOTA

Before declaring Stage 6 complete, build a simulation using the available historical fixture data.

Simulate at least:

* normal weekday
* normal weekend
* high-volume weekend
* European competition day

Calculate:

* number of API requests
* credits consumed
* fixtures refreshed
* paper picks covered
* estimated CLV capture coverage

The result must demonstrate that the design is viable within the free tier.

---

# PHASE 14 — DO NOT ENABLE REAL-MONEY BETTING

Paper trading only.

Do not set:

```yaml
paper_trading_mode: false
```

Do not enable live betting.

Do not claim the model has an edge.

The purpose of this stage is measurement infrastructure.

---

# PHASE 15 — FIRST PROSPECTIVE RUN

After implementation, perform a dry run first.

Show:

* imminent fixtures
* leagues selected
* API requests that WOULD be made
* estimated credits
* skipped leagues
* reason for each skip
* pending paper picks covered
* expected closing captures

Do NOT spend API credits during the dry run.

Only after the dry run is internally consistent may you perform a real API call.

If a real API call is needed for validation, use the smallest possible number of requests.

Report the exact credits consumed if the API exposes that information.

---

# PHASE 16 — ACCEPTANCE CRITERIA

Stage 6 is successful only if all are true:

1. Full test suite passes.
2. No existing tests regress.
3. Odds API usage is demonstrably compatible with the free tier.
4. The system does not request all 27 leagues unnecessarily.
5. Imminent fixtures are prioritised.
6. Pending paper picks are prioritised.
7. Duplicate refreshes are prevented.
8. Monthly quota guard exists.
9. Supabase queries remain egress-efficient.
10. Closing-line validation remains strict.
11. No model behaviour changes.
12. No betting gates are changed.
13. No CLV formula is changed.
14. Paper trading remains the only operating mode.
15. A dry-run demonstrates the expected request/credit behaviour.
16. Historical simulation demonstrates the expected monthly quota usage.

---

# PHASE 17 — REPORT

Create:

`docs/stage6-odds-api-optimization-YYYY-MM-DD.md`

Include:

## Before

* requests/day
* credits/day
* credits/month
* CLV coverage capability

## After

* requests/day
* credits/day
* projected credits/month
* quota safety margin
* expected CLV coverage

## Changes

List every changed file and why.

## Tests

Report:

* total tests
* new tests
* failures
* regressions

## Quota simulation

Show normal and worst reasonable scenarios.

## CLV readiness

State clearly:

* whether genuine closing lines can now be collected
* expected capture window
* expected coverage
* remaining blockers

## Model integrity

Explicitly confirm:

* no model parameters changed
* no feature engineering changed
* no betting thresholds changed
* no gates changed
* no model_version semantics changed

## Final decision

Choose exactly one:

* `READY TO START PROSPECTIVE PAPER EXPERIMENT`
* `NOT READY — BLOCKERS REMAIN`

Do not call the system profitable.

---

# GIT / COMMIT RULE

Do NOT commit automatically.

At the end:

* show `git diff --stat`
* show `git status`
* list every modified/untracked file
* report test results
* report quota simulation

Then STOP and wait for my review.

Do NOT push.

Do NOT merge to main.

Do NOT deploy.

Do NOT apply new production migrations unless absolutely necessary and explicitly approved.

---

# MOST IMPORTANT RULE

Do not optimise for the appearance of CLV coverage.

Optimise for **truthful, prospective, reproducible measurement**.

A lower number of valid closing lines is preferable to a larger number of invalid ones.

If you discover that the free Odds API cannot provide sufficient coverage under these constraints, STOP and report the limitation instead of weakening the validation rules.
